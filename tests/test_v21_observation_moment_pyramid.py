from __future__ import annotations

import copy
import unittest

import torch

from stmoe_imputer.data.transforms import (
    ensure_multiscale,
    masked_pool2d_spatial,
    observation_moment_pool2d_spatial,
)
from stmoe_imputer.models import DualBranchSTImputer

from _v14_utils import compact_v14_config, make_batch


class ObservationMomentPyramidTest(unittest.TestCase):
    def test_legacy_mode_is_the_original_hierarchical_pooling(self) -> None:
        generator = torch.Generator().manual_seed(21)
        target = torch.randn((1, 2, 3, 16, 16), generator=generator)
        mask = (torch.rand((1, 1, 3, 16, 16), generator=generator) > 0.55).float()
        actual = ensure_multiscale(
            {"x_f_gt": target, "m_f": mask},
            fine_to_mid=2,
            fine_to_coarse=4,
            pyramid_mode="legacy",
        )
        expected_mid, expected_mid_mask, _ = masked_pool2d_spatial(
            target * mask, mask, kernel_size=2, return_reliability=True
        )
        expected_coarse, expected_coarse_mask, _ = masked_pool2d_spatial(
            expected_mid, expected_mid_mask, kernel_size=2, return_reliability=True
        )
        torch.testing.assert_close(actual["x_m_obs"], expected_mid, atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual["x_c_obs"], expected_coarse, atol=0.0, rtol=0.0)

    def test_content_is_direct_observed_sum_over_count(self) -> None:
        target = torch.arange(1.0, 17.0).reshape(1, 1, 1, 4, 4)
        mask = torch.tensor(
            [[[[[1, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 1], [0, 0, 1, 0]]]]],
            dtype=torch.float32,
        )
        content, valid, reliability, evidence = observation_moment_pool2d_spatial(
            target, mask, kernel_size=4
        )
        expected = (target * mask).sum() / mask.sum()
        torch.testing.assert_close(content.squeeze(), expected)
        self.assertEqual(valid.item(), 1.0)
        self.assertEqual(reliability.item(), mask.mean().item())
        self.assertEqual(evidence.shape, (1, 7, 1, 1, 1))

    def test_empty_pool_has_zero_finite_content_and_evidence(self) -> None:
        target = torch.randn((1, 2, 2, 4, 4))
        mask = torch.zeros((1, 1, 2, 4, 4))
        content, valid, reliability, evidence = observation_moment_pool2d_spatial(
            target, mask, kernel_size=2
        )
        self.assertEqual(torch.count_nonzero(content), 0)
        self.assertEqual(torch.count_nonzero(valid), 0)
        self.assertEqual(torch.count_nonzero(reliability), 0)
        self.assertEqual(torch.count_nonzero(evidence), 0)
        self.assertTrue(torch.isfinite(evidence).all())

    def test_geometry_separates_equal_mean_and_coverage_masks(self) -> None:
        target = torch.full((1, 1, 1, 2, 2), 10.0)
        left = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])
        top = torch.tensor([[[[[1.0, 1.0], [0.0, 0.0]]]]])
        left_content, _, _, left_evidence = observation_moment_pool2d_spatial(
            target, left, kernel_size=2
        )
        top_content, _, _, top_evidence = observation_moment_pool2d_spatial(
            target, top, kernel_size=2
        )
        torch.testing.assert_close(left_content, top_content)
        torch.testing.assert_close(left_evidence[:, 0], top_evidence[:, 0])
        self.assertNotEqual(left_evidence[:, 2].item(), top_evidence[:, 2].item())
        self.assertNotEqual(left_evidence[:, 3].item(), top_evidence[:, 3].item())

    def test_hidden_ground_truth_does_not_change_dual_state(self) -> None:
        generator = torch.Generator().manual_seed(22)
        target = torch.randn((1, 2, 2, 8, 8), generator=generator)
        mask = (torch.rand((1, 1, 2, 8, 8), generator=generator) > 0.5).float()
        changed = target + (1.0 - mask) * 100000.0
        first = ensure_multiscale(
            {"x_f_gt": target, "m_f": mask}, pyramid_mode="observation_moment"
        )
        second = ensure_multiscale(
            {"x_f_gt": changed, "m_f": mask}, pyramid_mode="observation_moment"
        )
        for key in ("x_m_obs", "x_c_obs", "e_f", "e_m", "e_c"):
            torch.testing.assert_close(first[key], second[key], atol=0.0, rtol=0.0)

    def test_v21_forward_is_finite_and_evidence_path_gets_gradient(self) -> None:
        cfg = compact_v14_config(channels=2, time_steps=3, height=16, width=16)
        cfg["data"]["scales"]["pyramid_mode"] = "observation_moment"
        cfg["model"]["main"]["evidence_dim"] = 7
        cfg["model"]["main"]["evidence_zero_init"] = True
        legacy_batch = make_batch(cfg, seed=23)
        batch = ensure_multiscale(
            {"x_f_gt": legacy_batch["x_f_gt"], "m_f": legacy_batch["m_f"]},
            fine_to_mid=2,
            fine_to_coarse=4,
            pyramid_mode="observation_moment",
        )
        model = DualBranchSTImputer.from_config(cfg).eval()

        no_evidence = copy.copy(batch)
        for key in ("e_f", "e_m", "e_c"):
            no_evidence.pop(key)
        with torch.no_grad():
            with_evidence_output = model(batch)["x_hat_final"]
            without_evidence_output = model(no_evidence)["x_hat_final"]
        torch.testing.assert_close(
            with_evidence_output, without_evidence_output, atol=0.0, rtol=0.0
        )
        self.assertTrue(torch.isfinite(with_evidence_output).all())

        model.train()
        model(batch)["x_hat_final"].square().mean().backward()
        gradient = model.main_branch.main_backbone.embed_m.evidence_embed.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_optional_evidence_module_does_not_shift_common_initialization(self) -> None:
        legacy_cfg = compact_v14_config()
        v21_cfg = copy.deepcopy(legacy_cfg)
        v21_cfg["model"]["main"]["evidence_dim"] = 7
        torch.manual_seed(25)
        legacy = DualBranchSTImputer.from_config(legacy_cfg)
        torch.manual_seed(25)
        v21 = DualBranchSTImputer.from_config(v21_cfg)
        legacy_state = legacy.state_dict()
        v21_state = v21.state_dict()
        common_keys = set(legacy_state) & set(v21_state)
        self.assertGreater(len(common_keys), 0)
        for key in common_keys:
            torch.testing.assert_close(
                legacy_state[key], v21_state[key], atol=0.0, rtol=0.0, msg=key
            )

    def test_v21_forward_supports_all_three_dataset_geometries(self) -> None:
        for channels, time_steps, height, width in (
            (2, 12, 32, 32),
            (2, 12, 24, 12),
            (1, 7, 32, 32),
        ):
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v14_config(channels, time_steps, height, width)
                cfg["model"]["main"]["evidence_dim"] = 7
                legacy_batch = make_batch(cfg, seed=24)
                batch = ensure_multiscale(
                    {
                        "x_f_gt": legacy_batch["x_f_gt"],
                        "m_f": legacy_batch["m_f"],
                    },
                    pyramid_mode="observation_moment",
                )
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    output = model(batch)["x_hat_final"]
                self.assertEqual(output.shape, batch["x_f_gt"].shape)
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
