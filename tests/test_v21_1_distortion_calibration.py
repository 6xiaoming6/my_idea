from __future__ import annotations

import copy
import unittest

import torch

from stmoe_imputer.data.transforms import ensure_multiscale
from stmoe_imputer.models import DualBranchSTImputer

from _v14_utils import compact_v14_config


def v21_1_config(use_explicit_distortion: bool = True) -> dict:
    cfg = compact_v14_config(channels=2, time_steps=3, height=16, width=16)
    cfg["data"]["scales"]["pyramid_mode"] = "dual_observation_moment"
    cfg["model"]["architecture"] = "v21_distortion_calibrated_moe"
    cfg["model"]["main"]["evidence_dim"] = 0
    cfg["model"]["v21_1"] = {
        "use_explicit_distortion": use_explicit_distortion,
        "evidence_dim": 7,
        "gate_hidden": 8,
        "alpha_max": 0.5,
        "alpha_init": 0.01,
        "module_seed": 2111,
    }
    return cfg


def dual_batch(cfg: dict, seed: int = 211) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    syn = cfg["data"]["synthetic"]
    shape = (1, cfg["model"]["c_in"], syn["t"], syn["h"], syn["w"])
    target = torch.randn(shape, generator=generator)
    mask = (torch.rand((shape[0], 1, *shape[2:]), generator=generator) > 0.55).float()
    return ensure_multiscale(
        {"x_f_gt": target, "m_f": mask},
        fine_to_mid=cfg["data"]["scales"]["fine_to_mid"],
        fine_to_coarse=cfg["data"]["scales"]["fine_to_coarse"],
        pyramid_mode="dual_observation_moment",
    )


class V21DistortionCalibrationTest(unittest.TestCase):
    def test_dual_mode_preserves_both_existing_pyramid_contracts(self) -> None:
        cfg = v21_1_config()
        batch = dual_batch(cfg)
        legacy = ensure_multiscale(
            {"x_f_gt": batch["x_f_gt"], "m_f": batch["m_f"]},
            pyramid_mode="legacy",
        )
        measure = ensure_multiscale(
            {"x_f_gt": batch["x_f_gt"], "m_f": batch["m_f"]},
            pyramid_mode="observation_moment",
        )
        for scale in ("m", "c"):
            torch.testing.assert_close(
                batch[f"x_{scale}_obs"], legacy[f"x_{scale}_obs"], atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                batch[f"x_{scale}_measure"], measure[f"x_{scale}_obs"], atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                batch[f"r_{scale}_measure"], measure[f"r_{scale}"], atol=0.0, rtol=0.0
            )

    def test_hidden_truth_does_not_change_dual_pyramid(self) -> None:
        cfg = v21_1_config()
        first = dual_batch(cfg)
        changed_target = first["x_f_gt"] + (1.0 - first["m_f"]) * 100000.0
        second = ensure_multiscale(
            {"x_f_gt": changed_target, "m_f": first["m_f"]},
            pyramid_mode="dual_observation_moment",
        )
        for key in (
            "x_m_obs",
            "x_c_obs",
            "x_m_measure",
            "x_c_measure",
            "e_m",
            "e_c",
        ):
            torch.testing.assert_close(first[key], second[key], atol=0.0, rtol=0.0)

    def test_gate_is_bounded_anchor_initialized_and_trainable(self) -> None:
        cfg = v21_1_config()
        batch = dual_batch(cfg)
        model = DualBranchSTImputer.from_config(cfg)
        output = model(batch)
        features = output["features"]["v21_1"]
        torch.testing.assert_close(
            features["alpha_mid"], torch.zeros_like(features["alpha_mid"]), atol=0.0, rtol=0.0
        )
        alpha = features["alpha_coarse"]
        torch.testing.assert_close(alpha, torch.full_like(alpha, 0.01), atol=1e-6, rtol=0.0)
        self.assertTrue((alpha >= 0.0).all())
        self.assertTrue((alpha <= 0.5).all())
        self.assertTrue(torch.isfinite(output["x_hat_main"]).all())

        output["x_hat_main"].square().mean().backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "acceptance" in name and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_p3_and_p4_match_at_controlled_initialization(self) -> None:
        p3_cfg = v21_1_config(True)
        p4_cfg = v21_1_config(False)
        batch = dual_batch(p3_cfg)
        torch.manual_seed(212)
        p3 = DualBranchSTImputer.from_config(p3_cfg).eval()
        torch.manual_seed(212)
        p4 = DualBranchSTImputer.from_config(p4_cfg).eval()
        with torch.no_grad():
            p3_output = p3(batch)["x_hat_main"]
            p4_output = p4(batch)["x_hat_main"]
        torch.testing.assert_close(p3_output, p4_output, atol=0.0, rtol=0.0)

    def test_v14_common_parameters_keep_identical_initialization(self) -> None:
        v14_cfg = compact_v14_config()
        candidate_cfg = v21_1_config()
        torch.manual_seed(213)
        v14 = DualBranchSTImputer.from_config(v14_cfg)
        torch.manual_seed(213)
        candidate = DualBranchSTImputer.from_config(candidate_cfg)
        reference = v14.state_dict()
        actual = candidate.state_dict()
        common = set(reference) & set(actual)
        self.assertGreater(len(common), 0)
        for key in common:
            torch.testing.assert_close(reference[key], actual[key], atol=0.0, rtol=0.0, msg=key)

    def test_missing_dual_keys_raise_an_actionable_error(self) -> None:
        cfg = v21_1_config()
        batch = dual_batch(cfg)
        broken = copy.copy(batch)
        broken.pop("x_c_measure")
        model = DualBranchSTImputer.from_config(cfg)
        with self.assertRaisesRegex(ValueError, "dual_observation_moment"):
            model(broken)


if __name__ == "__main__":
    unittest.main()
