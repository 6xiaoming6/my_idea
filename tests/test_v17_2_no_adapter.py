from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint

from _v17_utils import compact_v17_config, make_batch


def v17_2_config(**model_updates) -> dict:
    return deep_update(
        compact_v17_config(),
        {
            "model": {
                "version": "v17.2-single",
                "architecture": "v17_2_no_adapter_hierarchical_scale_moe",
                "v17": {
                    "adapter_enabled": False,
                    **model_updates,
                },
                "v17_2": {
                    "enabled": True,
                    "source_ablation": "E1_no_adapter",
                    "remove_scale_adapter": True,
                },
            }
        },
    )


class V172NoAdapterTest(unittest.TestCase):
    def test_registry_forces_no_adapter_even_if_config_is_wrong(self) -> None:
        cfg = v17_2_config(adapter_enabled=True)
        model = DualBranchSTImputer.from_config(cfg)
        self.assertFalse(model.main_branch.adapter_enabled)
        adapter_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if any(
                token in name
                for token in (
                    "main_branch.adapter_f.",
                    "main_branch.adapter_m.",
                    "main_branch.adapter_c.",
                )
            )
        ]
        self.assertEqual(adapter_parameters, [])
        self.assertEqual(model.main_branch.adapter_parameter_count, 0)

    def test_v17_2_is_numerically_equivalent_to_e1(self) -> None:
        e1_cfg = deep_update(
            compact_v17_config(),
            {"model": {"v17": {"adapter_enabled": False}}},
        )
        candidate_cfg = v17_2_config()
        torch.manual_seed(1702)
        e1 = DualBranchSTImputer.from_config(e1_cfg).eval()
        torch.manual_seed(1702)
        candidate = DualBranchSTImputer.from_config(candidate_cfg).eval()
        candidate.load_state_dict(e1.state_dict(), strict=True)
        batch = make_batch(candidate_cfg, seed=19)
        with torch.no_grad():
            left = e1(batch)
            right = candidate(batch)
        for key in ("x_hat_main", "x_hat_final"):
            torch.testing.assert_close(left[key], right[key], atol=1e-6, rtol=1e-5)
        for key in ("fine", "mid", "coarse", "scale_gate", "route_branch_gate"):
            torch.testing.assert_close(
                left["gates"][key],
                right["gates"][key],
                atol=1e-6,
                rtol=1e-5,
            )
        torch.testing.assert_close(
            left["features"]["h_main"],
            right["features"]["h_main"],
            atol=1e-6,
            rtol=1e-5,
        )

    def test_exact_parameter_reduction_is_6384(self) -> None:
        common = {
            "model": {
                "main": {
                    "dim": 64,
                    "num_experts": 2,
                    "top_k": 1,
                    "num_groups": 8,
                },
                "v17": {"adapter_dim": 16},
            }
        }
        full_cfg = deep_update(compact_v17_config(), common)
        candidate_cfg = deep_update(v17_2_config(adapter_dim=16), common)
        full = DualBranchSTImputer.from_config(full_cfg)
        candidate = DualBranchSTImputer.from_config(candidate_cfg)
        full_count = sum(parameter.numel() for parameter in full.parameters())
        candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(full_count - candidate_count, 6384)

    def test_outputs_and_identity_diagnostics_are_finite(self) -> None:
        cfg = v17_2_config()
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            outputs = model(make_batch(cfg))
        self.assertTrue(outputs["v17_2_enabled"])
        self.assertTrue(outputs["v17_2_remove_adapter"])
        self.assertEqual(outputs["v17_2_source_ablation"], "E1_no_adapter")
        self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())
        self.assertEqual(
            outputs["diagnostics"]["v17_2"],
            {"adapter_enabled": False, "adapter_parameter_count": 0},
        )
        for group in ("adapter_delta_rms", "adapter_relative_rms"):
            for value in outputs["diagnostics"]["v17"][group].values():
                torch.testing.assert_close(value, torch.zeros_like(value))

    def test_supported_dataset_shapes(self) -> None:
        cases = (
            (2, 12, 32, 32, "fine_mid"),
            (2, 12, 24, 12, "fine_mid_coarse"),
            (1, 7, 32, 32, "fine_mid_coarse"),
        )
        for channels, time_steps, height, width, scale_mode in cases:
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v17_config(
                    channels=channels,
                    time_steps=time_steps,
                    height=height,
                    width=width,
                    scale_mode=scale_mode,
                )
                cfg = deep_update(
                    cfg,
                    {
                        "model": {
                            "version": "v17.2-single",
                            "architecture": (
                                "v17_2_no_adapter_hierarchical_scale_moe"
                            ),
                            "v17": {"adapter_enabled": False},
                            "v17_2": {
                                "enabled": True,
                                "remove_scale_adapter": True,
                            },
                        }
                    },
                )
                model = DualBranchSTImputer.from_config(cfg).eval()
                batch = make_batch(cfg)
                with torch.no_grad():
                    outputs = model(batch)
                self.assertEqual(
                    tuple(outputs["x_hat_main"].shape),
                    (1, channels, time_steps, height, width),
                )
                if scale_mode == "fine_mid":
                    torch.testing.assert_close(
                        outputs["gates"]["scale_gate"][:, 2],
                        torch.zeros(1),
                    )

    def test_hidden_target_changes_do_not_change_forward_prediction(self) -> None:
        cfg = v17_2_config()
        model = DualBranchSTImputer.from_config(cfg).eval()
        original = make_batch(cfg, seed=31)
        changed = {key: value.clone() for key, value in original.items()}
        hidden = 1.0 - changed["m_f"]
        changed["x_f_gt"] = changed["x_f_gt"] + hidden * 1000.0
        with torch.no_grad():
            left = model(original)
            right = model(changed)
        torch.testing.assert_close(left["x_hat_main"], right["x_hat_main"])
        for key in ("fine", "mid", "coarse", "scale_gate", "route_branch_gate"):
            torch.testing.assert_close(left["gates"][key], right["gates"][key])

    def test_best_checkpoint_round_trip(self) -> None:
        cfg = v17_2_config()
        model = DualBranchSTImputer.from_config(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, 3, {"val_mae": 1.0}, cfg)
            restored = DualBranchSTImputer.from_config(cfg)
            checkpoint = load_checkpoint(path, restored)
            self.assertEqual(checkpoint["epoch"], 3)
            for left, right in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
