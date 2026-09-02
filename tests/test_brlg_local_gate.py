from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.v_single.local_residual_gate import (
    BoundedLocalResidualGate,
)

from _v14_utils import compact_v14_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


def _gate_inputs(batch_size: int = 2, dim: int = 8):
    generator = torch.Generator().manual_seed(21)
    h_main = torch.randn(batch_size, dim, 3, 16, 12, generator=generator)
    delta = torch.randn(batch_size, 2, 3, 16, 12, generator=generator)
    x_base = torch.randn(batch_size, 2, 3, 16, 12, generator=generator)
    x_ctf = x_base + torch.randn(x_base.shape, generator=generator) * 0.2
    mask = (torch.rand(batch_size, 1, 3, 16, 12, generator=generator) > 0.4).float()
    x_obs = torch.randn(x_base.shape, generator=generator) * mask
    alpha = torch.full((batch_size, 1, 1, 1, 1), 0.4)
    return alpha, h_main, delta, x_ctf, x_base, x_obs, mask


class BRLGLocalGateTest(unittest.TestCase):
    def test_zero_initialized_regional_gate_exactly_matches_v14(self) -> None:
        base_cfg = compact_v14_config()
        brlg_cfg = deep_update(base_cfg, {
            "model": {"v14": {
                "local_final_gate_mode": "regional",
                "local_gate_max_relative_delta": 0.2,
            }}
        })
        batch = make_batch(base_cfg)
        torch.manual_seed(2026)
        base = DualBranchSTImputer.from_config(base_cfg).eval()
        torch.manual_seed(2026)
        brlg = DualBranchSTImputer.from_config(brlg_cfg).eval()

        # Make the shared correction non-zero so equality actually exercises
        # the new modulation rather than relying on V14's safe zero start.
        for model in (base, brlg):
            layer = model.main_branch.refiner.correction_adapter[-1]
            nn_value = 0.01
            layer.weight.data.fill_(nn_value)
            layer.bias.data.fill_(nn_value)
        with torch.no_grad():
            expected = base(batch)
            actual = brlg(batch)
        torch.testing.assert_close(
            actual["x_hat_final"], expected["x_hat_final"], atol=0.0, rtol=0.0
        )
        modulation = actual["diagnostics"]["v14"]["local_gate_modulation"]
        torch.testing.assert_close(modulation, torch.ones_like(modulation))

    def test_temporal_and_regional_shapes_and_bounds(self) -> None:
        inputs = _gate_inputs()
        for mode, expected_shape in (
            ("temporal", (2, 1, 3, 1, 1)),
            ("regional", (2, 1, 3, 4, 3)),
        ):
            with self.subTest(mode=mode):
                gate = BoundedLocalResidualGate(
                    feature_dim=8,
                    hidden_dim=8,
                    mode=mode,
                    max_relative_delta=0.2,
                    spatial_divisor=4,
                ).eval()
                gate.net[-1].bias.data.fill_(10.0)
                with torch.no_grad():
                    alpha, diagnostics = gate(*inputs, alpha_max=0.5)
                self.assertEqual(
                    tuple(diagnostics["local_gate_logits"].shape), expected_shape
                )
                self.assertTrue(torch.all(alpha >= 0.4 * 0.8 - 1e-6))
                self.assertTrue(torch.all(alpha <= 0.4 * 1.2 + 1e-6))
                self.assertTrue(torch.all(alpha <= 0.5))

    def test_hidden_ground_truth_cannot_change_brlg_prediction(self) -> None:
        cfg = deep_update(compact_v14_config(), {
            "model": {"v14": {
                "local_final_gate_mode": "regional",
                "local_gate_max_relative_delta": 0.2,
            }}
        })
        model = DualBranchSTImputer.from_config(cfg).eval()
        first_batch = make_batch(cfg)
        second_batch = {key: value.clone() for key, value in first_batch.items()}
        missing = (1.0 - second_batch["m_f"]).expand_as(second_batch["x_f_gt"])
        second_batch["x_f_gt"] = second_batch["x_f_gt"] + 1000.0 * missing
        with torch.no_grad():
            first = model(first_batch)
            second = model(second_batch)
        torch.testing.assert_close(
            first["x_hat_final"], second["x_hat_final"], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            first["diagnostics"]["v14"]["local_gate_modulation"],
            second["diagnostics"]["v14"]["local_gate_modulation"],
            atol=0.0,
            rtol=0.0,
        )

    def test_all_six_candidate_configs_build_and_forward(self) -> None:
        paths = sorted((ROOT / "configs/v14-exploration/brlg").glob("B*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(candidate=path.stem):
                patch = json.loads(path.read_text(encoding="utf-8"))
                cfg = deep_update(compact_v14_config(), patch)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    output = model(make_batch(cfg))
                self.assertTrue(torch.isfinite(output["x_hat_final"]).all())
                modulation = output["diagnostics"]["v14"][
                    "local_gate_modulation"
                ]
                self.assertTrue(torch.isfinite(modulation).all())
                torch.testing.assert_close(modulation, torch.ones_like(modulation))

    def test_invalid_or_confounded_configs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoundedLocalResidualGate(feature_dim=8, mode="pixel")
        with self.assertRaises(ValueError):
            BoundedLocalResidualGate(feature_dim=8, max_relative_delta=1.0)
        cfg = compact_v14_config()
        cfg["model"]["v14"].update({
            "local_final_gate_mode": "regional",
            "channel_final_gate": True,
        })
        with self.assertRaisesRegex(ValueError, "separate structural candidates"):
            DualBranchSTImputer.from_config(cfg)


if __name__ == "__main__":
    unittest.main()
