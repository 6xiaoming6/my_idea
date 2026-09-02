from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.v_single import V14SafeC2FMoE
from stmoe_imputer.models.v_single.safety_controller import SafetyController

from _v14_utils import compact_v14_config, make_batch


class V14StructureCandidateTest(unittest.TestCase):
    def test_monotonic_consistency_gate_orders_observed_advantage(self) -> None:
        controller = SafetyController(
            precondition_dim=4,
            consistency_dim=5,
            dropout=0.0,
            final_gate_mode="monotonic_consistency",
            monotonic_advantage_gain=2.0,
            monotonic_context_bound=0.5,
        ).eval()
        condition = torch.zeros(2, 4)
        # Columns: base error, CTF error, CTF-base, delta mean, q95.
        consistency = torch.tensor([
            [1.0, 2.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, -1.0, 1.0, 1.0],
        ])
        with torch.no_grad():
            alpha, diagnostics = controller.final_gate(condition, consistency)
        self.assertLess(float(alpha[0]), float(alpha[1]))
        self.assertLess(
            float(diagnostics["relative_observed_advantage"][0]),
            float(diagnostics["relative_observed_advantage"][1]),
        )

    def test_monotonic_gate_rejects_channel_candidate_combination(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate single-variable"):
            SafetyController(
                precondition_dim=4,
                c_out=2,
                channel_final_gate=True,
                final_gate_mode="monotonic_consistency",
            )

    def test_identifiable_residual_is_invariant_to_positive_raw_scale(self) -> None:
        delta = torch.randn(2, 2, 3, 4, 4)
        alpha = torch.full((2, 1, 1, 1, 1), 0.1)
        observed = torch.randn_like(delta)
        mask = torch.ones(2, 1, 3, 4, 4)
        first, _ = V14SafeC2FMoE._identifiable_effective_residual(
            delta, alpha, observed, mask
        )
        second, _ = V14SafeC2FMoE._identifiable_effective_residual(
            delta * 7.0, alpha, observed, mask
        )
        self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5))

    def test_channel_gate_initially_matches_v14(self) -> None:
        base_cfg = compact_v14_config(channels=2)
        channel_cfg = compact_v14_config(channels=2)
        channel_cfg["model"]["v14"].update({
            "channel_final_gate": True,
            "channel_gain_delta": 0.2,
        })
        batch = make_batch(base_cfg)
        torch.manual_seed(1234)
        base_model = DualBranchSTImputer.from_config(base_cfg).eval()
        torch.manual_seed(1234)
        channel_model = DualBranchSTImputer.from_config(channel_cfg).eval()
        with torch.no_grad():
            base = base_model(batch)
            channel = channel_model(batch)
        self.assertTrue(torch.equal(base["x_hat_final"], channel["x_hat_final"]))
        diagnostics = channel["diagnostics"]["v14"]
        self.assertTrue(torch.equal(diagnostics["channel_gain_0"], torch.ones(1)))
        self.assertTrue(torch.equal(diagnostics["channel_gain_1"], torch.ones(1)))

    def test_single_channel_does_not_enable_channel_calibrator(self) -> None:
        cfg = compact_v14_config(channels=1)
        cfg["model"]["v14"]["channel_final_gate"] = True
        model = DualBranchSTImputer.from_config(cfg)
        self.assertFalse(model.main_branch.controller.channel_final_gate)


if __name__ == "__main__":
    unittest.main()
