"""Focused checks for sparse conditional fusion and pair dense warmup."""
from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE


POOL = ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"]
COMMON = dict(
    c_in=2, dim=8, num_steps=4, expert_pool=POOL, top_k=2,
    routing_mode="hard", pair_mode="native", completion_feedback=False,
    routing_warmup_epochs=3, routing_transition_epochs=3,
)


class FusionExplorationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(3)
        cls.values = torch.randn(2, 2, 3, 4, 4)
        cls.mask = (torch.rand_like(cls.values) > 0.4).float()

    def test_zero_initialized_gates_preserve_native_top2(self) -> None:
        torch.manual_seed(19)
        native = TemporalSpatialCoE(**COMMON).eval()
        reference = native(self.values, self.mask)
        selected = reference["coe"]["selected_experts"]
        for mode in ("context", "response", "local_response", "proposal"):
            torch.manual_seed(19)
            model = TemporalSpatialCoE(**COMMON, fusion_mode=mode).eval()
            outputs = model(self.values, self.mask)
            torch.testing.assert_close(outputs["x_hat_main"], reference["x_hat_main"],
                                       rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(outputs["coe"]["selected_experts"], selected)
            self.assertTrue((outputs["coe"]["route_weights"].gt(0).sum(-1) == 2).all())
            for step in range(1, 5):
                self.assertLess(float(outputs["diagnostics"]["coe"][f"step{step}_fusion_shift_abs_mean"]), 1e-6)
            model.train()
            model(self.values, self.mask)["x_hat_main"].square().mean().backward()
            self.assertGreater(float(model.fusion_gate[-1].weight.grad.abs().sum()), 0)

    def test_proposal_gate_uses_only_observed_values(self) -> None:
        torch.manual_seed(23)
        model = TemporalSpatialCoE(**COMMON, fusion_mode="proposal").eval()
        original = model(self.values, self.mask)["x_hat_main"]
        changed_missing = self.values + (1 - self.mask) * 1000
        changed = model(changed_missing, self.mask)["x_hat_main"]
        torch.testing.assert_close(original, changed, rtol=0, atol=0)

    def test_only_first_two_rounds_are_dense_during_pair_warmup(self) -> None:
        model = TemporalSpatialCoE(
            **{**COMMON, "pair_mode": "interaction"}, pair_dense_warmup_steps=2,
        ).train()
        for epoch in (1, 4, 5, 6):
            model.set_routing_epoch(epoch)
            weights = model(self.values, self.mask)["coe"]["route_weights"]
            active = weights.detach().gt(0).sum(-1)
            if epoch < 6:
                self.assertTrue((active[:, :2] == 8).all())
                self.assertTrue((active[:, 2:] == 2).all())
            else:
                self.assertTrue((active == 2).all())


if __name__ == "__main__":
    unittest.main()
