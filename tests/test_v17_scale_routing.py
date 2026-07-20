from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import compact_v17_config, make_batch


class V17ScaleRoutingTest(unittest.TestCase):
    def test_scale_expert_and_route_gates_are_valid(self) -> None:
        cfg = compact_v17_config(scale_mode="fine_mid")
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            outputs = model(make_batch(cfg))
        scale = outputs["gates"]["scale_gate"]
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue((scale >= 0).all())
        torch.testing.assert_close(scale.sum(dim=1), torch.ones(scale.shape[0]))
        self.assertTrue((scale[:, 0] >= 0.25).all())
        torch.testing.assert_close(scale[:, 2], torch.zeros_like(scale[:, 2]))
        for name in ("fine", "mid", "coarse"):
            gate = outputs["gates"][name]
            self.assertTrue(torch.isfinite(gate).all())
            torch.testing.assert_close(gate.sum(dim=1), torch.ones(gate.shape[0]))
            indices = outputs["topk"][f"{name}_indices"]
            self.assertTrue(
                ((indices >= 0) & (indices < cfg["model"]["main"]["num_experts"])).all()
            )
        route = outputs["gates"]["route_branch_gate"]
        self.assertEqual(route.shape, (1, 1, 1, 1, 1))
        self.assertTrue(((route >= 0) & (route <= 1)).all())
