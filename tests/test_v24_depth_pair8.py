"""Depth/width and all-pairs eight-expert comparison checks."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/v24"))
import run_experiments as runner
from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE


EXPERTS = ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"]


class DepthPair8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matched_policy_has_three_ordered_eight_call_arms(self):
        policy = ROOT / "configs/v24/coe_depth_pair8_taxibj_experiments.json"
        plan, _, _, _ = runner.policy_plan(policy, "coe_depth_pair8_taxibj", 80)
        runner.validate_plan(plan)
        self.assertEqual([r["variant"] for r in plan["runs"]], [
            "depthpair_moe_s1_top8", "depthpair_moe_s2_top4",
            "depthpair_coe_s4_pair28",
        ])
        self.assertEqual([
            r["candidate_compute"]["inference_routed_calls_per_window"]
            for r in plan["runs"]
        ], [8, 8, 8])
        self.assertEqual([
            (r["config"]["model"]["coe"]["num_steps"],
             r["config"]["model"]["coe"]["top_k"])
            for r in plan["runs"]
        ], [(1, 8), (2, 4), (4, 2)])

    def test_pair_head_scores_all_28_pairs_and_dispatches_only_winner(self):
        model = TemporalSpatialCoE(
            c_in=2, dim=16, num_steps=4, top_k=2,
            expert_pool=EXPERTS, routing_mode="hard",
            pair_mode="interaction", expert_sharing="shared",
            completion_feedback=False, attention_heads=4,
        )
        self.assertEqual(len(model.pair_indices), 28)
        winner = ((model.pair_indices[:, 0] == 0) &
                  (model.pair_indices[:, 1] == 7)).nonzero().item()
        for router in model.routers:
            router.pair_head.weight.data.zero_()
            router.pair_head.bias.data.fill_(-1000.)
            router.pair_head.bias.data[winner] = 1000.
        calls = [0] * 8
        hooks = [
            expert.register_forward_hook(
                lambda _module, _input, _output, i=i:
                calls.__setitem__(i, calls[i] + 1)
            )
            for i, expert in enumerate(model.routed_experts())
        ]
        try:
            model.eval()
            observed = torch.randn(2, 2, 4, 6, 6)
            mask = (torch.rand_like(observed) > .4).float()
            with torch.no_grad():
                result = model(observed * mask, mask)["coe"]
            self.assertEqual(result["pair_logits"].shape, (2, 4, 28))
            self.assertTrue(torch.equal(
                result["pair_ids"], torch.full_like(result["pair_ids"], winner)
            ))
            self.assertTrue(torch.equal(
                result["selected_experts"],
                torch.tensor([0, 7]).expand(2, 4, 2),
            ))
            self.assertEqual(calls, [4, 0, 0, 0, 0, 0, 0, 4])
        finally:
            for hook in hooks:
                hook.remove()

    def test_moe_topk_and_independent_step_pools(self):
        for steps, top_k in ((1, 8), (2, 4)):
            model = TemporalSpatialCoE(
                c_in=2, dim=16, num_steps=steps, top_k=top_k,
                expert_pool=EXPERTS, routing_mode="hard",
                expert_sharing="per_step", completion_feedback=False,
                attention_heads=4,
            )
            if steps == 2:
                self.assertIsNot(model.routed_experts(0)[0],
                                 model.routed_experts(1)[0])
            model.eval()
            observed = torch.randn(2, 2, 4, 6, 6)
            mask = (torch.rand_like(observed) > .4).float()
            with torch.no_grad():
                weights = model(observed * mask, mask)["coe"]["route_weights"]
            self.assertTrue(torch.all((weights > 0).sum(-1) == top_k))


if __name__ == "__main__":
    unittest.main()
