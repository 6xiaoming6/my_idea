"""Focused checks for the primary-conditioned sparse partner study."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from stmoe_imputer.models.imputer import DualBranchSTImputer
from stmoe_imputer.partner_study import last_step_oracle, partner_candidate_loss
from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint


class PartnerStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setup_case(self, mode="partner"):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "configs/v24/coe_partner_taxibj_base.json").read_text())
        cfg["model"]["main"].update(dim=16, max_t=4, h=6, w=6)
        cfg["model"]["coe"].update(router_hidden_dim=16, attention_heads=4, pair_mode=mode)
        cfg["train"]["amp"] = False
        torch.manual_seed(29)
        model = DualBranchSTImputer.from_config(cfg)
        target = torch.randn(3, 2, 4, 6, 6)
        observed = (torch.rand_like(target) > 0.4).float()
        batch = {"x_f_gt": target, "x_f_obs": target * observed, "m_f": observed}
        return cfg, model, batch

    def test_sparse_partner_and_candidate_importance(self):
        _, model, batch = self.setup_case()
        calls = []
        hooks = [expert.register_forward_hook(
            lambda _module, inputs, _out, label=label: calls.append((label, inputs[0].shape[0]))
        ) for label, expert in zip(model.main_branch.expert_names,
                                   model.main_branch.routed_experts(0))]
        output = model(batch)
        for hook in hooks:
            hook.remove()
        # Experts are shared across three rounds, but each sample activates
        # exactly two in each round (the primary is reused, not recomputed).
        self.assertEqual(sum(count for _, count in calls), 2 * 3 * batch["x_f_gt"].shape[0])
        coe = output["coe"]
        self.assertTrue(torch.allclose(coe["route_importance"].sum(-1),
                                       torch.ones(3, 3), atol=1e-5))
        self.assertTrue(torch.allclose(coe["route_weights"].sum(-1),
                                       torch.ones(3, 3), atol=1e-5))
        self.assertTrue((coe["selected_experts"][:, :, 0] !=
                         coe["selected_experts"][:, :, 1]).all())

    def test_candidate_ranking_updates_partner_head_without_rng_advance(self):
        cfg, model, batch = self.setup_case()
        model.train()
        outputs = model(batch)
        before = torch.get_rng_state().clone()
        ranking, info = partner_candidate_loss(model, batch, outputs, cfg, 4, 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(info["partner_probe_forward_calls"], 3.)
        ranking.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.main_branch.partner_scorer.parameters()
                               if p.grad is not None), 0.)

    def test_last_round_oracle_order_and_safe_checkpoint(self):
        cfg, model, batch = self.setup_case()
        oracle = last_step_oracle(model, [batch], torch.device("cpu"), cfg, 3)
        self.assertLessEqual(oracle["all_best_mae"], oracle["anchor_best_mae"] + 1e-6)
        self.assertLessEqual(oracle["anchor_best_mae"], oracle["top2_mae"] + 1e-6)
        self.assertEqual(oracle["candidate_forward_calls_per_sample"], 15.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            save_checkpoint(path, model, optimizer, 1, {"val_mae": 1.}, cfg,
                            scheduler=scheduler, scaler=scaler,
                            rng_states=[{"torch_cpu": torch.get_rng_state()}])
            payload = load_checkpoint(path, model)
            self.assertEqual(payload["epoch"], 1)
            self.assertIsNotNone(payload["scheduler"])
            self.assertIsNotNone(payload["scaler"])
            self.assertEqual(len(payload["rng_states"]), 1)


if __name__ == "__main__":
    unittest.main()
