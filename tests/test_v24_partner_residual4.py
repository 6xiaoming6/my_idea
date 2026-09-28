"""Acceptance checks for the zero-initialized residual partner experiment."""
from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from stmoe_imputer.engine import _loss_gradient_alignment
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE


ROOT = Path(__file__).resolve().parents[1]


class ResidualPartnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setup_models(self):
        args = dict(c_in=2, dim=16, num_steps=4, top_k=2, routing_mode="hard",
                    expert_pool=["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"],
                    attention_heads=4, completion_feedback=False,
                    routing_warmup_epochs=3, routing_transition_epochs=3)
        torch.manual_seed(19)
        native = TemporalSpatialCoE(**args)
        torch.manual_seed(19)
        residual = TemporalSpatialCoE(**args, pair_mode="partner_residual")
        torch.manual_seed(23)
        target = torch.randn(2, 2, 4, 6, 6)
        mask = (torch.rand_like(target) > .4).float()
        return native, residual, {"x_f_gt": target, "x_f_obs": target * mask, "m_f": mask}

    def test_zero_correction_preserves_four_round_native_forward_but_not_balance(self):
        native, residual, batch = self.setup_models()
        for training, routing_epoch in ((False, 7), (True, 1), (True, 7)):
            native.train(training); residual.train(training)
            native.set_routing_epoch(routing_epoch); residual.set_routing_epoch(routing_epoch)
            with torch.no_grad():
                old = native(batch["x_f_obs"], batch["m_f"])
                new = residual(batch["x_f_obs"], batch["m_f"])
            for key in ("selected_experts", "primary_ids", "partner_ids"):
                if key == "primary_ids":
                    expected = old["coe"]["route_logits"].topk(2, -1).indices[..., 0]
                    actual = new["coe"][key]
                elif key == "partner_ids":
                    expected = old["coe"]["route_logits"].topk(2, -1).indices[..., 1]
                    actual = new["coe"][key]
                else:
                    expected, actual = old["coe"][key], new["coe"][key]
                self.assertTrue(torch.equal(expected, actual), key)
            torch.testing.assert_close(old["coe"]["route_weights"], new["coe"]["route_weights"], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(old["x_hat_main"], new["x_hat_main"], atol=2e-6, rtol=2e-6)
            self.assertTrue(torch.isfinite(new["coe"]["partner_scores"]).all())
            self.assertEqual((new["coe"]["route_weights"] > 0).sum(-1).unique().tolist(), [2])
            cfg = json.loads((ROOT / "configs/v24/coe_main_s4_e8_taxibj_base.json").read_text())
            _, old_logs = compute_main_stage_loss(old, batch, cfg)
            _, new_logs = compute_main_stage_loss(new, batch, cfg)
            # Conditional partner candidates restrict the soft pair prior to
            # the chosen primary, so equal hard routes do not imply equal balance.
            self.assertGreater(abs(float(old_logs["l_coe_balance"] - new_logs["l_coe_balance"])), 1e-4)

    def test_topk_ties_and_learned_correction(self):
        native, residual, batch = self.setup_models()
        for a, b in zip(native.routers, residual.routers):
            a[-1].weight.data.zero_(); a[-1].bias.data.zero_()
            b.layers[-1].weight.data.zero_(); b.layers[-1].bias.data.zero_()
        native.eval(); residual.eval()
        with torch.no_grad():
            old = native(batch["x_f_obs"], batch["m_f"])
            tied = residual(batch["x_f_obs"], batch["m_f"])
        self.assertTrue(torch.equal(old["coe"]["selected_experts"], tied["coe"]["selected_experts"]))
        alternative = int(torch.tensor([i for i in range(8) if i not in
                                        tied["coe"]["selected_experts"][0, 0].tolist()])[0])
        def corrected(features, primary):
            scores = features.new_zeros((features.shape[0], 8))
            scores[:, alternative] = 100.
            return scores
        with patch.object(residual.partner_scorer, "forward", side_effect=corrected):
            with torch.no_grad():
                changed = residual(batch["x_f_obs"], batch["m_f"])
        self.assertEqual(int(changed["coe"]["partner_ids"][0, 0]), alternative)
        self.assertFalse(torch.equal(changed["coe"]["selected_experts"][:, 0],
                                     tied["coe"]["selected_experts"][:, 0]))

    def test_corrected_fusion_starts_at_native_and_uses_learned_score(self):
        native, residual, batch = self.setup_models()
        residual.partner_fusion = "corrected"
        native.eval(); residual.eval()
        with torch.no_grad():
            expected = native(batch["x_f_obs"], batch["m_f"])
            initial = residual(batch["x_f_obs"], batch["m_f"])
        torch.testing.assert_close(expected["coe"]["route_weights"],
                                   initial["coe"]["route_weights"], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(expected["x_hat_main"], initial["x_hat_main"],
                                   atol=2e-6, rtol=2e-6)
        with patch.object(residual.partner_scorer, "forward",
                          side_effect=lambda features, primary:
                          torch.nn.functional.one_hot(
                              initial["coe"]["partner_ids"][:, 0], 8
                          ).to(features.dtype) * 2.):
            with torch.no_grad():
                changed = residual(batch["x_f_obs"], batch["m_f"])
        self.assertGreater(float((changed["coe"]["route_weights"][:, 0] -
                                  initial["coe"]["route_weights"][:, 0]).abs().sum()), 0.)

    def test_head_only_rank_scores_detach_backbone(self):
        _, model, batch = self.setup_models()
        model.partner_fusion = "corrected"
        model.partner_aux_head_only = True
        torch.nn.init.normal_(model.partner_scorer.network[-1].weight, std=.05)
        model.train(); model.set_routing_epoch(7)
        outputs = model(batch["x_f_obs"], batch["m_f"])
        score = outputs["coe"]["partner_rank_scores"][:, 0].sum()
        decoder = list(model.decoder.parameters())
        router = list(model.routers[0].parameters())
        scorer = list(model.partner_scorer.parameters())
        gradients = torch.autograd.grad(score, decoder + router + scorer, allow_unused=True)
        self.assertTrue(all(g is None or g.abs().sum() == 0 for g in gradients[:len(decoder)+len(router)]))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients[len(decoder)+len(router):]
                               if g is not None), 0.)

    def test_ranking_score_can_reach_decoder_and_alignment_reports_direction(self):
        _, model, batch = self.setup_models()
        model.train(); model.set_routing_epoch(7)
        torch.nn.init.normal_(model.partner_scorer.network[-1].weight, std=.05)
        output = model(batch["x_f_obs"], batch["m_f"])
        ids = output["coe"]["partner_ids"][:, 0, None]
        score = output["coe"]["partner_scores"][:, 0].gather(1, ids).sum()
        grads = torch.autograd.grad(score, list(model.decoder.parameters()), allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads if g is not None), 0.)
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        task_norm, rank_norm, cosine = _loss_gradient_alignment(parameter.sum(), -parameter.sum(), [parameter])
        self.assertGreater(task_norm, 0.)
        self.assertGreater(rank_norm, 0.)
        self.assertAlmostEqual(cosine, -1., places=6)


if __name__ == "__main__":
    unittest.main()
