"""Focused selective-commit, oracle-label, and compatibility checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import torch

from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE
from stmoe_imputer.losses import supervision_mask
from v25_ras_coe.model import RASCoE
from v25_ras_coe.losses import compute_repair_acceptance_loss, repair_oracle_labels
from v25_ras_coe.quality import RASQualityMetrics

ROOT = Path(__file__).resolve().parents[1]


def config(mode="latent_point", feedback=False):
    cfg = json.loads((ROOT / 'configs/v24/coe_partner_residual4_taxibj_base.json').read_text())
    coe = cfg['model']['coe']
    coe.update(num_steps=2, fixed_expert_steps=[None, None],
               pair_mode='partner_residual', partner_fusion='corrected',
               partner_aux_head_only=True, completion_feedback=mode != 'none',
               repair_acceptance=mode, repair_feedback_to_router=feedback,
               repair_accept_aux_head_only=True)
    cfg['model']['main'].update(dim=8, max_t=3, h=4, w=4)
    cfg['loss'].update(lambda_coe_accept=0.1, repair_accept_margin=0.0)
    return cfg


class RepairAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        cls.x = torch.randn(2, 2, 3, 4, 4)
        cls.mask = (torch.rand_like(cls.x) > 0.4).float()

    def test_disabled_equivalence(self):
        cfg = config('none')
        torch.manual_seed(7)
        old = TemporalSpatialCoE.from_config(cfg).eval()
        torch.manual_seed(7)
        new = RASCoE.from_config(cfg).eval()
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        left, right = old(self.x, self.mask), new(self.x, self.mask)
        for key in ('x_hat_main', 'h_st_aux'):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        for key in ('route_logits', 'route_weights', 'selected_experts'):
            torch.testing.assert_close(left['coe'][key], right['coe'][key], rtol=0, atol=0)

    def test_feedback_zero_init_preserves_supervised_commit_start(self):
        torch.manual_seed(41)
        a4 = RASCoE.from_config(config(feedback=False)).eval()
        torch.manual_seed(41)
        a5 = RASCoE.from_config(config(feedback=True)).eval()
        left, right = a4(self.x, self.mask), a5(self.x, self.mask)
        torch.testing.assert_close(left['x_hat_main'], right['x_hat_main'], rtol=0, atol=0)
        for key in ('route_logits', 'route_weights', 'selected_experts'):
            torch.testing.assert_close(left['coe'][key], right['coe'][key], rtol=0, atol=0)

    def test_invalid_repair_configuration_is_rejected(self):
        cfg = config()
        cfg['model']['coe']['completion_feedback'] = False
        with self.assertRaisesRegex(ValueError, 'completion feedback'):
            RASCoE.from_config(cfg)
        cfg = config(feedback=True)
        cfg['model']['coe']['router_features'] = 'grouped'
        with self.assertRaisesRegex(ValueError, 'legacy hard Top-2'):
            RASCoE.from_config(cfg)

    def test_gate_shape_range_and_gradients(self):
        model = RASCoE.from_config(config()).train()
        out = model(self.x, self.mask)
        coe = out['coe']
        self.assertEqual(coe['repair_acceptance_logits'].shape, (2, 2, 1, 3, 4, 4))
        self.assertTrue(((coe['repair_acceptance_weights'] > 0) &
                         (coe['repair_acceptance_weights'] < 1)).all())
        self.assertEqual(coe['repair_rejection_maps'].shape, (2, 2, 1, 3, 4, 4))
        out['x_hat_main'].square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.repair_acceptance_gate.parameters()))

    def test_full_accept_and_reject_commit(self):
        model = RASCoE.from_config(config()).eval()
        decoder_inputs = []
        hook = model.decoder.register_forward_pre_hook(lambda _module, inputs: decoder_inputs.append(inputs[0].detach().clone()))
        try:
            with torch.no_grad():
                model.repair_acceptance_gate.net[-1].bias.fill_(40)
                accepted = model(self.x, self.mask)
            for proposal, final in zip(accepted['coe']['candidate_predictions'], accepted['coe']['predictions']):
                torch.testing.assert_close(proposal, final, rtol=1e-5, atol=1e-5)
            decoder_inputs.clear()
            with torch.no_grad():
                model.repair_acceptance_gate.net[-1].bias.fill_(-40)
                model(self.x, self.mask)
            point_missing = (~self.mask.bool()).any(dim=1, keepdim=True).expand_as(decoder_inputs[0])
            # Decoder sees initial, then each round's candidate and committed hidden.
            self.assertEqual(len(decoder_inputs), 7)
            for index in range(2):
                before = decoder_inputs[0] if index == 0 else decoder_inputs[3]
                candidate = decoder_inputs[3 * index + 2]
                committed = decoder_inputs[3 * index + 3]
                torch.testing.assert_close(committed[point_missing], before[point_missing], rtol=0, atol=1e-6)
                torch.testing.assert_close(committed[~point_missing], candidate[~point_missing], rtol=0, atol=1e-6)
        finally:
            hook.remove()

    def test_no_target_leakage(self):
        cfg = config()
        model = RASCoE.from_config(cfg).eval()
        out_a = model(self.x, self.mask)
        batch_a = {'x_f_gt': out_a['coe']['initial_completion'].detach().clone(), 'm_f': self.mask,
                   'target_mask': torch.ones_like(self.mask)}
        batch_b = copy.deepcopy(batch_a)
        batch_b['x_f_gt'] = out_a['coe']['candidate_completions'][0].detach().clone()
        out_b = model(self.x, self.mask)
        torch.testing.assert_close(out_a['x_hat_main'], out_b['x_hat_main'], rtol=0, atol=0)
        left, _ = compute_repair_acceptance_loss(out_a, batch_a, cfg)
        right, _ = compute_repair_acceptance_loss(out_b, batch_b, cfg)
        self.assertNotAlmostEqual(float(left), float(right), places=5)

    def test_oracle_labels_and_empty_supervision(self):
        old = torch.tensor([[[[[3.0, 1.0]]]]])
        candidate = torch.tensor([[[[[1.0, 4.0]]]]])
        target = torch.zeros_like(old)
        selected = torch.ones_like(old, dtype=torch.bool)
        positive, negative, _, _ = repair_oracle_labels(old, candidate, target, selected)
        self.assertEqual(positive.flatten().tolist(), [True, False])
        self.assertEqual(negative.flatten().tolist(), [False, True])
        cfg = config()
        model = RASCoE.from_config(cfg).train()
        out = model(self.x, self.mask)
        batch = {'x_f_gt': self.x, 'm_f': self.mask,
                 'target_mask': torch.zeros_like(self.mask)}
        loss, logs = compute_repair_acceptance_loss(out, batch, cfg)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(logs['accept_valid_points']), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_auxiliary_loss_only_updates_gate(self):
        cfg = config()
        model = RASCoE.from_config(cfg).train()
        out = model(self.x, self.mask)
        batch = {'x_f_gt': self.x, 'm_f': self.mask}
        loss, _ = compute_repair_acceptance_loss(out, batch, cfg)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.repair_acceptance_gate.parameters()))
        self.assertIsNone(model.encoder[0].weight.grad)
        self.assertIsNone(next(model.pattern_experts['T'].parameters()).grad)

    def test_feedback_changes_router_features(self):
        model = RASCoE.from_config(config(feedback=True))
        hidden = torch.randn(2, 8, 3, 4, 4)
        values = torch.randn(2, 2, 3, 4, 4)
        support_summary = torch.zeros(2, 80)
        missing = (1 - self.mask)
        zero = torch.zeros(2, 1, 3, 4, 4)
        one = torch.ones_like(zero)
        left = model._router_features(hidden, values, support_summary, values, missing,
                                      repair_signal=zero)
        right = model._router_features(hidden, values, support_summary, values, missing,
                                       repair_signal=one)
        self.assertEqual(left.shape[1] + 0, right.shape[1])
        self.assertFalse(torch.equal(left, right))

    def test_quality_metrics_use_exact_counts(self):
        cfg = config()
        model = RASCoE.from_config(cfg).eval()
        outputs = model(self.x, self.mask)
        batch = {'x_f_gt': self.x, 'm_f': self.mask}
        first = RASQualityMetrics(); second = RASQualityMetrics()
        first.update(outputs, batch); second.update(outputs, batch)
        first.merge(second)
        result = first.compute()
        self.assertIn('coe_step1_candidate_harm_rate', result)
        self.assertIn('coe_step2_mae', result)
        self.assertIn('coe_all_steps_monotonic_sample_rate', result)
        self.assertIn('coe_accept_f1', result)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))


if __name__ == '__main__':
    unittest.main()
