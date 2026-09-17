"""Algebra, leakage, initialization, gradients and train/val/test contracts."""
import copy
import io
import math
import unittest

import torch

from test_dual_moe import ROOT, config, batch
from stmoe_imputer.config import deep_update, load_config
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.recoverability import (
    spacetime_basis, observation_system, ridge_decomposition, aligned_evaluate)
from stmoe_imputer.engine import train_one_epoch, evaluate, build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.routing_metrics import RecoverabilityMetricAccumulator
from stmoe_imputer.utils.checkpoint import snapshot_model_state
from stmoe_imputer.utils.train_logger import TrainLogger


def recovery_config(mode='constrained', channels=2):
    cfg = deep_update(config(channels), load_config(ROOT/'configs/presets/dual_moe_st_dilated.json'))
    cfg['model']['dual_moe'].update(dim=8, coarse_nodes=[4, 2], completion_expert_hidden=8,
                                  backend_diagnostics=False, recoverability={'mode': mode, 'ridge': .05, 'strength': .1})
    cfg['loss']['dual_moe_recoverability_weight'] = .01 if mode != 'off' else 0.
    cfg['train']['amp'] = False
    return cfg


class RecoverabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_basis_fixed_orthogonal_and_degenerate_axes(self):
        phi = spacetime_basis(3, 5, 7, 'cpu')
        torch.testing.assert_close(phi.T@phi/len(phi), torch.eye(4), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(spacetime_basis(1, 1, 1, 'cpu'), torch.tensor([[1., 0., 0., 0.]]))

    def test_matched_count_geometry_and_target_dependence(self):
        phi = spacetime_basis(1, 5, 5, 'cpu')
        a = torch.ones(2, 1, 1, 25, 1)
        m = torch.zeros(2, 1, 1, 5, 5)
        m[0, 0, 0, 2, [0, 1, 3, 4]] = 1
        for y, x in [(0, 0), (0, 4), (4, 0), (4, 4)]: m[1, 0, 0, y, x] = 1
        z = (phi@torch.tensor([1., 2., 3., 0.])).reshape(1, 1, 1, 5, 5).expand(2, -1, -1, -1, -1)
        g, rhs, cov = observation_system(a, m, z, phi)
        torch.testing.assert_close(cov[0], cov[1])
        obs, q = ridge_decomposition(g, rhs, .01)
        self.assertAlmostEqual(float(q[0, 0, 0, 2, 2]), 1., places=5)
        self.assertLess(float(q[1, 0, 0, 2, 2]), .1)
        # Extra observations reduce PSD weakness for a FIXED basis/assignment.
        g2, b2, _ = observation_system(a, torch.ones_like(m), z, phi)
        _, q2 = ridge_decomposition(g2, b2, .01)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(q-q2).min()), -1e-6)
        prior = torch.randn_like(obs)
        coef = obs+q@prior
        eye = torch.eye(4)
        torch.testing.assert_close((g+.01*eye)@coef, rhs+.01*prior, atol=1e-5, rtol=1e-5)

    def test_no_observations_all_prior_and_hidden_nan_safe(self):
        phi = spacetime_basis(2, 3, 3, 'cpu')
        a = torch.ones(1, 2, 2, 9, 1)
        m = torch.zeros(1, 1, 2, 3, 3)
        g, rhs, cov = observation_system(a, m, torch.full_like(m, float('nan')), phi)
        obs, q = ridge_decomposition(g, rhs, .05)
        torch.testing.assert_close(obs, torch.zeros_like(obs))
        torch.testing.assert_close(q, torch.eye(4).expand_as(q))
        self.assertEqual(float(cov.sum()), 0.)

    def test_region_permutation_invariance(self):
        a = torch.rand(1, 2, 3, 12, 4).softmax(-1)
        coef = torch.randn(1, 2, 4, 4, 2)
        phi = spacetime_basis(3, 3, 4, 'cpu')
        p = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(aligned_evaluate(a, coef, phi), aligned_evaluate(a[..., p], coef[:, :, p], phi))

    def test_all_modes_preserve_base_weights_and_off_exact(self):
        models = {}
        for mode in ('off', 'scalar', 'matrix', 'constrained', 'fit_only'):
            torch.manual_seed(42)
            models[mode] = DualBranchSTImputer.from_config(recovery_config(mode))
        baseline = models['off'].state_dict()
        for model in models.values():
            for name, value in baseline.items():
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        counts = [sum(p.numel() for p in models[m].parameters()) for m in ('scalar', 'matrix', 'constrained', 'fit_only')]
        self.assertEqual(len(set(counts)), 1)
        cfg = recovery_config('off'); del cfg['model']['dual_moe']['recoverability']
        torch.manual_seed(42); old = DualBranchSTImputer.from_config(cfg)
        self.assertEqual(set(old.state_dict()), set(baseline))
        data = batch(h=4, w=5, n=1)
        torch.testing.assert_close(old(data)['x_hat_main'], models['off'](data)['x_hat_main'], atol=0, rtol=0)

    def test_modes_train_val_restore_test_and_logging(self):
        for mode in ('scalar', 'matrix', 'constrained', 'fit_only'):
            cfg = recovery_config(mode)
            model = DualBranchSTImputer.from_config(cfg)
            optimizer = build_optimizer(model, cfg)
            data = [batch(h=4, w=5, n=1) for _ in range(3)]
            train = train_one_epoch(model, [data[0]], optimizer, torch.device('cpu'), cfg, 1)
            for name, p in model.named_parameters():
                if 'recoverability' in name or 'aggregation.mid.experts.0' in name:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
            val = evaluate(model, [data[1]], torch.device('cpu'), cfg)
            state = snapshot_model_state(model)
            with torch.no_grad(): next(model.parameters()).add_(100)
            model.load_state_dict(state)
            test = evaluate(model, [data[2]], torch.device('cpu'), cfg)
            for result in (train, val, test):
                self.assertTrue(all(math.isfinite(v) for v in result.values()))
                self.assertIn('l_recoverability', result)
                self.assertIn('recovery_mid_all_branch_mae', result)
            stream = io.StringIO(); TrainLogger._log_dual_moe(stream, val)
            self.assertIn('recoverability (model-relative', stream.getvalue())

    def test_no_target_or_hidden_input_leak_and_extreme_masks(self):
        for mode in ('scalar', 'matrix', 'constrained', 'fit_only'):
            model = DualBranchSTImputer.from_config(recovery_config(mode)).eval()
            data = batch(h=3, w=4, n=1)
            first = model(data)
            changed = {**data, 'x_f_gt': torch.full_like(data['x_f_gt'], float('nan')),
                       'x_f_obs': torch.where(data['m_f'].bool(), data['x_f_obs'], float('nan'))}
            second = model(changed)
            torch.testing.assert_close(first['x_hat_main'], second['x_hat_main'])
            for scale in ('mid', 'coarse'):
                torch.testing.assert_close(first['recoverability'][scale]['prediction'], second['recoverability'][scale]['prediction'])
            for observed in (0., 1.):
                data = batch(channels=2, t=1, h=1, w=1, n=1); data['m_f'].fill_(observed)
                out = model(data); loss, _ = compute_main_stage_loss(out, data, recovery_config(mode))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))

    def test_cpu_autocast_keeps_small_solves_finite(self):
        cfg = recovery_config(); model = DualBranchSTImputer.from_config(cfg)
        data = batch(t=3, h=4, w=5, n=1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model(data)
            loss, _ = compute_main_stage_loss(out, data, cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_diagnostics_missing_only_partition_invariant(self):
        truth = torch.zeros(2, 1, 1, 2, 2)
        mask = torch.zeros_like(truth); mask[0, ..., 0, 0] = 1
        weak = torch.tensor([.1, .2, .5, .9]*2).reshape_as(truth)
        out = {'x_hat_main': truth+2, 'recoverability': {'mid': {'prediction': truth+3,
                'weakness': weak, 'coverage': 1-weak}}}
        data = {'x_f_gt': truth, 'm_f': mask}
        whole = RecoverabilityMetricAccumulator(); whole.update(out, data)
        parts = RecoverabilityMetricAccumulator()
        for i in range(2):
            parts.update({'x_hat_main': out['x_hat_main'][i:i+1], 'recoverability': {'mid':
                         {k:v[i:i+1] for k,v in out['recoverability']['mid'].items()}}},
                         {k:v[i:i+1] for k,v in data.items()})
        self.assertEqual(whole.compute(), parts.compute())
        result = whole.compute()
        self.assertEqual(result['recovery_mid_all_count'], 7)
        self.assertEqual(result['recovery_mid_all_mae'], 2)
        self.assertEqual(result['recovery_mid_all_branch_mae'], 3)

    def test_invalid_settings(self):
        for patch in ({'mode':'bad'}, {'ridge':0}, {'ridge':float('nan')}, {'strength':True}, {'basis':'learnable'}):
            cfg = recovery_config(); cfg['model']['dual_moe']['recoverability'].update(patch)
            with self.assertRaises(ValueError): DualBranchSTImputer.from_config(cfg)


if __name__ == '__main__': unittest.main()
