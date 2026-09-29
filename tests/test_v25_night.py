"""Frozen ten-arm protocol and one-batch CPU forward/backward checks."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

import torch

from v25_ras_coe.model import RASCoE
from v25_ras_coe.losses import compute_v25_loss
from v25_ras_coe.quality import RASQualityMetrics

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v25_night_runner', ROOT/'scripts/v25/run_ras_night.py')
night = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(night)


class NightProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.jobs = night.plan()

    def test_ten_ordered_frozen_arms(self):
        self.assertEqual([job['id'] for job in self.jobs], [f'E{i}' for i in range(10)])
        expected = [
            ('partner_residual', False, 'none', 'none', False, 0),
            ('partner_residual', True, 'none', 'none', False, 0),
            ('partner_residual', True, 'point', 'none', False, 0),
            ('partner_residual', True, 'none', 'latent_point', False, 0),
            ('partner_residual', True, 'none', 'latent_point', False, .1),
            ('partner_residual', True, 'none', 'latent_point', True, .1),
            ('partner_residual', True, 'none', 'latent_point', True, .03),
            ('partner_residual', True, 'none', 'latent_point', True, .3),
            ('native', False, 'none', 'none', False, 0),
            ('native', True, 'none', 'latent_point', True, .1),
        ]
        for job, target in zip(self.jobs, expected):
            cfg = job['config']; coe = cfg['model']['coe']
            actual = (coe['pair_mode'], coe['completion_feedback'], coe['acceptance'],
                      coe['repair_acceptance'], coe['repair_feedback_to_router'],
                      cfg['loss']['lambda_coe_accept'])
            self.assertEqual(actual, target, job['id'])
            self.assertEqual(cfg['train']['epochs'], 100)
            self.assertEqual(cfg['train']['scheduler']['total_epochs'], 100)
            self.assertEqual(cfg['train']['val_epoch'], 2 if job['id'] == 'E0' else 5)
            self.assertEqual(cfg['data']['batch_size'], 32)
            self.assertEqual(cfg['seed'], 7)
            self.assertFalse(cfg['model']['main']['use_multiscale'])

    def test_all_ten_one_batch_finite_backward(self):
        torch.manual_seed(93)
        target = torch.randn(2, 2, 3, 4, 4)
        mask = (torch.rand_like(target) > .4).float()
        batch = {'x_f_gt': target, 'm_f': mask, 'mask_family': torch.tensor([0, 1])}
        x_obs = torch.where(mask.bool(), target, 0.)
        for job in self.jobs:
            cfg = night.concrete_config(job, 1)
            cfg['model']['main'].update(dim=8, max_t=3, h=4, w=4)
            model = RASCoE.from_config(cfg).train()
            output = model(x_obs, mask)
            output['x_hat_final'] = output['x_hat_main']
            self.assertEqual(output['x_hat_main'].shape, target.shape, job['id'])
            self.assertEqual(output['coe']['route_probs'].shape, (2, 4, 8), job['id'])
            self.assertTrue(torch.isfinite(output['coe']['route_probs']).all(), job['id'])
            self.assertTrue(torch.isfinite(output['coe']['route_weights']).all(), job['id'])
            quality = RASQualityMetrics()
            quality.update(output, batch)
            measures = quality.compute()
            self.assertIn('coe_step4_mae', measures, job['id'])
            self.assertIn('coe_family_random_point_mae', measures, job['id'])
            self.assertIn('coe_family_node_outage_mae', measures, job['id'])
            total, _ = compute_v25_loss(output, batch, cfg)
            self.assertTrue(torch.isfinite(total), job['id'])
            total.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()), job['id'])
            if cfg['model']['coe']['repair_acceptance'] == 'latent_point':
                weights = output['coe']['repair_acceptance_weights']
                self.assertTrue(torch.isfinite(weights).all(), job['id'])
                self.assertTrue(((weights > 0) & (weights < 1)).all(), job['id'])

    def test_complete_receipt_and_summary_require_all_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='v25_night_receipt_') as tmp:
            original = night.SUITE
            night.SUITE = Path(tmp)
            try:
                job = self.jobs[0]
                group = night.SUITE / job['directory']
                run_dir = group / 'actual_run'
                (run_dir / 'logs').mkdir(parents=True)
                (run_dir / 'checkpoints').mkdir()
                cfg = night.concrete_config(job, 1)
                (run_dir / 'config.json').write_text(json.dumps(cfg))
                (run_dir / 'checkpoints/best.pt').write_bytes(b'checkpoint')
                with (run_dir / 'logs/metrics.jsonl').open('w') as stream:
                    for epoch in range(1, 101):
                        stream.write(json.dumps({'epoch': epoch, 'train': {'mae': 2.0},
                            'val': {'mae': 1.0} if epoch % 2 == 0 else None,
                            'perf': {'train_time_sec': 1.0, 'epoch_time_sec': 1.1}}) + '\n')
                    stream.write(json.dumps({'stage': 'test', 'metrics': {'mae': 1.2}}) + '\n')
                for name in ('train.log', 'val.log', 'test.log'):
                    (run_dir / 'logs' / name).write_text('ok\n')
                receipt = {'status': 'finished', 'run_dir': str(run_dir),
                    'config_sha256': night._digest(cfg), 'completed_epochs': 100,
                    'best_epoch': 80, 'best_val_mae': 1.0, 'attempt': 1,
                    'total_params': 10, 'peak_memory_gb': 0.1,
                    'test': {'mae': 1.2, 'rmse': 2.0, 'wape': .2,
                             'coe_initial_mae': 1.5, 'coe_step4_mae': 1.2}}
                night.save_json(group / 'receipt.json', receipt)
                self.assertIsNone(night.complete(job))
                night.link_outputs(group, run_dir)
                night.analysis_files(group, receipt)
                night.save_json(group / 'summary.json', night.row_for(job, receipt))
                self.assertIsNotNone(night.complete(job))
                rows = night.summarize(self.jobs)
                self.assertEqual(rows[0]['status'], 'complete')
                self.assertEqual(rows[0]['best_epoch'], 80)
                self.assertEqual(rows[0]['initial_mae'], 1.5)
                self.assertEqual(rows[0]['step1_overrepair_prevention_rate'], None)
                self.assertTrue((night.SUITE / 'summary.csv').is_file())
                self.assertIn('Pairwise differences', (night.SUITE / 'comparison.md').read_text())
            finally:
                night.SUITE = original

    def test_prevention_rate_uses_total_harm_counts(self):
        old = torch.zeros(1, 1, 1, 1, 2)
        candidate = torch.tensor([[[[[1.0, 0.0]]]]])
        accepted = torch.tensor([[[[[0.0, 1.0]]]]])
        output = {'x_hat_final': accepted,
                  'coe': {'initial_completion': old,
                          'candidate_completions': [candidate],
                          'completions': [accepted]}}
        batch = {'x_f_gt': old, 'm_f': torch.zeros_like(old)}
        metrics = RASQualityMetrics()
        metrics.update(output, batch)
        result = metrics.compute()
        self.assertEqual(result['coe_step1_candidate_harm_rate'], .5)
        self.assertEqual(result['coe_step1_accepted_harm_rate'], .5)
        self.assertEqual(result['coe_step1_overrepair_prevention_rate'], 0.0)

    def test_native_feedback_zero_init_preserves_base_route(self):
        e8, e9 = self.jobs[8:]
        c8 = night.concrete_config(e8, 1)
        c9 = night.concrete_config(e9, 1)
        for cfg in (c8, c9):
            cfg['model']['main'].update(dim=8, max_t=3, h=4, w=4)
        torch.manual_seed(7)
        base = RASCoE.from_config(c8).eval()
        torch.manual_seed(7)
        ras = RASCoE.from_config(c9).eval()
        target = torch.randn(2, 2, 3, 4, 4)
        mask = (torch.rand_like(target) > .4).float()
        x_obs = torch.where(mask.bool(), target, 0.)
        left, right = base(x_obs, mask), ras(x_obs, mask)
        torch.testing.assert_close(left['coe']['route_logits'][:, 0],
                                   right['coe']['route_logits'][:, 0], rtol=0, atol=0)
        self.assertEqual(right['coe']['repair_acceptance_weights'].shape,
                         (2, 4, 1, 3, 4, 4))


if __name__ == '__main__':
    unittest.main()
