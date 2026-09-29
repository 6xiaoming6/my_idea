"""The six v25 ablations differ only in the intended repair controls."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v25_ablation_runner', ROOT / 'scripts/v25/run_ablations.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class V25AblationPlanTests(unittest.TestCase):
    def test_six_ordered_matched_taxibj_jobs(self):
        plan = runner._plan('taxibj', 100, 32, [7])
        self.assertEqual([job['variant'] for job in plan['jobs']], list(runner.VARIANTS))
        self.assertEqual(plan['batch_size'], 32)
        configs = [job['config'] for job in plan['jobs']]
        for cfg in configs:
            self.assertEqual(cfg['model']['architecture'], 'v25_ras_coe')
            self.assertFalse(cfg['model']['main']['use_multiscale'])
            self.assertEqual(cfg['model']['coe']['num_steps'], 4)
            self.assertEqual(len(cfg['model']['coe']['expert_pool']), 8)
            self.assertEqual(cfg['model']['coe']['top_k'], 2)
            self.assertEqual(cfg['model']['coe']['pair_mode'], 'partner_residual')
            self.assertEqual(cfg['model']['coe']['partner_fusion'], 'corrected')
            self.assertEqual(cfg['train']['epochs'], 100)
            self.assertEqual(cfg['train']['val_epoch'], 5)
            self.assertEqual(cfg['train']['best_checkpoint_name'], 'best.pth')
        self.assertEqual([c['model']['coe']['completion_feedback'] for c in configs],
                         [False, True, True, True, True, True])
        self.assertEqual([c['model']['coe']['repair_acceptance'] for c in configs],
                         ['none', 'none', 'none', 'latent_point', 'latent_point', 'latent_point'])
        self.assertEqual([c['model']['coe']['repair_feedback_to_router'] for c in configs],
                         [False, False, False, False, False, True])
        self.assertEqual([c['loss']['lambda_coe_accept'] for c in configs],
                         [0, 0, 0, 0, 0.1, 0.1])

    def test_bikenyc_and_epoch_override(self):
        plan = runner._plan('bikenyc', 20, 16, [7])
        self.assertEqual(len(plan['jobs']), 6)
        self.assertTrue(all(job['config']['train']['epochs'] == 20 for job in plan['jobs']))
        self.assertTrue(all(job['config']['train']['scheduler']['total_epochs'] == 20
                            for job in plan['jobs']))
        self.assertTrue(all(job['config']['data']['dataset_name'] == 'BikeNYC'
                            for job in plan['jobs']))


if __name__ == '__main__':
    unittest.main()
