"""Follow-up protocol and real CPU/Gloo control pipeline tests; no GPU use."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/v22'))
import verification as verify
from train import build_config, load_suite
from stmoe_imputer.models.registry import MODEL_REGISTRY
from stmoe_imputer.models import DualBranchSTImputer
from test_v22_coarsening_moe import config
from _v14_utils import make_batch


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.builder = MODEL_REGISTRY['v22_coarsening_moe']

    def tearDown(self):
        MODEL_REGISTRY['v22_coarsening_moe'] = self.builder

    def test_protocol_reuses_quick_fingerprints_and_isolates_controls(self):
        policy, suite = verify.load_policy('configs/v22/verification.json')
        before = copy.deepcopy(suite)
        jobs = list(verify.jobs(policy, suite, ['0', '1']))
        self.assertEqual(len(jobs), 18)
        for v, d, m, r, s, cfg, _ in jobs:
            original, _ = build_config(suite, v, d, m, r, s, world_size=2)
            self.assertEqual(cfg, original)
        controls = list(verify.jobs(policy, suite, ['0', '1'], controls=True))
        self.assertEqual(len(controls), 12)
        self.assertEqual({job[4] for job in controls}, {42, 2026, 3407})
        for v, d, m, r, s, cfg, _ in controls:
            self.assertEqual(cfg['train']['epochs'], 120)
            self.assertEqual(cfg['model']['v22']['geometry_control'], v)
            self.assertIn('verification_source_sha256', cfg['model']['v22'])
            original, _ = build_config(suite, 'uniform', d, m, r, s, world_size=2)
            self.assertNotEqual(cfg['experiment_policy']['fingerprint'], original['experiment_policy']['fingerprint'])
        self.assertEqual(suite, before)

    def test_repeated_shared_equals_fixed_in_eval(self):
        cfg = config('fixed_stats')
        torch.manual_seed(42)
        original = DualBranchSTImputer.from_config(cfg).eval()
        cfg['model']['v22'].update(mode='uniform', geometry_control='uniform_fixed_shared')
        verify.install_control_builder()
        torch.manual_seed(42)
        repeated = DualBranchSTImputer.from_config(cfg).eval()
        batch = make_batch(cfg, seed=9)
        with torch.no_grad():
            torch.testing.assert_close(original(batch)['x_hat_final'], repeated(batch)['x_hat_final'])

    def test_independent_branches_are_distinct_and_checkpoint_reloadable(self):
        cfg = config('uniform')
        cfg['model']['v22']['geometry_control'] = 'uniform_fixed_independent'
        verify.install_control_builder()
        model = DualBranchSTImputer.from_config(cfg)
        branches = model.main_branch.scales[0].branch_inputs
        self.assertNotEqual(branches[0].weight.data_ptr(), branches[1].weight.data_ptr())
        self.assertFalse(torch.equal(branches[0].weight, branches[1].weight))
        reloaded = DualBranchSTImputer.from_config(cfg)
        reloaded.load_state_dict(model.state_dict())
        model.eval(); reloaded.eval()
        batch = make_batch(cfg)
        with torch.no_grad():
            torch.testing.assert_close(model(batch)['x_hat_final'], reloaded(batch)['x_hat_final'])

    def test_control_worker_full_train_val_best_test_cpu_ddp(self):
        suite = load_suite(profile='quick')
        for variant in ('uniform_fixed_shared', 'uniform_fixed_independent'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                cfg, _ = verify.control_config(suite, variant, 'BikeNYC', 'random', '0.4', 42, 1, 2)
                cfg['device'] = 'cpu'
                cfg['output_dir'] = tmp
                cfg['data'].update(batch_size=1, num_workers=0)
                cfg['data']['synthetic'].update(num_train=3, num_val=3, t=2, h=8, w=8)
                cfg['model']['v22'].update(dim=8, num_groups=2, dropout=0.1)
                cfg['distributed'].update(per_rank_batch_size=1, global_batch_size=2, timeout_seconds=60)
                path = Path(tmp) / 'cfg.json'
                path.write_text(json.dumps(cfg))
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                           str(ROOT / 'scripts/v22/verification.py'), '--worker', '-c', str(path),
                           '--name', f'ablation_v22_{variant}', '--synthetic', '--quiet', '--no_plot']
                result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=100,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                                 MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                hit = verify.completed(cfg, variant)
                self.assertIsNotNone(hit)
                self.assertGreater(hit['test_mae'], 0)


if __name__ == '__main__':
    unittest.main()
