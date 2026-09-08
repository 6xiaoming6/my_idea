"""V22.1 math, isolation and real two-process CPU/Gloo integration checks."""
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
import run_refinement as runner
from refinement_model import BoundedRouter, V221CoarseningMoE
from train import build_config
from stmoe_imputer.models.v_single.v22_coarsening_moe import V22CoarseningMoE, CoarseningScale
from test_v22_coarsening_moe import config
from _v14_utils import make_batch


def options(fallback=0.):
    return dict(max_strength=.5, initial_strength=.1, fallback_probability=fallback)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def router(self, fallback=0.):
        scale = CoarseningScale(8, 2, {'mode': 'moe'}, 2, 0.)
        return BoundedRouter(scale.router, options(fallback))

    def test_probability_bounds_and_uniform_initialization(self):
        router = self.router().eval()
        x = torch.randn(4, 15, 2, 4, 4)
        gate = router(x).softmax(1)
        torch.testing.assert_close(gate, torch.full_like(gate, 1/3))
        with torch.no_grad():
            router.preference[-1].bias.copy_(torch.tensor([200., -200., -200.]))
            router.strength.bias.fill_(200.)
        gate = router(x).softmax(1)
        torch.testing.assert_close(gate.sum(1), torch.ones_like(gate[:, 0]))
        self.assertGreaterEqual(float(gate.min()), 1/6 - 1e-6)
        self.assertLessEqual(float(gate.max()), 2/3 + 1e-6)
        gate.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters()))

    def test_fallback_is_train_only_without_rescaling(self):
        router = self.router(.5)
        with torch.no_grad():
            router.preference[-1].bias.copy_(torch.tensor([3., -3., 0.]))
        x = torch.randn(64, 15, 1, 2, 2)
        torch.manual_seed(17)
        train_gate = router(x).softmax(1)
        uniform_samples = (train_gate - 1/3).abs().flatten(1).max(1).values < 1e-6
        self.assertGreater(int(uniform_samples.sum()), 0)
        self.assertLess(int(uniform_samples.sum()), 64)
        router.eval()
        a = router(x).softmax(1)
        b = router(x).softmax(1)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(float(router.last_diagnostics['fallback_fraction']), 0.)
        torch.testing.assert_close(train_gate[~uniform_samples], a[~uniform_samples])

    def test_initial_prediction_matches_uniform_and_original_is_unchanged(self):
        cfg = config('uniform')
        torch.manual_seed(42)
        old = V22CoarseningMoE(cfg).eval()
        new_cfg = copy.deepcopy(cfg)
        new_cfg['model']['v22'].update(mode='moe', refinement=options())
        torch.manual_seed(42)
        new = V221CoarseningMoE(new_cfg).eval()
        batch = make_batch(cfg, seed=4)
        with torch.no_grad():
            a = old(batch['x_f_obs'], batch['m_f'])['x_hat_main']
            b = new(batch['x_f_obs'], batch['m_f'])['x_hat_main']
        torch.testing.assert_close(a, b)
        self.assertEqual(cfg['model']['v22']['mode'], 'uniform')
        self.assertNotIn('refinement', cfg['model']['v22'])

    def test_missing_values_cannot_leak_and_checkpoint_round_trip(self):
        for channels in (1, 2):
            cfg = config('moe', channels=channels, h=5, w=7)
            cfg['model']['v22']['refinement'] = options(.25)
            model = V221CoarseningMoE(cfg).eval()
            x = torch.randn(2, channels, 2, 5, 7)
            mask = (torch.rand(2, 1, 2, 5, 7) > .4).float()
            hidden = x.masked_fill(~mask.expand_as(x).bool(), float('nan'))
            with torch.no_grad():
                out = model(x, mask)['x_hat_main']
                torch.testing.assert_close(out, model(hidden, mask)['x_hat_main'])
                for m in (torch.zeros_like(mask), torch.ones_like(mask)):
                    self.assertTrue(torch.isfinite(model(x, m)['x_hat_main']).all())
                clone = V221CoarseningMoE(cfg).eval()
                clone.load_state_dict(model.state_dict())
                torch.testing.assert_close(out, clone(x, mask)['x_hat_main'])

    def test_protocol_preserves_baseline_config_and_separates_ablation(self):
        policy, suite = runner.load_policy('configs/v22/refinement.json')
        before = copy.deepcopy(suite)
        work = list(runner.jobs(policy, suite, ['0', '1']))
        self.assertEqual(len(work), 24)
        self.assertEqual({(job[1], job[2], job[3]) for job in work},
                         {(dataset, pattern, '0.4') for dataset in ('BikeNYC', 'TaxiBJ', 'CHAP')
                          for pattern in ('fixed', 'random')})
        self.assertEqual({job[4] for job in work}, {42})
        for name, d, m, r, seed, cfg, _ in work:
            self.assertEqual(cfg['train']['epochs'], 60)
            self.assertEqual(cfg['train']['val_epoch'], 2)
            if name in policy['baselines']:
                original, _ = build_config(suite, name, d, m, r, seed, epochs=60, world_size=2)
                self.assertEqual(cfg, original)
                for budget in (100, 120):
                    old_budget, _ = build_config(suite, name, d, m, r, seed, epochs=budget, world_size=2)
                    self.assertNotEqual(cfg['experiment_policy']['fingerprint'], old_budget['experiment_policy']['fingerprint'])
            else:
                self.assertEqual(cfg['model']['version'], 'v22.1')
                self.assertEqual(cfg['train']['epochs'], 60)
                self.assertEqual(cfg['train']['val_epoch'], 2)
                self.assertEqual(cfg['distributed']['global_batch_size'], 8)
                self.assertEqual(cfg['output_dir'], 'outputs/v22/refinement')
                self.assertEqual(cfg['model']['v22']['refinement_source'], runner.extension_identity())
        self.assertEqual(before, suite)
        self.assertNotEqual(work[2][-2]['experiment_policy']['fingerprint'], work[3][-2]['experiment_policy']['fingerprint'])

    def test_real_cpu_ddp_train_val_best_reload_test(self):
        policy, suite = runner.load_policy('configs/v22/refinement.json')
        for name in policy['variants']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                cfg, _ = runner.refined_config(policy, suite, name, 'BikeNYC', 'random', '0.4', 42)
                cfg['device'] = 'cpu'
                cfg['output_dir'] = tmp
                cfg['train'].update(epochs=1, val_epoch=1)
                cfg['data'].update(batch_size=1, num_workers=0)
                cfg['data']['synthetic'].update(num_train=3, num_val=3, t=2, h=8, w=8)
                cfg['model']['v22'].update(dim=8, num_groups=2)
                cfg['distributed'].update(per_rank_batch_size=1, global_batch_size=2, timeout_seconds=60)
                path = Path(tmp) / 'config.json'
                path.write_text(json.dumps(cfg))
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                           str(ROOT / 'scripts/v22/run_refinement.py'), '--worker', '-c', str(path),
                           '--name', f'ablation_v22_{name}', '--synthetic', '--quiet', '--no_plot']
                result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=100, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                hit = runner.completed(cfg, name)
                self.assertIsNotNone(hit)
                records = [json.loads(line) for line in (Path(hit['run'])/'logs/metrics.jsonl').read_text().splitlines()]
                self.assertIn('v22_s2_mix_strength', records[0]['train'])
                self.assertEqual(records[0]['val']['v22_s2_fallback_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
