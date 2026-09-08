"""Full Idea-2 correctness, leakage, gradients, checkpoints and CPU DDP."""
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
sys.path.insert(0, str(ROOT/'scripts/v22'))
from full_model import FullCoarseningMoE, LocalCoarsener
from run_full_idea import configuration, completed
from test_v22_coarsening_moe import config


def small(routing='moe', top_k=3, channels=2):
    cfg = config(channels=channels)
    cfg['model']['v22']['full_idea'] = dict(dim=8, num_groups=2, dropout=0.,
        strides=[2, 4], radii=[0, 1, 2], routing=routing, top_k=top_k, exchange_rounds=2)
    return cfg


class FullIdeaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self, c=2, h=5, w=7):
        torch.manual_seed(5)
        return torch.randn(2, c, 3, h, w), (torch.rand(2, 1, 3, h, w) > .4).float()

    def test_routed_regions_equal_dense_observed_aggregation(self):
        model = FullCoarseningMoE(small()).eval()
        x, mask = self.sample()
        out = model(x, mask, return_regions=True)
        flat = lambda z: z.permute(0, 2, 3, 4, 1).reshape(6, 35, -1)
        normalized = flat((x-out['v22_center'])/out['v22_scale'])
        m = flat(mask)
        for state in out['regions']:
            weights, indices = state['assignment']
            torch.testing.assert_close(weights.sum(-1), torch.ones(6, 35))
            dense = weights.new_zeros(6, 35, state['mass'].shape[1])
            dense.scatter_add_(2, indices[None].expand(6, -1, -1), weights)
            observed_mass = dense.transpose(1, 2) @ m
            expected = (dense.transpose(1, 2) @ (normalized*m))/observed_mass.clamp_min(1e-6)
            torch.testing.assert_close(state['mean'], expected)
            torch.testing.assert_close(state['observed_mass'], observed_mass)
            torch.testing.assert_close(state['mass'].sum(1), torch.full((6, 1), 35.))
            torch.testing.assert_close(state['observed_mass'].sum(1), m.sum(1))
            restored = LocalCoarsener.prolong(state['tokens'], state['assignment'])
            torch.testing.assert_close(restored, dense @ state['tokens'])

    def test_hidden_gt_and_external_coarse_cannot_leak(self):
        x, mask = self.sample()
        model = FullCoarseningMoE(small()).eval()
        hidden = x.masked_fill(~mask.expand_as(x).bool(), float('nan'))
        with torch.no_grad():
            a = model(x, mask, return_regions=True)
            b = model(hidden, mask, x_m=torch.full_like(x, 1e9), x_f_gt=x+1e9, return_regions=True)
        torch.testing.assert_close(a['x_hat_main'], b['x_hat_main'], atol=0, rtol=0)
        for s, changed in zip(a['regions'], b['regions']):
            torch.testing.assert_close(s['mean'], changed['mean'], atol=0, rtol=0)
            torch.testing.assert_close(s['assignment'][0], changed['assignment'][0], atol=0, rtol=0)

    def test_task_gradients_reach_assignment_router_and_coarse_completion(self):
        x, mask = self.sample()
        model = FullCoarseningMoE(small())
        prediction = model(x, mask)['x_hat_main']
        loss = ((prediction-x).square()*(1-mask)).mean()
        loss.backward()
        region = model.regions[0]
        for param in (region.router[-1].weight, region.experts[1].query.weight,
                      region.experts[2].key.weight, region.complete.q.weight,
                      region.complete.temporal.weight, region.feedback.weight, model.fuse.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.abs().sum()), 0.)

    def test_coarse_information_changes_final_prediction(self):
        model = FullCoarseningMoE(small()).eval()
        x, mask = self.sample()
        with torch.no_grad():
            before = model(x, mask)['x_hat_main']
            model.regions[0].complete.out.bias.add_(2.)
            after = model(x, mask)['x_hat_main']
        self.assertGreater(float((before-after).abs().max()), 1e-5)

    def test_modes_empty_full_constant_checkpoint_and_two_optimizer_steps(self):
        for routing, k in [('moe', 3), ('moe', 2), ('uniform', 3), ('fixed', 3)]:
            model = FullCoarseningMoE(small(routing, k))
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            x, mask = self.sample()
            for _ in range(2):
                optimizer.zero_grad()
                model(x, mask)['x_hat_main'].square().mean().backward()
                self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
                optimizer.step()
            model.eval()
            for v in (0., 1.):
                out = model(torch.full_like(x, 3.), torch.full_like(mask, v))
                self.assertTrue(torch.isfinite(out['x_hat_main']).all())
            clone = FullCoarseningMoE(small(routing, k)).eval()
            clone.load_state_dict(copy.deepcopy(model.state_dict()))
            torch.testing.assert_close(model(x, mask)['x_hat_main'], clone(x, mask)['x_hat_main'])

    def test_dataset_shapes_and_bfloat16(self):
        for channels, h, w in [(2, 32, 32), (2, 24, 12), (1, 32, 32)]:
            x = torch.randn(1, channels, 2, h, w)
            mask = (torch.rand(1, 1, 2, h, w) > .4).float()
            model = FullCoarseningMoE(small(channels=channels))
            with torch.autocast('cpu', dtype=torch.bfloat16):
                out = model(x, mask)['x_hat_main']
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_policy_and_cpu_train_val_best_reload_test(self):
        policy = json.loads((ROOT/'configs/v22/full_idea.json').read_text())
        for world in (1, 2):
            cfg, _ = configuration(policy, 'full', 'BikeNYC', 'random', '0.4', 42, world, cpu=True, smoke=True)
            self.assertEqual(cfg['distributed']['world_size'], world)
            self.assertEqual(cfg['train']['epochs'], 1)
            self.assertIn('full_idea/smoke', cfg['output_dir'])
            with tempfile.TemporaryDirectory() as directory:
                cfg['output_dir'] = directory
                cfg['train']['epochs'] = 2
                path = Path(directory)/'input.json'
                path.write_text(json.dumps(cfg))
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                           f'--nproc_per_node={world}', str(ROOT/'scripts/v22/run_full_idea.py'),
                           '--worker', '-c', str(path), '--name', 'ablation_v22_full_idea_full',
                           '--synthetic', '--quiet', '--no_plot']
                result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=120, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                self.assertEqual(result.returncode, 0, result.stdout[-8000:])
                hit = completed(cfg, 'full_idea_full')
                self.assertIsNotNone(hit)
                run = Path(hit['run'])
                self.assertEqual(len(list((run/'checkpoints').glob('*.pt'))), 1)
                self.assertTrue(all((run/'logs'/f'{s}.log').is_file() for s in ('train', 'val', 'test')))

    @unittest.skipUnless(os.environ.get('V22_FULL_REAL_SMOKE') == '1', 'Opt-in real windows, CPU only')
    def test_real_windows_all_three_datasets_and_patterns(self):
        from test_v22_real_smoke import first_batch
        from stmoe_imputer.models.imputer import DualBranchSTImputer
        from stmoe_imputer.models.aux_branch import NullResidualBranch
        from stmoe_imputer.engine import build_optimizer, train_one_epoch, evaluate
        policy = json.loads((ROOT/'configs/v22/full_idea.json').read_text())
        for dataset, pattern, rate in policy['points']:
            with self.subTest(dataset=dataset, pattern=pattern):
                cfg, paths = configuration(policy, 'full', dataset, pattern, rate, 42, cpu=True)
                batches = {s: first_batch(cfg, path, cfg['data']['mask'][f'{s}_csv']) for s, path in paths.items()}
                torch.manual_seed(42)
                model = DualBranchSTImputer(FullCoarseningMoE(cfg), NullResidualBranch(cfg['model']['c_in']))
                optimizer = build_optimizer(model, cfg)
                train = train_one_epoch(model, [batches['train']], optimizer, torch.device('cpu'), cfg, 1)
                val = evaluate(model, [batches['val']], torch.device('cpu'), cfg, epoch=1)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'best.pt'
                    torch.save(model.state_dict(), path)
                    model.load_state_dict(torch.load(path, weights_only=True))
                test = evaluate(model, [batches['test']], torch.device('cpu'), cfg, epoch=1)
                for result in (train, val, test):
                    self.assertTrue(all(torch.isfinite(torch.tensor(result[k])) for k in ('loss', 'mae', 'rmse')))
                print(f'FULL_REAL_SMOKE {dataset} {pattern}: train={train["loss"]:.6f} val_mae={val["mae"]:.6f} test_mae={test["mae"]:.6f}', flush=True)


if __name__ == '__main__':
    unittest.main()
