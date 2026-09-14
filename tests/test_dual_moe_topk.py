import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from test_dual_moe import ROOT, batch, config
from stmoe_imputer.config import deep_update, load_config
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.dual_moe import PointRouter
from stmoe_imputer.losses import token_topk_balance_loss, compute_main_stage_loss
from stmoe_imputer.routing_metrics import DualMoEMetricAccumulator
from stmoe_imputer.engine import train_one_epoch, evaluate, build_optimizer
from stmoe_imputer.utils.checkpoint import save_checkpoint, load_checkpoint, snapshot_model_state


def topk_config(channels=2):
    cfg = config(channels)
    cfg['model']['dual_moe'].update(design='learned_regions_v2', aggregation_mode='topk', aggregation_top_k=2,
                                   completion_mode='topk', completion_top_k=2)
    cfg['loss'].update(dual_moe_expert_weight=.01, dual_moe_aggregation_balance_weight=.01,
                       dual_moe_completion_balance_weight=.01)
    return cfg


class TopKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_exact_selection_gradients_and_invalid_k(self):
        for k in (1, 2, 3):
            router = PointRouter(4, 3, 'topk', k)
            x = torch.randn(2, 4, 2, 3, 4)
            gates = router(x)
            torch.testing.assert_close(gates.sum(1), torch.ones_like(gates[:, 0]))
            self.assertTrue(((gates > 0).sum(1) == k).all())
            selected = router.routing_info['selected']
            self.assertFalse(selected.requires_grad)
            self.assertTrue(router.routing_info['probabilities'].requires_grad)
            (gates*torch.arange(1, 4).reshape(1, 3, 1, 1, 1)).sum().backward()
            self.assertGreater(float(router.head.weight.grad.abs().sum()), 0)
            router.eval(); torch.testing.assert_close(router(x), router(x), rtol=0, atol=0)
        for k in (None, 0, 4, True, 1.5):
            with self.assertRaises(ValueError): PointRouter(4, 3, 'topk', k)

    def test_memory_snapshot_is_independent_and_restores_buffers(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
        state = snapshot_model_state(model)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        self.assertTrue(all(not v.requires_grad and v.device.type == 'cpu' for v in state.values()))
        with torch.no_grad():
            for v in model.state_dict().values(): v.add_(1)
        for k in state: torch.testing.assert_close(state[k], original[k], rtol=0, atol=0)
        model.load_state_dict(state)
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, original[k], rtol=0, atol=0)

    def test_memory_best_full_training_matches_disk_without_checkpoint_files(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        import run_scale_completion_experiments as runner
        tests = []
        with tempfile.TemporaryDirectory() as tmp:
            for persist in (True, False):
                cfg = topk_config(); cfg['output_dir'] = str(Path(tmp)/str(persist))
                cfg['train'].update(epochs=3, val_epoch=1, save_best_checkpoint=persist)
                path = Path(tmp)/f'{persist}.json'; path.write_text(json.dumps(cfg))
                result = subprocess.run([sys.executable, str(ROOT/'scripts/train.py'), '-c', str(path), '--synthetic', '--no_plot', '--quiet', '--name', 'ablation_memory_test'], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:]+result.stderr[-2000:])
                run = next(Path(cfg['output_dir']).rglob('metrics.jsonl')).parent.parent
                records = [json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()]
                test = next(r for r in records if r.get('stage') == 'test'); tests.append(test)
                self.assertEqual(len(list(run.rglob('*.pt'))), int(persist))
                if not persist: self.assertFalse((run/'checkpoints').exists())
                saved_cfg = json.loads((run/'config.json').read_text())
                job = {'cfg': saved_cfg, 'name': 'memory_test'}
                self.assertIsNotNone(runner.completed(job))
                if not persist:
                    self.assertEqual(test['extra']['best_model_source'], 'memory')
                    self.assertTrue(test['extra']['best_weights_restored'])
                    self.assertIn('CPU memory', (run/'logs/train.log').read_text())
                    test['extra']['best_weights_restored'] = False
                    (run/'logs/metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
                    self.assertIsNone(runner.completed(job))
            self.assertEqual(tests[0]['metrics'], tests[1]['metrics'])
            self.assertEqual(tests[0]['extra']['best_epoch'], tests[1]['extra']['best_epoch'])

    def test_larger_front_pools_routing_metrics_and_matched_initialization(self):
        for e in (3, 4, 6, 8):
            cfg = topk_config(); cfg['model']['dual_moe']['aggregation_experts'] = e
            torch.manual_seed(42); model = DualBranchSTImputer.from_config(cfg)
            uniform_cfg = copy.deepcopy(cfg)
            uniform_cfg['model']['dual_moe']['aggregation_mode'] = 'uniform'
            uniform_cfg['loss']['dual_moe_aggregation_balance_weight'] = 0
            torch.manual_seed(42); uniform = DualBranchSTImputer.from_config(uniform_cfg)
            for name, value in model.state_dict().items():
                if name.startswith('main_branch.aggregation.') and name.endswith('router.head.weight'): continue
                torch.testing.assert_close(value, uniform.state_dict()[name], rtol=0, atol=0)
            data = batch(n=1)
            for current, config_used in [(model, cfg), (uniform, uniform_cfg)]:
                output = current(data)
                for group in ('mid', 'coarse'):
                    self.assertEqual(output['aggregation_gates'][group].shape[1], e)
                    self.assertEqual(output['region_assignments'][group].shape[1], e)
                loss, _ = compute_main_stage_loss(output, data, config_used); loss.backward()
                self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in current.parameters() if x.requires_grad))
                acc = DualMoEMetricAccumulator(); acc.update(output, data); metrics = acc.compute()
                self.assertIn(f'aggregation_mid_observed_e{e-1}_mean', metrics)
                if current is model:
                    for group in ('aggregation_mid', 'aggregation_coarse'):
                        self.assertAlmostEqual(metrics[f'topk_{group}_selected_per_token'], 2)
                        self.assertAlmostEqual(sum(metrics[f'topk_{group}_e{i}_load'] for i in range(e)), 1)
                self.assertAlmostEqual(metrics['topk_completion_selected_per_token'], 2)
            for group in ('mid', 'coarse'):
                router = model.main_branch.aggregation[group].router
                with torch.no_grad():
                    router.head.weight.zero_(); router.head.bias.copy_(torch.arange(e, dtype=torch.float))
                route = model(data)['routing_details'][f'aggregation_{group}']
                penalty = token_topk_balance_loss(route, data['m_f'])
                grad = torch.autograd.grad(penalty, router.head.bias)[0]
                self.assertGreater(float(grad.abs().sum()), 0)
                self.assertLess(float(grad[0]), 0)
        for e in (0, -1, True, 3.5):
            cfg = topk_config(); cfg['model']['dual_moe']['aggregation_experts'] = e
            with self.assertRaises(ValueError): DualBranchSTImputer.from_config(cfg)

    def test_eight_expert_top4_full_softmax_and_equal_initialization(self):
        reference = None
        for k in (2, 4, 8):
            cfg = topk_config(); cfg['model']['dual_moe'].update(aggregation_experts=8, aggregation_top_k=k)
            if k==8: cfg['loss']['dual_moe_aggregation_balance_weight'] = 0
            torch.manual_seed(42); model = DualBranchSTImputer.from_config(cfg)
            if reference is None: reference = {n:v.clone() for n,v in model.state_dict().items()}
            else:
                for n,v in model.state_dict().items(): torch.testing.assert_close(v, reference[n], rtol=0, atol=0)
            data = batch(n=1); output = model(data)
            for group in ('mid', 'coarse'):
                gate = output['aggregation_gates'][group]
                self.assertTrue(((gate>0).sum(1)==k).all())
                if k==8:
                    torch.testing.assert_close(gate, output['routing_details'][f'aggregation_{group}']['probabilities'], rtol=0, atol=0)
            loss, logs = compute_main_stage_loss(output, data, cfg); loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
            if k==8: self.assertEqual(float(logs['l_balance_aggregation_weighted']), 0.)

    def test_eight_expert_training_validation_best_reload_and_logs(self):
        cfg = topk_config(); cfg['model']['dual_moe']['aggregation_experts'] = 8
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir'] = tmp; path = Path(tmp)/'input.json'; path.write_text(json.dumps(cfg))
            result = subprocess.run([sys.executable, str(ROOT/'scripts/train.py'), '-c', str(path), '--synthetic', '--no_plot', '--quiet', '--name', 'smoke_e8'], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:]+result.stderr[-2000:])
            checkpoints = list(Path(tmp).rglob('best.pt')); self.assertEqual(len(checkpoints), 1)
            run = checkpoints[0].parent.parent
            records = [json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(sum('epoch' in r for r in records), 2)
            test = [r for r in records if r.get('stage') == 'test']; self.assertEqual(len(test), 1)
            self.assertIn('topk_aggregation_mid_e7_load', test[0]['metrics'])
            for file in ('train.log', 'val.log', 'test.log'):
                content = (run/'logs'/file).read_text()
                self.assertIn('balance', content); self.assertIn('topk', content)

    def test_balance_has_real_router_gradient_and_ignores_excluded_tokens(self):
        logits = torch.tensor([3., 2., -2.]).reshape(1, 3, 1, 1, 1).expand(2, 3, 1, 2, 2).clone().requires_grad_()
        mask = torch.ones(2, 1, 1, 2, 2); mask[1] = 0
        def make(z):
            return {'probabilities': z.softmax(1), 'selected': torch.zeros_like(z).scatter_(1, z.topk(2, dim=1).indices, 1.), 'top_k': 2}
        loss = token_topk_balance_loss(make(logits), mask)
        expected = 3*(logits.softmax(1)[0, :2].mean((1, 2, 3))*.5).sum()
        torch.testing.assert_close(loss, expected)
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertGreater(float(gradient[0, :2].sum()), 0)
        self.assertLess(float(gradient[0, 2].sum()), 0)
        self.assertEqual(float(gradient[1].abs().sum()), 0)
        changed = logits.detach().clone(); changed[1] *= -100
        torch.testing.assert_close(loss, token_topk_balance_loss(make(changed), mask))
        self.assertLess(float(token_topk_balance_loss(make(logits-.1*gradient), mask)), float(loss))
        empty = token_topk_balance_loss(make(logits), torch.zeros_like(mask))
        self.assertEqual(float(empty), 0)
        self.assertTrue(torch.isfinite(torch.autograd.grad(empty, logits)[0]).all())

    def test_shared_initialization_and_k_equals_experts_dense_equivalence(self):
        cfg = config(); cfg['model']['dual_moe'].update(aggregation_mode='learned', completion_mode='learned')
        torch.manual_seed(42); dense = DualBranchSTImputer.from_config(cfg).eval()
        sparse_cfg = topk_config()
        sparse_cfg['model']['dual_moe'].update(aggregation_top_k=3, completion_top_k=3)
        torch.manual_seed(42); sparse = DualBranchSTImputer.from_config(sparse_cfg).eval()
        for name, value in dense.state_dict().items():
            if not name.endswith('router.head.weight'):
                torch.testing.assert_close(value, sparse.state_dict()[name], rtol=0, atol=0)
        sparse.load_state_dict(dense.state_dict())
        data = batch()
        torch.testing.assert_close(dense(data)['x_hat_final'], sparse(data)['x_hat_final'], rtol=0, atol=0)

    def test_both_balance_terms_enter_total_loss_and_both_router_gradients(self):
        cfg = topk_config(); model = DualBranchSTImputer.from_config(cfg)
        routers = [a.router for a in model.main_branch.aggregation.values()]+[model.main_branch.completion_router]
        with torch.no_grad():
            for router in routers: router.head.bias.copy_(torch.tensor([3., 2., -2.]))
        data = batch(); out = model(data)
        total, logs = compute_main_stage_loss(out, data, cfg)
        no_balance = copy.deepcopy(cfg)
        no_balance['loss'].update(dual_moe_aggregation_balance_weight=0, dual_moe_completion_balance_weight=0)
        base, _ = compute_main_stage_loss(out, data, no_balance)
        torch.testing.assert_close(total-base, logs['l_balance_aggregation_weighted']+logs['l_balance_completion_weighted'])
        for name, router in zip(('aggregation_mid', 'aggregation_coarse', 'completion'), routers):
            valid = data['m_f'] if name.startswith('aggregation') else 1-data['m_f']
            loss = token_topk_balance_loss(out['routing_details'][name], valid)
            grad = torch.autograd.grad(loss, router.head.bias, retain_graph=True)[0]
            self.assertGreater(float(grad.abs().sum()), 0)
        invalid = copy.deepcopy(cfg); invalid['loss']['dual_moe_aggregation_balance_weight'] = -1
        with self.assertRaises(ValueError): compute_main_stage_loss(out, data, invalid)
        old = DualBranchSTImputer.from_config(config())(data)
        with self.assertRaises(ValueError): compute_main_stage_loss(old, data, cfg)

    def test_shapes_gradients_no_hidden_input_leak_and_empty_masks(self):
        for c, t, h, w in [(2, 12, 32, 32), (2, 12, 24, 12), (1, 7, 32, 32), (1, 2, 7, 11), (1, 1, 1, 1)]:
            cfg = topk_config(c); model = DualBranchSTImputer.from_config(cfg)
            data = batch(c, t, h, w, n=1); out = model(data)
            self.assertTrue(torch.isfinite(out['x_hat_final']).all())
            for name, route in out['routing_details'].items():
                self.assertTrue((route['selected'].sum(1) == 2).all())
            loss, _ = compute_main_stage_loss(out, data, cfg); loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
            changed = copy.deepcopy(data); changed['x_f_gt'].fill_(float('nan'))
            changed['x_f_obs'] = torch.where(data['m_f'].bool(), data['x_f_obs'], torch.full_like(data['x_f_obs'], float('nan')))
            torch.testing.assert_close(out['x_hat_final'], model(changed)['x_hat_final'], rtol=0, atol=0)
        for observed in (0, 1):
            cfg = topk_config(); model = DualBranchSTImputer.from_config(cfg); data = batch()
            data['m_f'].fill_(observed); data['x_f_obs'] = data['x_f_gt']*observed
            out = model(data); loss, logs = compute_main_stage_loss(out, data, cfg)
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            self.assertEqual(float(logs['l_balance_aggregation' if observed == 0 else 'l_balance_completion']), 0)

    def test_routing_metrics_exact_across_batch_partitions_and_checkpoint(self):
        cfg = topk_config(); model = DualBranchSTImputer.from_config(cfg).eval(); data = batch(n=3)
        whole, parts = DualMoEMetricAccumulator(), DualMoEMetricAccumulator()
        whole.update(model(data), data)
        for lo, hi in [(0, 2), (2, 3)]:
            small = {k: v[lo:hi] for k, v in data.items()}; parts.update(model(small), small)
        a, b = whole.compute(), parts.compute()
        for k in a: self.assertAlmostEqual(a[k], b[k], delta=1e-4*max(1, abs(a[k])))
        for name in ('aggregation_mid', 'aggregation_coarse', 'completion'):
            self.assertAlmostEqual(a[f'topk_{name}_selected_per_token'], 2)
            self.assertAlmostEqual(sum(a[f'topk_{name}_e{e}_load'] for e in range(3)), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'best.pt'; save_checkpoint(path, model, None, 1, {}, cfg)
            clone = DualBranchSTImputer.from_config(cfg); load_checkpoint(path, clone, map_location='cpu')
            torch.testing.assert_close(model(data)['x_hat_final'], clone(data)['x_hat_final'], rtol=0, atol=0)

    def test_training_entry_saves_actual_losses_and_load_in_logs(self):
        cfg = topk_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir'] = tmp; path = Path(tmp)/'input.json'; path.write_text(json.dumps(cfg))
            result = subprocess.run([sys.executable, str(ROOT/'scripts/train.py'), '-c', str(path), '--synthetic', '--no_plot', '--quiet', '--name', 'smoke_topk'], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:]+result.stderr[-2000:])
            checkpoints = list(Path(tmp).rglob('best.pt')); self.assertEqual(len(checkpoints), 1)
            run = checkpoints[0].parent.parent
            records = [json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(sum('epoch' in r for r in records), 2)
            self.assertEqual(sum(r.get('stage') == 'test' for r in records), 1)
            for file in ('train.log', 'val.log', 'test.log'):
                text = (run/'logs'/file).read_text()
                self.assertIn('balance', text); self.assertIn('topk', text)

    @unittest.skipUnless(os.environ.get('TOPK_REAL_TEST') == '1' and torch.cuda.is_available(), 'Opt-in three datasets fixed/random CUDA AMP check')
    def test_real_train_val_reload_test(self):
        import numpy as np
        for folder, prefix, c in [('TaxiBJ', 'taxibj', 2), ('BikeNYC', 'bikenyc', 2), ('CHAP/beijing', 'chap_beijing', 1)]:
            values = {}
            for split in ('train', 'val', 'test'):
                with np.load(ROOT/f'data/{folder}/{prefix}_{split}.npz', allow_pickle=False) as f:
                    values[split] = torch.from_numpy(f['x_f_gt'][:1].copy()).float()
            for pattern in ('fixed', 'random'):
                cfg = deep_update(config(c), load_config(ROOT/'configs/presets/dual_moe_topk.json'))
                model = DualBranchSTImputer.from_config(cfg).cuda(); optimizer = build_optimizer(model, cfg); splits = {}
                for split, x in values.items():
                    a = np.loadtxt(ROOT/f'data/{folder}/{pattern}_mask/0.4/{split}.csv', delimiter=',', max_rows=1, ndmin=2)
                    _, _, t, h, w = x.shape
                    mask = torch.from_numpy(a.copy()).float().reshape(1, 1, 1 if a.size == h*w else t, h, w).expand(1, 1, t, h, w)
                    splits[split] = {k: v.cuda() for k, v in {'x_f_gt': x, 'x_f_obs': x*mask, 'm_f': mask}.items()}
                best = float('inf')
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'best.pt'
                    for epoch in (1, 2):
                        train = train_one_epoch(model, [splits['train']], optimizer, torch.device('cuda'), cfg, epoch)
                        val = evaluate(model, [splits['val']], torch.device('cuda'), cfg)
                        if val['mae'] < best:
                            best = val['mae']; save_checkpoint(path, model, optimizer, epoch, val, cfg)
                    load_checkpoint(path, model, map_location='cuda')
                    test = evaluate(model, [splits['test']], torch.device('cuda'), cfg)
                for logs in (train, val, test):
                    self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in logs.values()))
                    self.assertAlmostEqual(logs['topk_completion_selected_per_token'], 2)


if __name__ == '__main__': unittest.main()
