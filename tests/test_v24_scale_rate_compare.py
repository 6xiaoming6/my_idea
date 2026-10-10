import copy
import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/v24')]
from run_scale_rate_compare import jobs, run_directory, launch, prepare_initialization
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash, configure


class ScaleRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.jobs = jobs()

    def tiny(self, name):
        j = copy.deepcopy(self.jobs[name])
        j['config']['model']['main'].update(dim=8, h=8, w=8)
        j['config']['model']['coe']['router_hidden_dim'] = 8
        return j

    def model(self, j):
        torch.manual_seed(7)
        return DualBranchSTImputer.from_config(j['config'])

    def batch(self, n=2):
        g = torch.Generator().manual_seed(31)
        x = torch.randn(n, 2, 3, 8, 8, generator=g)
        mask = (torch.rand(x.shape, generator=g) > .4).float()
        return {'x_f_gt': x, 'x_f_obs': x*mask, 'm_f': mask}

    def test_sixteen_pairs_rates_and_paths(self):
        self.assertEqual(len(self.jobs), 16)
        for dataset in ('taxibj', 'bikenyc'):
            for rate in (.2, .4, .6, .8):
                prefix = f'{dataset}_r{round(rate*100):02}'
                pair = [copy.deepcopy(self.jobs[prefix+'_'+s]['config']) for s in ('cmff', 'top1')]
                for c in pair:
                    self.assertEqual(c['data']['mask']['missing_rate'], rate)
                    for key in ('train_mask_diversity', 'eval_mask_diversity'):
                        self.assertEqual(c['data'][key]['rates'], [rate])
                        self.assertEqual(len(c['data'][key]['families']), 4)
                    self.assertEqual(c['experiment_plan']['protocol']['rate'], rate)
                    c.pop('experiment_plan')
                    for key in ('scale_policy', 'scale_soft_warmup_epochs', 'scale_soft_transition_epochs'):
                        c['model']['coe']['id_priority'].pop(key, None)
                self.assertEqual(*pair)
                self.assertIn(f'/random/rate{rate:g}/', str(run_directory(self.jobs[prefix+'_top1'], 'stamp')))

    def test_initial_equality_gradients_and_all_paths(self):
        fixed = self.tiny('taxibj_r40_cmff'); free = self.tiny('taxibj_r40_top1')
        a, b = self.model(fixed), self.model(free)
        self.assertEqual(state_hash(a.state_dict()), state_hash(b.state_dict()))
        a.eval(); b.eval(); batch = self.batch()
        torch.testing.assert_close(a(batch)['x_hat_main'], b(batch)['x_hat_main'], rtol=0, atol=0)
        b.train(); b.main_branch.set_routing_epoch(11)
        out = b(batch); loss, _ = compute_coe_loss(out, batch, free['config']); loss.backward()
        grads = [p.grad for p in b.main_branch.scale_heads.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        b.eval(); paths = list(itertools.product(range(3), repeat=4)); batch = self.batch(81)
        calls = []
        hooks = [e.register_forward_pre_hook(lambda e, args: calls.append(len(args[0]))) for e in b.main_branch.routed_experts()]
        with torch.no_grad(): out = b.main_branch(batch['x_f_obs'], batch['m_f'], forced_scales=torch.tensor(paths))
        for hook in hooks: hook.remove()
        self.assertEqual(sum(calls), 81*8)
        self.assertEqual(out['coe']['triscale_choices'].tolist(), [list(p) for p in paths])
        self.assertTrue((out['coe']['triscale_executed'].sum(-1) == 1).all())

    def test_fixed_matches_w_baseline(self):
        from run_backbone_exploration import jobs as wjobs
        w = wjobs()['W01']; w['config']['model']['main'].update(dim=8, h=8, w=8)
        w['config']['model']['coe']['router_hidden_dim'] = 8
        a = self.model(w).eval(); b = self.model(self.tiny('taxibj_r40_cmff')).eval()
        with torch.no_grad(): torch.testing.assert_close(a(self.batch())['x_hat_main'], b(self.batch())['x_hat_main'], rtol=0, atol=0)

    def test_soft_start_execution_and_eval(self):
        j = self.tiny('taxibj_r40_top1'); model = self.model(j); b = self.batch()
        for epoch, expected, mass in ((1, 24, 1.), (5, 24, 1.), (6, 24, 5/6), (10, 24, 1/6), (11, 8, 0.)):
            model.train(); model.main_branch.set_routing_epoch(epoch)
            calls = []
            hooks = [e.register_forward_pre_hook(lambda e, a: calls.append(len(a[0]))) for e in model.main_branch.routed_experts()]
            out = model(b)
            for h in hooks: h.remove()
            self.assertEqual(sum(calls), len(b['m_f'])*expected)
            self.assertAlmostEqual(model.main_branch.scale_soft_mass(), mass)
            torch.testing.assert_close(out['coe']['triscale_weights'].sum(-1), torch.ones(2,4))
            ctx = model.main_branch._initialize(b['x_f_obs'], b['m_f'])
            ctx, row = model.main_branch._round(ctx, 0)
            torch.testing.assert_close(ctx['counts'].sum(-1), torch.ones(2))
            model.zero_grad(); loss, _ = compute_coe_loss(out, b, j['config']); loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.main_branch.scale_heads.parameters()))
            model.eval()
            with torch.no_grad(): result = model(b)
            self.assertTrue((result['coe']['triscale_executed'].sum(-1)==1).all())
        saved = model.state_dict(); other = self.model(j); other.load_state_dict(saved)
        self.assertEqual(int(other.main_branch.stage_epoch), 11)

    def test_resume_skip_and_report(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from report_scale_rate_compare import export_report
        j = self.tiny('taxibj_r20_top1'); c = j['config']
        c['train'].update(epochs=2, val_epoch=1); c['train']['scheduler']['total_epochs'] = 2
        c['data'].update(batch_size=2, pin_memory=False)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); suite = root/'suite'; suite.mkdir()
            path = root/'tiny.npz'; np.savez(path, x_f_gt=np.random.RandomState(1).randn(4,2,3,8,8).astype('float32'))
            j['sources'] = {s: str(path) for s in ('train', 'val', 'test')}
            prepare_initialization(suite, {j['variant']: j})
            run = root/'resume'; receipt = suite/'results'/f'{j["variant"]}.json'
            configure(c)
            train(j, run, receipt, 'cpu', stop_after=1); train(j, run, receipt, 'cpu')
            train(j, root/'full', root/'full.json', 'cpu')
            a = torch.load(run/'checkpoints/last.pth', weights_only=False)
            b = torch.load(root/'full/checkpoints/last.pth', weights_only=False)
            for k in ('model', 'optimizer', 'scaler', 'scheduler', 'rng_states'):
                self.assertEqual(state_hash(a[k]), state_hash(b[k]), k)
            with patch('run_four_direction_exploration.subprocess_run', side_effect=AssertionError('must skip')):
                launch(suite, j, 0)
            group = suite/'evaluations_rate0.2'/f'{j["variant"]}.json'
            evaluate_sets(run/'checkpoints/best.pth', c['experiment_plan']['protocol'], str(path), group, device='cpu')
            report = export_report(root, suite, {j['variant']: j})
            self.assertIn('完整批次', report.read_text())
            self.assertLess(report.read_text().index('具体做法'), report.read_text().index('完整结果'))


if __name__ == '__main__': unittest.main()
