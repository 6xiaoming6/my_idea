"""Small CPU checks for I06-I08; no experimental queue or GPU is launched."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'src'), str(ROOT/'scripts/v25'), str(ROOT/'scripts/v24')]
import test_v25_observed_multiscale as fixtures
from run_taxibj_input_communication import jobs
from stmoe_imputer.losses import (compute_coe_loss, visible_observation_consistency_loss,
                                 full_target_observation_consistency_loss)
from stmoe_imputer.utils.deterministic import state_hash


def aggregate(value, factor):
    b,c,t,h,w = value.shape
    return value.reshape(b,c,t,h//factor,factor,w//factor,factor).sum((4,6))


class ConsistencyControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.helper = fixtures.V25Tests()
        self.base = self.helper.setup_jobs(Path(self.tmp.name))['taxibj_B2_seed7']
        self.batch = self.helper.batch(self.base)

    def job(self, method):
        job = copy.deepcopy(self.base)
        resolved = jobs(methods=[method])[f'taxibj_{method}_seed7']['config']
        for name in ('observed_history', 'observation_consistency'):
            if name in resolved['model']['coe']:
                job['config']['model']['coe'][name] = copy.deepcopy(resolved['model']['coe'][name])
        if 'lambda_observation_consistency' in resolved['loss']:
            job['config']['loss']['lambda_observation_consistency'] = resolved['loss']['lambda_observation_consistency']
        return job

    def test_eight_jobs_preserve_frozen_five_and_same_budget(self):
        current = jobs()
        self.assertEqual([j['method'] for j in current.values()], [f'I{i:02}' for i in range(1,9)])
        parent = ROOT/'outputs/v25/experiments/taxibj_input_communication'
        for suite in ('ea1ca776ade869d6','b5925acd4615e143'):
            old = json.loads((parent/suite/'plan.json').read_text())['jobs']
            for name, job in old.items():
                self.assertEqual(current[name], job)
        original = current['taxibj_I05_seed7']['config']
        for method in ('I06','I07','I08'):
            cfg = current[f'taxibj_{method}_seed7']['config']
            self.assertEqual(cfg['data'], original['data'])
            self.assertEqual(cfg['train'], original['train'])
            self.assertEqual(cfg['experiment_plan']['protocol'], original['experiment_plan']['protocol'])
            self.assertEqual(cfg['model']['coe']['core_validation'], original['model']['coe']['core_validation'])

    def test_observed_loss_forward_is_b2_and_never_reads_coarse_targets(self):
        job = self.job('I06')
        baseline, model = self.helper.model(self.base).eval(), self.helper.model(job).eval()
        self.assertEqual(state_hash(baseline.state_dict()), state_hash(model.state_dict()))
        poison = copy.deepcopy(self.batch)
        for name, factor in [('f',1),('m',2),('c',4)]:
            poison['x_'+name+'_gt'] = torch.full_like(aggregate(self.batch['x_f_gt'],factor),float('nan'))
            poison['x_'+name+'_obs'][~poison['m_'+name].bool()] = float('nan')
        with torch.no_grad():
            expected = baseline(self.batch)['x_hat_main']
            torch.testing.assert_close(model(poison)['x_hat_main'], expected, rtol=0, atol=0)
        output = {'x_hat_main': expected}
        a,_ = visible_observation_consistency_loss(output,self.batch,job['config'])
        b,_ = visible_observation_consistency_loss(output,poison,job['config'])
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_visible_loss_formula_missing_scales_and_mean_conversion(self):
        cfg = self.job('I06')['config']
        batch = copy.deepcopy(self.batch)
        pred = (batch['x_f_gt']+3.).detach().requires_grad_(True)
        outputs = {'x_hat_main':pred, 'coe':{'predictions':[pred]*4,'use_routed':False}}
        comp = torch.where(batch['m_f'].bool(),batch['x_f_obs'],pred)
        expected = torch.stack([
            (aggregate(comp,factor)[batch['m_'+name].bool()]-batch['x_'+name+'_obs'][batch['m_'+name].bool()]).abs().mean()/factor**2
            for name,factor in [('m',2),('c',4)]]).mean()
        actual,_ = visible_observation_consistency_loss(outputs,batch,cfg)
        torch.testing.assert_close(actual,expected)
        total,_ = compute_coe_loss(outputs,batch,cfg)
        base,_ = compute_coe_loss(outputs,batch,self.base['config'])
        torch.testing.assert_close(total-base,.1*expected)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))
        self.assertFalse(bool(pred.grad[batch['m_f'].bool()].any()))
        mean_cfg = copy.deepcopy(cfg);mean_cfg['data']['scales']['pooling_mode']='mean'
        mean_batch = copy.deepcopy(batch)
        for name,factor in [('m',2),('c',4)]:mean_batch['x_'+name+'_obs']/=factor**2
        mean,_ = visible_observation_consistency_loss(outputs,mean_batch,mean_cfg)
        torch.testing.assert_close(mean,actual)
        # When all coarse cells are visible, I04 and I06 agree exactly.
        for name,factor in [('m',2),('c',4)]:
            batch['m_'+name].fill_(1)
            batch['x_'+name+'_obs'] = aggregate(batch['x_f_gt'],factor)
            batch['x_'+name+'_gt'] = batch['x_'+name+'_obs']
        full,_ = full_target_observation_consistency_loss(outputs,batch,cfg)
        visible,_ = visible_observation_consistency_loss(outputs,batch,cfg)
        torch.testing.assert_close(full,visible)
        for name in ('m','c'):
            batch['m_'+name].zero_();batch['x_'+name+'_obs'].fill_(float('nan'))
        empty,_ = visible_observation_consistency_loss(outputs,batch,cfg)
        self.assertEqual(float(empty),0.)

    def test_uniform_projection_has_no_extra_parameters_and_matches_i05_initially(self):
        baseline = self.helper.model(self.base)
        uniform = self.helper.model(self.job('I07')).eval()
        learned = self.helper.model(self.job('I05')).eval()
        self.assertFalse(hasattr(uniform.main_branch,'x_consistency'))
        self.assertEqual(state_hash(uniform.state_dict()),state_hash(baseline.state_dict()))
        with torch.no_grad():
            out = uniform(self.batch)
            torch.testing.assert_close(out['x_hat_main'],learned(self.batch)['x_hat_main'],rtol=0,atol=0)
        self.assertEqual(float(out['coe']['x_consistency_diagnostics']['weight_min']),1.)
        uniform.train()
        loss,_ = compute_coe_loss(uniform(self.batch),self.batch,self.job('I07')['config'])
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in uniform.parameters()))

    def test_history_projection_is_causal_sparse_and_both_heads_learn(self):
        job = self.job('I08');model = self.helper.model(job)
        plain = self.helper.model(self.job('I05')).eval()
        history = self.helper.model(self.job('I03')).eval()
        common = {k:v for k,v in model.state_dict().items() if not k.startswith('main_branch.x_history.')}
        self.assertEqual(state_hash(common),state_hash(plain.state_dict()))
        without_projection = {k:v for k,v in model.state_dict().items() if not k.startswith('main_branch.x_consistency.')}
        self.assertEqual(state_hash(without_projection),state_hash(history.state_dict()))
        seen=[];original=model.main_branch._memory
        def check(context):
            seen.append(len(context['x_pool']))
            self.assertTrue(all(not x.requires_grad for x in context['x_pool']))
            return original(context)
        calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in model.main_branch.routed_experts()]
        try:
            with patch.object(model.main_branch,'_memory',side_effect=check):out=model(self.batch)
        finally:
            for hook in hooks:hook.remove()
        self.assertEqual(seen,[0,1,2,3]);self.assertEqual(sum(calls),8*len(self.batch['m_f']))
        with torch.no_grad():
            torch.testing.assert_close(out['x_hat_main'],plain(self.batch)['x_hat_main'],rtol=0,atol=0)
        loss,_=compute_coe_loss(out,self.batch,job['config']);loss.backward()
        for head in (model.main_branch.x_history,model.main_branch.x_consistency):
            gradients=[p.grad for p in head.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(any(bool(g.abs().sum()>0) for g in gradients))
        model.eval();poison=copy.deepcopy(self.batch)
        for name,factor in [('f',1),('m',2),('c',4)]:
            poison['x_'+name+'_gt']=torch.full_like(aggregate(self.batch['x_f_gt'],factor),float('nan'))
            poison['x_'+name+'_obs'][~poison['m_'+name].bool()]=float('nan')
        with torch.no_grad():
            torch.testing.assert_close(model(poison)['x_hat_main'],model(self.batch)['x_hat_main'],rtol=0,atol=0)


class ComparisonDependencyTests(unittest.TestCase):
    def test_reports_include_old_five_without_retraining_and_reject_tampering(self):
        from report_taxibj_input_communication import comparison_jobs
        from prepare_data import sha
        parent=ROOT/'outputs/v25/experiments/taxibj_input_communication'
        dependencies={str(parent/name):sha(parent/name/'plan.json')
                      for name in ('ea1ca776ade869d6','b5925acd4615e143')}
        current=jobs(methods=['I06','I07','I08'])
        with tempfile.TemporaryDirectory() as directory:
            suite=Path(directory);plan=suite/'plan.json'
            plan.write_text(json.dumps({'comparison_suites':dependencies}))
            entries=comparison_jobs(suite,current)
            self.assertEqual([j['method'] for _,_,j,_ in entries],[f'I{i:02}' for i in range(1,9)])
            self.assertEqual(sum(external for _,_,_,external in entries),5)
            self.assertEqual(len(current),3)
            dependencies[next(iter(dependencies))]='changed'
            plan.write_text(json.dumps({'comparison_suites':dependencies}))
            with self.assertRaisesRegex(RuntimeError,'Comparison plan changed'):
                comparison_jobs(suite,current)


if __name__ == '__main__':
    unittest.main()
