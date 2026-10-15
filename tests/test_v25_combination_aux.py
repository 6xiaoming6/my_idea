"""I13-I22: isolated auxiliaries, sparse execution, diagnostics and replay."""
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
from stmoe_imputer.losses import compute_coe_loss, visible_observation_consistency_loss, supervision_mask
from stmoe_imputer.utils.deterministic import state_hash
from preflight_taxibj_input_communication import check_consistency
from report_taxibj_input_communication import combination_ranking, OOD_CATEGORIES


class CombinationAuxTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.helper = fixtures.V25Tests()
        self.base = self.helper.setup_jobs(self.root)['taxibj_B2_seed7']
        self.batch = self.helper.batch(self.base)

    def job(self, method):
        j = copy.deepcopy(self.base)
        full = jobs(methods=[method])[f'taxibj_{method}_seed7']
        j.update(method=method, variant=full['variant'])
        for key in ('scale_delta_exploration','observed_history','observation_consistency','core_validation'):
            if key in full['config']['model']['coe']:
                j['config']['model']['coe'][key] = copy.deepcopy(full['config']['model']['coe'][key])
        j['config']['loss'] = copy.deepcopy(full['config']['loss'])
        return j

    def test_config_budget_and_single_factor_controls(self):
        selected = jobs(methods=[f'I{i:02}' for i in range(13,23)])
        self.assertEqual([j['method'] for j in selected.values()], [f'I{i:02}' for i in range(13,23)])
        self.assertEqual(len(jobs()), 8)
        base = selected['taxibj_I13_seed7']['config']
        for j in selected.values():
            for key in ('train','data','seed'):
                self.assertEqual(j['config'][key], base[key])
            self.assertEqual(j['config']['experiment_plan']['protocol'], base['experiment_plan']['protocol'])
        for method in ('I17','I18','I19','I20','I21','I22'):
            a = copy.deepcopy(base); b = copy.deepcopy(selected[f'taxibj_{method}_seed7']['config'])
            a.pop('experiment_plan'); b.pop('experiment_plan')
            if method in ('I17','I18'): b['loss']['visible_consistency_aux']['weight'] = .1
            if method == 'I19': b['model']['coe']['observation_consistency'].pop('weight_mode')
            if method == 'I20': b['model']['coe']['observed_history'].pop('gain')
            if method == 'I21': b['model']['coe']['core_validation']['communication'] = 'conditional'
            if method == 'I22': b['loss'].pop('lambda_raw_f')
            self.assertEqual(a,b,method)

    def test_old_behavior_off_and_i13_forward_parity(self):
        base, new = self.job('I08'), self.job('I13')
        a, b = self.helper.model(base).eval(), self.helper.model(new).eval()
        self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
        out_a, out_b = a(self.batch), b(self.batch)
        torch.testing.assert_close(out_a['x_hat_main'],out_b['x_hat_main'],rtol=0,atol=0)
        cfg = copy.deepcopy(base['config'])
        cfg['loss'].update(visible_consistency_aux={'weight':0,'source':'pre_projection'},lambda_raw_f=0)
        expected, logs = compute_coe_loss(out_a,self.batch,base['config'])
        actual, newlogs = compute_coe_loss(out_b,self.batch,cfg)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(state_hash(logs),state_hash(newlogs))
        loss, _ = compute_coe_loss(out_b,self.batch,new['config'])
        aux, _ = visible_observation_consistency_loss({'x_hat_main':out_b['x_hat_raw_consistency']},self.batch,cfg)
        torch.testing.assert_close(loss,expected+.1*aux)
        gradients = torch.autograd.grad(aux,tuple(b.main_branch.observation_fusion.parameters()),allow_unused=True)
        self.assertTrue(any(g is not None and bool(g.abs().sum()>0) for g in gradients))

    def test_aux_source_formula_poison_empty_and_mean(self):
        cfg = self.job('I13')['config']; batch = copy.deepcopy(self.batch)
        raw = (batch['x_f_gt']+3).requires_grad_()
        post = batch['x_f_gt'].clone()
        out = {'x_hat_main':post,'x_hat_raw_consistency':raw,
               'coe':{'predictions':[post]*4,'use_routed':False}}
        completed = torch.where(batch['m_f'].bool(),batch['x_f_obs'],raw)
        expected = []
        for scale,factor in (('m',2),('c',4)):
            b,c,t,h,w = raw.shape
            agg = completed.reshape(b,c,t,h//factor,factor,w//factor,factor).sum((4,6))
            mask = batch['m_'+scale].bool()
            expected.append((agg[mask]-batch['x_'+scale+'_obs'][mask]).abs().sum()/mask.sum()/factor**2)
        aux = torch.stack(expected).mean()
        loss, logs = compute_coe_loss(out,batch,cfg)
        torch.testing.assert_close(loss,.1*aux)
        loss.backward(); self.assertGreater(float(raw.grad.abs().sum()),0)
        self.assertFalse(bool(raw.grad[batch['m_f'].bool()].any()))
        poison = copy.deepcopy(batch)
        for scale in ('m','c'):
            poison['x_'+scale+'_gt'] = torch.full_like(batch['x_'+scale+'_obs'],float('nan'))
            poison['x_'+scale+'_obs'][~batch['m_'+scale].bool()] = float('nan')
        torch.testing.assert_close(compute_coe_loss(out,poison,cfg)[0],loss,rtol=0,atol=0)
        mean_cfg = copy.deepcopy(cfg); mean_cfg['data']['scales']['pooling_mode']='mean'
        mean_batch = copy.deepcopy(batch)
        for scale,factor in (('m',2),('c',4)): mean_batch['x_'+scale+'_obs']/=factor**2
        torch.testing.assert_close(compute_coe_loss(out,mean_batch,mean_cfg)[0],loss)
        for scale in ('m','c'):
            poison['m_'+scale].zero_(); poison['x_'+scale+'_obs'].fill_(float('nan'))
        self.assertEqual(float(compute_coe_loss(out,poison,cfg)[0]),0.)
        # Empty M retains the fixed half weight on the observed C term.
        one = copy.deepcopy(batch); one['m_m'].zero_()
        torch.testing.assert_close(compute_coe_loss(out,one,cfg)[0],.05*expected[1])
        del out['x_hat_raw_consistency']
        with self.assertRaisesRegex(ValueError,'requires x_hat_raw_consistency'): compute_coe_loss(out,batch,cfg)
        legacy = self.job('I06')['config']; legacy['loss']['visible_consistency_aux']={'weight':.1,'source':'prediction'}
        with self.assertRaisesRegex(ValueError,'legacy'): compute_coe_loss(out,batch,legacy)

    def test_all_new_models_sparse_projection_and_gradients(self):
        for method in [f'I{i:02}' for i in range(13,23)]:
            with self.subTest(method=method):
                j = self.job(method); model = self.helper.model(j).train(); calls=[]
                hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in model.main_branch.routed_experts()]
                try: out=model(self.batch)
                finally:
                    for hook in hooks: hook.remove()
                self.assertEqual(sum(calls),8*len(self.batch['m_f']))
                check_consistency(out,self.batch,j['config'])
                if method=='I14': self.assertTrue(bool((out['coe']['triscale_choices'][:,3]==0).all()))
                if method=='I19': self.assertFalse(hasattr(model.main_branch,'x_consistency'))
                if method=='I15': self.assertFalse(hasattr(model.main_branch,'x_history'))
                if method=='I16': self.assertNotIn('x_hat_raw_consistency',out)
                loss,logs=compute_coe_loss(out,self.batch,j['config']); loss.backward()
                self.assertTrue(torch.isfinite(loss)); self.assertIn('l_visible_consistency_aux',logs)
                if method=='I21':
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.main_branch.history_gate.parameters()))
                    d=out['coe']['diagnostics']
                    self.assertEqual([float(d[f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
                    self.assertTrue(bool(model.main_branch.x_history.inject[-1].weight.grad.abs().sum()>0))
                    for key,value in d.items():
                        if 'memory_gate_abs_mean' in key:self.assertEqual(float(value),0.)
                if method=='I22':
                    selected=supervision_mask(self.batch['x_f_gt'],self.batch['m_f'],self.batch.get('target_mask'))
                    expected=(out['x_hat_raw_consistency'][selected]-self.batch['x_f_gt'][selected]).abs().mean()
                    torch.testing.assert_close(logs['l_raw_f'],expected)

    def test_history_gain_halves_injection_same_context(self):
        model=self.helper.model(self.job('I13')).main_branch
        from stmoe_imputer.models.observed_multiscale_coe import ObservedMultiScaleCoE
        h=torch.randn(2,8,3,8,8); mask=torch.zeros(2,2,3,8,8)
        context={'h':h,'mask':mask,'x_scale':torch.ones(2,2,1,1,1),'x_pool':[torch.randn(2,2,3,8,8)]}
        delta=torch.randn_like(h); weights=torch.ones(2,1,2,3,8,8)
        with patch.object(ObservedMultiScaleCoE,'_memory',return_value=(torch.zeros_like(h),None,None)), patch.object(model.x_history,'forward',return_value=(delta,weights)):
            model.history_gain=1.; a=model._memory(context)[0]
            model.history_gain=.5; b=model._memory(context)[0]
        torch.testing.assert_close(b,a*.5,rtol=0,atol=0)

    def test_diagnostics_use_raw_and_count_empty_regions(self):
        from stmoe_imputer.engine import _CoEQualityMetrics
        j=self.job('I13'); model=self.helper.model(j); out=model(self.batch)
        compute_coe_loss(out,self.batch,j['config'])
        metrics=_CoEQualityMetrics(); metrics.update(out,self.batch); result=metrics.compute()
        selected=supervision_mask(self.batch['x_f_gt'],self.batch['m_f'],self.batch.get('target_mask'))
        expected=(out['x_hat_raw_consistency'][selected]-self.batch['x_f_gt'][selected]).abs().mean()
        self.assertAlmostEqual(result['combination_raw_all_mae'],float(expected),places=5)
        self.assertEqual(result['combination_raw_all_count'],result['combination_raw_constrained_count']+result['combination_raw_unconstrained_count'])
        batch=copy.deepcopy(self.batch);batch['m_m'].zero_();batch['m_c'].zero_()
        metrics=_CoEQualityMetrics(); metrics.update(out,batch); result=metrics.compute()
        self.assertEqual(result['combination_raw_constrained_count'],0)
        self.assertNotIn('combination_raw_constrained_mae',result)
        merged=_CoEQualityMetrics();merged.merge(metrics);self.assertEqual(merged.compute(),result)

    def test_ranking_equal_category_weight_and_id_filter(self):
        protocols=['in_distribution']+[p for group in OOD_CATEGORIES.values() for p in group]
        base={'sets':{p:{'metrics':{'mae':8.,'rmse':10.}} for p in protocols}}
        evals={'I08':base,'I13':copy.deepcopy(base),'I14':copy.deepcopy(base)}
        evals['I13']['sets']['unseen_triple']['metrics']['mae']=4.
        evals['I14']['sets']['in_distribution']['metrics']['mae']=8.87091
        rows=combination_ranking(evals)
        self.assertEqual(rows[0]['method'],'I13');self.assertAlmostEqual(rows[0]['mae_score'],5/6)
        self.assertFalse(rows[-1]['eligible'])
        del evals['I13']['sets']['unseen_geometry_repeat']
        self.assertNotIn('I13',[r['method'] for r in combination_ranking(evals)])

    def test_strict_resume_and_six_evaluations(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from run_b3_c3 import digest
        j=self.job('I13');j['config']['train'].update(epochs=2,val_epoch=1)
        j['config']['train']['scheduler']['total_epochs']=2
        run=self.root/'resumed'
        train(j,run,self.root/'r.json','cpu',stop_after=1)
        train(j,run,self.root/'r.json','cpu')
        train(j,self.root/'full',self.root/'f.json','cpu')
        a=torch.load(run/'checkpoints/last.pth',weights_only=False)
        b=torch.load(self.root/'full/checkpoints/last.pth',weights_only=False)
        for k in ('model','optimizer','scheduler','scaler','rng_states'): self.assertEqual(state_hash(a[k]),state_hash(b[k]),k)
        ev=self.root/'evaluation.json'
        evaluate_sets(run/'checkpoints/best.pth',j['config']['experiment_plan']['protocol'],j['sources']['test'],ev,digest(j['config']),'cpu')
        record=json.loads(ev.read_text());self.assertEqual(len(record['sets']),6)
        for row in record['sets'].values():
            self.assertEqual(row['metrics']['coe_expert_execution_count'],8.)
            self.assertIn('combination_raw_all_mae',row['metrics'])
            self.assertIn('l_visible_consistency_aux',row['metrics'])

if __name__=='__main__': unittest.main()
