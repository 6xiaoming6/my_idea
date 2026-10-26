"""Semantic tests for I37's observation/estimate input separation."""
import copy
import sys
import unittest
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests'), str(ROOT/'scripts/v25'), str(ROOT/'scripts/v24')]
import test_v25_combination_aux as fixtures
from input_refinement_config import make_job, CANDIDATES, protocols, select_candidate
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.observed_consistency_coe import _blocks
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash
from preflight_taxibj_input_communication import check_consistency


class InputRefinementTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CombinationAuxTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def job(self, method):
        j=self.f.job('I37')
        spec=make_job(method)['config']['model']['coe'].get('input_refinement')
        if spec:j['config']['model']['coe']['input_refinement']=spec
        return j

    def context(self, model):
        branch=model.main_branch;b=self.f.batch
        branch._observations=[(b['x_'+s+'_obs']/f**2,b['m_'+s]) for s,f in zip('fmc',[1,2,4])]
        try:return branch._initialize(b['x_f_obs'],b['m_f'])
        finally:del branch._observations

    def test_default_and_neutral_parity(self):
        j=self.f.job('I37');a=self.f.helper.model(j).train()
        off=copy.deepcopy(j);off['config']['model']['coe']['input_refinement']={'enabled':False}
        neutral=copy.deepcopy(j);neutral['config']['model']['coe']['input_refinement']={'enabled':True}
        for cfg in [off,neutral]:
            b=self.f.helper.model(cfg).train()
            self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
            a.zero_grad(set_to_none=True);oa=a(self.f.batch);ob=b(self.f.batch)
            torch.testing.assert_close(oa['x_hat_raw_consistency'],ob['x_hat_raw_consistency'],atol=0,rtol=0)
            torch.testing.assert_close(oa['x_hat_main'],ob['x_hat_main'],atol=0,rtol=0)
            la=compute_coe_loss(oa,self.f.batch,j['config'])[0];lb=compute_coe_loss(ob,self.f.batch,cfg['config'])[0]
            torch.testing.assert_close(la,lb,atol=0,rtol=0);la.backward();lb.backward()
            self.assertEqual(state_hash({n:p.grad for n,p in a.named_parameters()}),state_hash({n:p.grad for n,p in b.named_parameters()}))

    def test_frozen_definitions_and_three_seed_budget(self):
        self.assertEqual(CANDIDATES,[f'I{i}' for i in range(54,61)])
        self.assertEqual([make_job(n)['config']['seed'] for n in CANDIDATES], [7]*7)
        self.assertEqual(make_job('I61')['config']['seed'],27)
        self.assertEqual(make_job('I62',selected='I54')['config']['seed'],17)
        self.assertEqual(make_job('I63',selected='I54')['config']['seed'],27)
        for n in CANDIDATES:
            j=make_job(n);ref=make_job('I61')
            self.assertEqual(j['config']['loss'],ref['config']['loss'])
            self.assertEqual(j['config']['train'],ref['config']['train'])
            for name in ['observed_history','observation_consistency','core_validation','scale_delta_exploration']:
                self.assertEqual(j['config']['model']['coe'][name],ref['config']['model']['coe'][name])
        _,dev,confirm=protocols()
        self.assertEqual([x['seed'] for x in dev['evaluations'].values()],list(range(2026102201,2026102207)))
        self.assertEqual([x['seed'] for x in confirm['evaluations'].values()],list(range(2026102211,2026102217)))
        def record(value,split='val'):
            return dict(split=split,sets={k:dict(metrics=dict(mae=value,rmse=value*2)) for k in dev['evaluations']})
        with self.assertRaises(ValueError):select_candidate(record(10),{'I54':record(9,'test')},{'I54':1})

    def test_initialization_h0_and_all_variant_contracts(self):
        reference=self.f.helper.model(self.f.job('I37')).eval()
        expected=reference(self.f.batch)['coe']['initial_prediction']
        for n in CANDIDATES:
            with self.subTest(method=n):
                j=self.job(n);m=self.f.helper.model(j).train()
                self.assertEqual(state_hash(m.state_dict()),state_hash(reference.state_dict()))
                calls=[];hooks=[e.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0]))) for e in m.main_branch.routed_experts()]
                try:out=m(self.f.batch)
                finally:
                    for h in hooks:h.remove()
                self.assertEqual(sum(calls),8*len(self.f.batch['m_f']))
                self.assertTrue(bool((out['coe']['triscale_choices'][:,3]==0).all()))
                torch.testing.assert_close(expected,out['coe']['initial_prediction'],atol=0,rtol=0)
                check_consistency(out,self.f.batch,j['config'])
                loss,_=compute_coe_loss(out,self.f.batch,j['config']);loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                self.assertEqual([float(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
                if n in ['I54','I56','I57','I58','I60']:
                    self.assertTrue(all(float(out['coe']['diagnostics'][f'input_step{i}_estimate_abs'])==0 for i in range(1,5)))
                poisoned=copy.deepcopy(self.f.batch)
                for s in 'fmc':poisoned['x_'+s+'_obs'][~poisoned['m_'+s].bool()]=float('nan')
                poisoned['x_f_gt'].fill_(999999)
                with torch.no_grad():
                    a=m.eval()(self.f.batch);b=m(poisoned)
                    torch.testing.assert_close(a['x_hat_main'],b['x_hat_main'],atol=0,rtol=0)

    def test_router_statistics_exclude_missing_predictions(self):
        model=self.f.helper.model(self.job('I55'));b=model.main_branch;c=self.context(model)
        value=c['completion'].clone();dirty=value.clone();dirty[~c['mask'].bool()]=999999
        args=(c['h'],value,c['ss'],torch.zeros_like(value),1-c['mask'],None)
        x=b._router_features(*args)
        y=b._router_features(c['h'],dirty,c['ss'],torch.zeros_like(value),1-c['mask'],None)
        torch.testing.assert_close(x,y,atol=0,rtol=0)
        expected=c['x'].sum((2,3,4))/c['mask'].sum((2,3,4)).clamp_min(1)
        torch.testing.assert_close(x[:,2*b.dim:2*b.dim+b.c_in],expected)
        self.assertEqual(float(x[:,2*b.dim+b.c_in:2*b.dim+2*b.c_in].abs().sum()),0.)

    def test_coarse_budget_is_static_and_metadata_is_explicit(self):
        model=self.f.helper.model(self.job('I60'));b=model.main_branch;c=self.context(model);ids=torch.arange(len(c['h']))
        ledger=c['input_regions']
        for sid,factor in [(1,2),(2,4)]:
            _,value,mask,support,pos,fill=b._native_inputs(c,c['h'],sid,ids)
            self.assertEqual(float(support.abs().sum()+pos.abs().sum()+fill.abs().sum()),0.)
            target_sum=_blocks(self.f.batch['x_f_gt']*ledger.owners[sid],factor).sum(-1)
            torch.testing.assert_close(value*ledger.counts[sid],target_sum,atol=1e-4,rtol=0)
            changed=dict(c,last_prediction=torch.randn_like(c['last_prediction'])*1e6,completion=torch.randn_like(c['completion'])*1e6)
            other=b._native_inputs(changed,c['h'],sid,ids)[1]
            torch.testing.assert_close(value,other,atol=0,rtol=0)
        original=copy.deepcopy(self.f.batch)
        meanjob=self.job('I58');meanjob['config']['data']['scales']['pooling_mode']='mean'
        meanmodel=self.f.helper.model(meanjob).eval();sum_model=self.f.helper.model(self.job('I58')).eval()
        for s,factor in [('m',2),('c',4)]:original['x_'+s+'_obs']/=factor**2
        torch.testing.assert_close(meanmodel(original)['x_hat_main'],sum_model(self.f.batch)['x_hat_main'],atol=1e-5,rtol=1e-5)

    def test_feedback_bound_and_detached_current_source(self):
        m=self.f.helper.model(self.job('I59'));b=m.main_branch;c=self.context(m);ids=torch.arange(len(c['h']))
        current=torch.full_like(c['last_prediction'],10000.,requires_grad=True)
        c['last_prediction']=current
        for sid in range(3):
            _,values,mask,_,_,fill=b._native_inputs(c,c['h'],sid,ids)
            self.assertFalse(fill.requires_grad)
            self.assertTrue(bool((fill.abs() <= .1*c['input_sigma']+1e-6).all()))
        edge=copy.deepcopy(self.f.batch)
        for s in 'fmc':edge['m_'+s].zero_()
        with torch.no_grad():self.assertTrue(bool(torch.isfinite(m.eval()(edge)['x_hat_main']).all()))


if __name__=='__main__':unittest.main()
