"""Final input controls: preserve evidence, frozen budget and original backbone."""
import copy
import unittest
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
import test_v25_combination_aux as fixtures
from final_input_config import CANDIDATES, make_job, protocols, choose
from input_refinement_config import make_job as previous_job
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash
from preflight_taxibj_input_communication import check_consistency


class FinalInputTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CombinationAuxTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)

    def job(self, method):
        j=self.f.job('I37')
        j['config']['model']['coe']['input_refinement']=copy.deepcopy(make_job(method)['config']['model']['coe']['input_refinement'])
        return j

    def test_budget_and_frozen_training(self):
        self.assertEqual(CANDIDATES,[f'I{i}' for i in range(64,76)])
        ref=previous_job('I61')['config']
        for n in CANDIDATES:
            cfg=make_job(n)['config']; self.assertEqual(cfg['seed'],7)
            for key in ['loss','train']: self.assertEqual(cfg[key],ref[key])
            for key in ['core_validation','observed_history','observation_consistency','scale_delta_exploration']:
                self.assertEqual(cfg['model']['coe'][key],ref['model']['coe'][key])
        _,dev,confirm=protocols()
        self.assertEqual([s['seed'] for s in dev['evaluations'].values()],list(range(2026102301,2026102307)))
        self.assertEqual([s['seed'] for s in confirm['evaluations'].values()],list(range(2026102311,2026102317)))
        def record(v,split='val'):
            return dict(split=split,sets={n:dict(metrics=dict(mae=v,rmse=2*v)) for n in dev['evaluations']})
        selected=choose(record(10),{'I64':record(11)},{'I64':1})
        self.assertIsNone(selected['selected']); self.assertFalse(selected['qualified_by_development_id'])
        with self.assertRaises(ValueError): choose(record(10),{'I64':record(9,'test')},{'I64':1})

    def test_all_initialization_h0_native_calls_loss_and_no_hidden_leak(self):
        ref=self.f.helper.model(self.f.job('I37')).eval()
        with torch.no_grad(): initial=ref(self.f.batch)['coe']['initial_prediction']
        for n in CANDIDATES:
            with self.subTest(method=n):
                j=self.job(n); m=self.f.helper.model(j).train()
                self.assertEqual(state_hash(m.state_dict()),state_hash(ref.state_dict()))
                calls=[]; hooks=[e.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0]))) for e in m.main_branch.routed_experts()]
                try: out=m(self.f.batch)
                finally:
                    for hook in hooks: hook.remove()
                self.assertEqual(sum(calls),8*len(self.f.batch['m_f']))
                self.assertTrue(bool((out['coe']['triscale_choices'][:,3]==0).all()))
                torch.testing.assert_close(initial,out['coe']['initial_prediction'],atol=0,rtol=0)
                check_consistency(out,self.f.batch,j['config'])
                loss,_=compute_coe_loss(out,self.f.batch,j['config']); loss.backward()
                self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                self.assertEqual([float(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
                d=out['coe']['diagnostics']
                spec=j['config']['model']['coe']['input_refinement']
                for step in range(1,5):
                    if spec['support_gain']==0: self.assertEqual(float(d[f'input_step{step}_support_rms']),0.)
                    if spec['position_gain']==0: self.assertEqual(float(d[f'input_step{step}_position_rms']),0.)
                    if spec['expert_fill']=='visible_initial' and spec['prediction_gain']==0:
                        self.assertEqual(float(d[f'input_step{step}_fallback_prediction_abs']),0.)
                dirty=copy.deepcopy(self.f.batch)
                for s in 'fmc': dirty['x_'+s+'_obs'][~dirty['m_'+s].bool()]=float('nan')
                dirty['x_f_gt'].fill_(999999)
                with torch.no_grad():
                    torch.testing.assert_close(m.eval()(self.f.batch)['x_hat_main'],m(dirty)['x_hat_main'],atol=0,rtol=0)

    def test_visible_f_preserving_fallback_hand_calculation_and_gradient(self):
        # A 2x2 fine region: visible [10,40], missing P0 [8,12].
        # The mean fallback is 12.5 + gain*5, independent of native mask.
        for gain, expected in [(0.,12.5),(.25,13.75),(.5,15.),(1.,17.5)]:
            b=self.f.helper.model(self.job('I71')).main_branch
            b.input_refinement['prediction_gain']=gain
            fill=torch.tensor([10.,8.,12.,40.]).reshape(1,1,1,2,2).requires_grad_()
            fm=torch.tensor([1.,0.,0.,1.]).reshape_as(fill)
            h=torch.zeros(1,b.dim,1,2,2)
            coarse=torch.full((1,1,1,1,1),99.)
            mask=torch.zeros_like(coarse); support=mask.expand(1,2,1,1,1).clone(); pos=mask.expand(1,3,1,1,1).clone()
            c=dict(completion=fill,mask=fm,observed_pyramid=[None,(coarse,mask,support,pos)])
            _,values,nm,_,_,fallback=b._native_inputs(c,h,1,torch.tensor([0]))
            self.assertEqual(float(fallback),expected); self.assertEqual(float(nm.sum()),0.)
            fallback.sum().backward()
            torch.testing.assert_close(fill.grad.flatten(),torch.tensor([.25,gain/4,gain/4,.25]))
            c['observed_pyramid'][1]=(coarse,torch.ones_like(mask),support,pos)
            values=b._native_inputs(c,h,1,torch.tensor([0]))[1]
            self.assertEqual(float(values),99.)

    def test_gain_one_exact_parity_and_metadata_scaling(self):
        initial=self.job('I70'); equivalent=copy.deepcopy(initial)
        equivalent['config']['model']['coe']['input_refinement']['expert_fill']='visible_initial'
        a=self.f.helper.model(initial); b=self.f.helper.model(equivalent)
        oa=a(self.f.batch); ob=b(self.f.batch)
        torch.testing.assert_close(oa['x_hat_main'],ob['x_hat_main'],atol=0,rtol=0)
        compute_coe_loss(oa,self.f.batch,initial['config'])[0].backward()
        compute_coe_loss(ob,self.f.batch,equivalent['config'])[0].backward()
        self.assertEqual(state_hash({n:p.grad for n,p in a.named_parameters()}),state_hash({n:p.grad for n,p in b.named_parameters()}))
        br=self.f.helper.model(self.job('I74')).main_branch; batch=self.f.batch
        br._observations=[(batch['x_'+s+'_obs']/f**2,batch['m_'+s]) for s,f in zip('fmc',[1,2,4])]
        try: c=br._initialize(batch['x_f_obs'],batch['m_f'])
        finally: del br._observations
        for sid in range(3):
            native=br._native_inputs(c,c['h'],sid,torch.arange(len(c['h'])))
            torch.testing.assert_close(native[3],.25*c['observed_pyramid'][sid][2],atol=0,rtol=0)
            torch.testing.assert_close(native[4],.25*c['observed_pyramid'][sid][3],atol=0,rtol=0)

    def test_sum_mean_and_no_observation_edges(self):
        for n in ['I71','I75']:
            j=self.job(n); mean=copy.deepcopy(j); mean['config']['data']['scales']['pooling_mode']='mean'
            batch=copy.deepcopy(self.f.batch)
            for s,f in [('m',2),('c',4)]: batch['x_'+s+'_obs']/=f*f
            with torch.no_grad():
                a=self.f.helper.model(j).eval()(self.f.batch)['x_hat_main']
                b=self.f.helper.model(mean).eval()(batch)['x_hat_main']
                torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
                model=self.f.helper.model(j).eval()
                for visible in [0.,1.]:
                    edge=copy.deepcopy(self.f.batch)
                    for s in 'fmc':
                        edge['m_'+s].fill_(visible)
                        if not visible: edge['x_'+s+'_obs'].fill_(float('nan'))
                        else:
                            from stmoe_imputer.utils.deterministic import pool
                            factor={'f':1,'m':2,'c':4}[s]
                            edge['x_'+s+'_obs']=pool(edge['x_f_gt'],factor)*factor**2
                    out=model(edge)
                    self.assertTrue(bool(torch.isfinite(out['x_hat_main']).all()))

if __name__ == '__main__': unittest.main()
