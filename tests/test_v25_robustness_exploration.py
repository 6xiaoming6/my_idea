"""I30-I41: causal module controls, visible reconstruction safety and old parity."""
import copy
import json
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25')]
import test_v25_combination_aux as fixtures
from run_taxibj_input_communication import jobs,EXPLORATION
from stmoe_imputer.losses import compute_coe_loss
from preflight_taxibj_input_communication import check_consistency

class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CombinationAuxTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_unchanged_i25_and_protocol(self):
        parent=json.loads((ROOT/'outputs/v25/experiments/taxibj_input_communication/2c07a1e3f7f177b7/jobs/taxibj_I25_seed7.json').read_text())
        self.assertEqual(jobs(methods=['I25'])['taxibj_I25_seed7']['config'],parent['config'])
        js=jobs(methods=list(EXPLORATION));self.assertEqual(len(js),12)
        self.assertEqual(len(jobs()),8)
        for j in js.values():
            for k in ('train','data','seed'):self.assertEqual(j['config'][k],parent['config'][k])
            self.assertEqual(j['config']['experiment_plan']['protocol'],parent['config']['experiment_plan']['protocol'])
        # Normalize explicit interventions back to I25: no incidental changes.
        for j in js.values():
            c=copy.deepcopy(j['config']);ref=copy.deepcopy(parent['config']);c.pop('experiment_plan');ref.pop('experiment_plan')
            coe=c['model']['coe'];base=ref['model']['coe']
            for key in ('observed_history','scale_delta_exploration','core_validation'):coe[key]=copy.deepcopy(base[key])
            c['loss']=copy.deepcopy(ref['loss']);self.assertEqual(c,ref)

    def test_all_controls_execution_and_gradients(self):
        f=self.f
        for method in EXPLORATION:
            with self.subTest(method=method):
                j=f.job(method);m=f.helper.model(j).train();b=m.main_branch
                calls=[];hooks=[e.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0]))) for e in b.routed_experts()]
                try:out=m(f.batch)
                finally:
                    for h in hooks:h.remove()
                self.assertEqual(sum(calls),8*len(f.batch['m_f']))
                check_consistency(out,f.batch,j['config'])
                fixed=method in ('I32','I38','I39','I40')
                choices=out['coe']['triscale_choices']
                if fixed:
                    self.assertTrue(bool((choices==choices.new_tensor([2,1,0,0])).all()))
                else:self.assertTrue(bool((choices[:,3]==0).all()))
                history=method not in ('I30','I33','I38','I40')
                self.assertEqual(hasattr(b,'x_history'),history)
                if history:
                    diag=out['coe']['diagnostics']
                    self.assertEqual([float(diag[f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
                loss,_=compute_coe_loss(out,f.batch,j['config']);self.assertTrue(bool(torch.isfinite(loss)));loss.backward()
                no_delta=method in ('I31','I33','I39','I40')
                self.assertEqual(b.core_spec['communication']=='none',no_delta)
                if no_delta:
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in b.history_gate.parameters()))
                    if hasattr(b,'delta_steps'):self.assertEqual(b.delta_steps,())
                if history:self.assertGreater(float(b.x_history.inject[-1].weight.grad.abs().sum()),0)
                self.assertTrue(any(p.grad is not None for p in b.x_consistency.parameters()))

    def test_initial_prediction_parity(self):
        f=self.f;a=f.helper.model(f.job('I25')).eval();out=a(f.batch)
        for method in ('I30','I31','I33','I34','I35','I36','I37','I41'):
            b=f.helper.model(f.job(method)).eval()
            for k,v in b.state_dict().items():torch.testing.assert_close(v,a.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(b(f.batch)['x_hat_main'],out['x_hat_main'],rtol=0,atol=0)

    def test_visible_loss_exact_mask_gradient_and_empty(self):
        f=self.f;j=f.job('I36');m=f.helper.model(j).eval();out=m(f.batch)
        cfg=copy.deepcopy(j['config']);cfg['loss'].pop('lambda_visible_raw_f')
        base,_=compute_coe_loss(out,f.batch,cfg);total,logs=compute_coe_loss(out,f.batch,j['config'])
        raw=out['x_hat_raw_consistency'];visible=f.batch['m_f'].bool().expand_as(raw)
        expected=(raw[visible].float()-f.batch['x_f_obs'][visible]).abs().mean()
        torch.testing.assert_close(total-base,.1*expected)
        g=torch.autograd.grad(logs['l_visible_raw_f'] if logs['l_visible_raw_f'].requires_grad else total-base,raw,retain_graph=True)[0]
        self.assertGreater(float(g[visible].abs().sum()),0)
        # Subtracting two large loss graphs leaves float32 cancellation noise.
        torch.testing.assert_close(g[~visible],torch.zeros_like(g[~visible]),atol=1e-8,rtol=0)
        # Hidden F placeholders and coarse labels cannot enter the new term.
        dirty={k:v.clone() if torch.is_tensor(v) else v for k,v in f.batch.items()}
        dirty['x_f_obs'][~visible]=float('nan');dirty['x_m_gt']=torch.full_like(dirty['x_m_obs'],float('nan'))
        dirty['x_c_gt']=torch.full_like(dirty['x_c_obs'],float('nan'))
        _,other=compute_coe_loss(out,dirty,j['config']);torch.testing.assert_close(logs['l_visible_raw_f'],other['l_visible_raw_f'])
        empty={**f.batch,'m_f':torch.zeros_like(f.batch['m_f'])}
        _,el=compute_coe_loss(out,empty,j['config']);self.assertEqual(float(el['l_visible_raw_f']),0)
        bad=dict(out);bad.pop('x_hat_raw_consistency')
        with self.assertRaises(ValueError):compute_coe_loss(bad,f.batch,j['config'])
        for weight in (-1,float('nan')):
            badcfg=copy.deepcopy(j['config']);badcfg['loss']['lambda_visible_raw_f']=weight
            with self.assertRaises(ValueError):compute_coe_loss(out,f.batch,badcfg)

    def test_zero_new_loss_keeps_old_exact(self):
        f=self.f;j=f.job('I25');out=f.helper.model(j)(f.batch)
        a,la=compute_coe_loss(out,f.batch,j['config']);cfg=copy.deepcopy(j['config']);cfg['loss']['lambda_visible_raw_f']=0
        b,lb=compute_coe_loss(out,f.batch,cfg)
        torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertEqual(set(la),set(lb))

    def test_fixed_mix_and_checkpoint_replay(self):
        f=self.f
        for method,beta in [('I34',.5),('I35',.8),('I37',.5)]:
            j=f.job(method);m=f.helper.model(j);out=m(f.batch)
            self.assertFalse(hasattr(m.main_branch,'x_loss_mix_logit'))
            _,logs=compute_coe_loss(out,f.batch,j['config'])
            torch.testing.assert_close(logs['l_raw_f_weighted'],beta*logs['l_raw_f'])
            torch.testing.assert_close(logs['l_final_weighted'],(1-beta)*logs['l_final'])
        original=f.job;f.job=lambda method:original('I37' if method=='I13' else method)
        f.test_strict_resume_and_six_evaluations()

if __name__=='__main__':unittest.main()
