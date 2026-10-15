"""I25 convex learned F supervision: parity, gradients, optimizer and replay."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25')]
import test_v25_combination_aux as fixtures
from run_taxibj_input_communication import jobs
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.utils.deterministic import state_hash
import run_interleaved as interleave

class LearnableMixTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CombinationAuxTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_initial_parity_and_exact_scalar_gradient(self):
        f=self.f;a=f.helper.model(f.job('I24')).eval();rng=torch.get_rng_state().clone()
        j=f.job('I25');b=f.helper.model(j).eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        shared={k:v for k,v in b.state_dict().items() if k!='main_branch.x_loss_mix_logit'}
        self.assertEqual(state_hash(a.state_dict()),state_hash(shared))
        oa,ob=a(f.batch),b(f.batch)
        torch.testing.assert_close(oa['x_hat_main'],ob['x_hat_main'],rtol=0,atol=0)
        la,_=compute_coe_loss(oa,f.batch,f.job('I24')['config'])
        lb,logs=compute_coe_loss(ob,f.batch,j['config'])
        torch.testing.assert_close(la,lb)
        beta=ob['f_loss_raw_weight'];self.assertAlmostEqual(float(beta),.3,places=6)
        self.assertEqual(float(logs['f_loss_raw_weight']+logs['f_loss_final_weight']),1.)
        p=b.main_branch.x_loss_mix_logit
        grad=torch.autograd.grad(lb,p,retain_graph=True)[0]
        torch.testing.assert_close(grad,beta*(1-beta)*(logs['l_raw_f']-logs['l_final']))
        opt=build_optimizer(b,j['config']);groups=[g for g in opt.param_groups if any(x is p for x in g['params'])]
        self.assertEqual(len(groups),1);self.assertEqual(groups[0]['weight_decay'],0)
        self.assertEqual(groups[0]['lr'],j['config']['train']['lr_main'])
        before=p.detach().clone();lb.backward();opt.step()
        self.assertNotEqual(float(before),float(p));self.assertTrue(bool(torch.isfinite(p)))

    def test_guards_and_old_configs(self):
        frozen=json.loads((ROOT/'outputs/v25/experiments/taxibj_input_communication/37ae81a02eea687f/plan.json').read_text())['jobs']
        self.assertEqual(jobs(methods=['I23','I24']),frozen)
        f=self.f;j=f.job('I25');model=f.helper.model(j);out=model(f.batch)
        for key in ('f_loss_raw_weight','x_hat_raw_consistency'):
            bad=dict(out);bad.pop(key)
            with self.assertRaisesRegex(ValueError,'requires'):compute_coe_loss(bad,f.batch,j['config'])
        cfg=copy.deepcopy(j['config']);cfg['loss']['lambda_raw_f']=.3
        with self.assertRaisesRegex(ValueError,'excludes fixed'):compute_coe_loss(out,f.batch,cfg)
        for initial in (0,1,-1,float('nan')):
            bad=copy.deepcopy(j);bad['config']['loss']['learnable_f_mix']['initial_raw_weight']=initial
            with self.assertRaises(ValueError):f.helper.model(bad)
        for value in (-100.,100.):
            with torch.no_grad():model.main_branch.x_loss_mix_logit.fill_(value)
            _,logs=compute_coe_loss(model(f.batch),f.batch,j['config'])
            self.assertEqual(float(logs['f_loss_raw_weight']+logs['f_loss_final_weight']),1.)

    def test_checkpoint_replay_and_six_evaluations(self):
        f=self.f;original=f.job;f.job=lambda method:original('I25' if method=='I13' else method)
        f.test_strict_resume_and_six_evaluations()
        cp=torch.load(f.root/'resumed/checkpoints/last.pth',weights_only=False)
        self.assertIn('main_branch.x_loss_mix_logit',cp['model'])

    def test_handoff_rejects_live_controller_or_unpaused_worker(self):
        old=self.f.root/'old';old.mkdir()
        parent={'pid':101};worker={'pid':102}
        interleave.write(old/'paused.json',{'status':'paused_by_user'})
        interleave.write(old/'state.json',{'phase':'paused_by_user'})
        interleave.write(old/'interleave_lease.json',{'parent':parent,'worker':worker,'controller':{'pid':103}})
        plan={'adopt_suspended_from':{'path':str(old),'pause_sha256':interleave.file_sha(old/'paused.json'),
              'lease_sha256':interleave.file_sha(old/'interleave_lease.json')},'interleave_after':{}}
        with patch.object(interleave,'same_process',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'controller is still alive'):interleave.adopt_suspended(plan,parent,worker)
        with patch.object(interleave,'same_process',side_effect=lambda p:p['pid']!=103),patch.object(interleave,'process',return_value={'state':'S'}):
            with self.assertRaisesRegex(RuntimeError,'remain suspended'):interleave.adopt_suspended(plan,parent,worker)
        with patch.object(interleave,'same_process',side_effect=lambda p:p['pid']!=103),patch.object(interleave,'process',return_value={'state':'T'}),patch.object(interleave,'locks_owned'),patch.object(interleave,'boundary_status',return_value=True),patch.object(interleave,'children',return_value=[]):
            interleave.adopt_suspended(plan,parent,worker)

if __name__=='__main__':unittest.main()
