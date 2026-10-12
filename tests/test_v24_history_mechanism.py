import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_history_mechanism import jobs,PARENT
from run_b3_c3 import load
from run_history_plane_interaction import prepare_initialization
from run_history_followup import reserve_run
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.jobs=jobs()

    def cfg(self,label):
        c=copy.deepcopy(self.jobs[f'taxibj_{label}_seed7']['config'])
        c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
        return c

    def model(self,c):
        torch.manual_seed(7);return DualBranchSTImputer.from_config(c)

    def batch(self):
        g=torch.Generator().manual_seed(19);x=torch.randn(2,2,3,8,8,generator=g)*10
        mask=(torch.rand(x.shape,generator=g)>.4).float()
        return dict(x_f_obs=x*mask,x_f_gt=x,m_f=mask,mask_family=torch.tensor([0,1]))

    def test_frozen_protocol_and_initialization_zero_equivalence(self):
        self.assertEqual(len(self.jobs),8)
        c=self.cfg('Z08');c['model']['coe'].pop('history_mechanism')
        y=self.model(c);rng=torch.get_rng_state();b=self.batch();expected=y(b)['x_hat_main']
        for label,job in self.jobs.items():
            c=self.cfg(job['method']);m=self.model(c)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),label)
            self.assertEqual(state_hash(y.state_dict()),state_hash(m.state_dict()),label)
            torch.testing.assert_close(m(b)['x_hat_main'],expected,rtol=0,atol=0)
            rate=.6 if job['method'] in ('Z07','Z08') else .4
            for value in [c['data']['mask']['missing_rate'],c['data']['train_mask_diversity']['rates'][0],c['data']['eval_mask_diversity']['rates'][0],c['experiment_plan']['protocol']['rate']]:self.assertEqual(value,rate)
            self.assertEqual(c['train']['epochs'],100);self.assertEqual(c['data']['batch_size'],32)
            self.assertEqual(c['loss']['lambda_coe_mid'],0);self.assertEqual(len(c['experiment_plan']['protocol']['evaluations']),6)

    def test_active_y12_compatibility_and_legacy_feature_hook(self):
        c=self.cfg('Z08');z=self.model(c);c['model']['coe'].pop('history_mechanism');y=self.model(c)
        with torch.no_grad():
            y.main_branch.x_history.inject[-1].weight.fill_(.015)
            y.main_branch.x_interaction[-1].weight.fill_(.02)
            y.main_branch.history_gate[-1].bias.fill_(.02)
        z.load_state_dict(y.state_dict());b=self.batch()
        a=y(b);zz=z(b)
        torch.testing.assert_close(a['x_hat_main'],zz['x_hat_main'],rtol=0,atol=0)
        compute_coe_loss(a,b,c)[0].backward();compute_coe_loss(zz,b,c)[0].backward()
        for (k,p),(kk,q) in zip(y.named_parameters(),z.named_parameters()):
            self.assertEqual(k,kk)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        # The new dispatch hook leaves old concat and product forwards unchanged.
        from types import MethodType
        for label in ['X09','X10']:
            old=load(ROOT/'outputs/v24-COE/experiments/history_plane_interaction/86284122bd9a18c2/plan.json')['jobs'][f'taxibj_{label}_seed7']['config']
            old=copy.deepcopy(old);old['model']['main']['dim']=8;old['model']['coe']['router_hidden_dim']=8
            m=self.model(old)
            with torch.no_grad():m.main_branch.x_interaction[-1].weight.fill_(.03)
            expected=m(b)['x_hat_main']
            m.main_branch._interaction_features=MethodType(lambda self,a,b:torch.cat((a,b),1) if self.x_spec['interaction']=='concat' else torch.cat((a*b,a-b),1),m.main_branch)
            torch.testing.assert_close(expected,m(b)['x_hat_main'],rtol=0,atol=0)

    def test_history_controls_and_innovation_are_exact(self):
        mask=torch.ones(2,2,3,8,8);scale=torch.ones(2,2,1,1,1)
        pool=[torch.ones_like(mask)*v for v in [2.,5.,11.]]
        for label,expected in [('Z04',6.),('Z05',11.),('Z06',-5.)]:
            m=self.model(self.cfg(label)).main_branch
            v,w=m.x_history.read(pool,mask,scale)
            torch.testing.assert_close(v,torch.full_like(v,expected),rtol=1e-6,atol=1e-6)
            if label in ('Z04','Z05'):self.assertFalse(any(p.requires_grad for p in m.x_history.score.parameters()))
            if label=='Z06':
                v,_=m.x_history.read(pool[:1],mask,scale);self.assertEqual(float(v.abs().sum()),0)
        for label in ['Z02','Z03']:
            m=self.model(self.cfg(label)).main_branch;a=torch.randn(2,8,3,8,8);b=torch.randn_like(a)
            f=m._interaction_features(a,b)
            kept=a*b if label=='Z02' else a-b
            torch.testing.assert_close(f[:,:8] if label=='Z02' else f[:,8:],kept)
            self.assertEqual(float((f[:,8:] if label=='Z02' else f[:,:8]).abs().sum()),0)
            with torch.no_grad():m.x_interaction[-1].weight.fill_(.1)
            m.x_interaction(f).square().mean().backward()
            grad=m.x_interaction[0].weight.grad
            self.assertEqual(float((grad[:,8:] if label=='Z02' else grad[:,:8]).abs().sum()),0)

    def test_native_top2_no_gt_leak_finite_gradients_and_no_cross_batch_pool(self):
        b=self.batch()
        for label in ['Z01','Z02','Z03','Z04','Z05','Z06','Z07','Z08']:
            c=self.cfg(label);m=self.model(c);main=m.main_branch
            calls=[];hooks=[e.register_forward_pre_hook(lambda m,a:calls.append(len(a[0]))) for e in main.routed_experts(0)]
            out=m(b)
            for h in hooks:h.remove()
            self.assertEqual(sum(calls),16)
            torch.testing.assert_close(out['coe']['triscale_choices'],torch.tensor([[2,1,0,0]]*2))
            compute_coe_loss(out,b,c)[0].backward()
            for block in [main.x_history,main.x_interaction,main.history_gate]:
                grads=[p.grad for p in block.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads),label)
            torch.testing.assert_close(m(b)['x_hat_main'],m(dict(b,x_f_gt=b['x_f_gt']*99))['x_hat_main'],rtol=0,atol=0)
            for i in range(1,5):self.assertEqual(float(out['coe']['diagnostics'][f'x_step{i}_pool_size']),i-1)
            for value in [0.,1.]:
                mask=torch.full_like(b['m_f'],value)
                self.assertTrue(torch.isfinite(m(dict(b,m_f=mask,x_f_obs=b['x_f_gt']*mask))['x_hat_main']).all())
            torch.testing.assert_close(m(b)['x_hat_main'],out['x_hat_main'],rtol=0,atol=0)

    def test_resume_completed_skip_and_mixed_rate_evaluation(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        import run_four_direction_exploration as base
        from report_history_followup import collect
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);suite=root/'suite';js={}
            for label in ['Z06','Z07']:
                job=copy.deepcopy(self.jobs[f'taxibj_{label}_seed7']);c=self.cfg(label);job['config']=c
                c['output_dir']=str(root/'outputs/v24-COE');c['train'].update(epochs=2,val_epoch=1)
                c['train']['scheduler']['total_epochs']=2;c['data'].update(batch_size=2,pin_memory=False)
                c['experiment_plan']['protocol']={'rate':c['data']['mask']['missing_rate'],'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
                for split in ['train','val','test']:
                    p=root/(split+'.npz');np.savez(p,x_f_gt=np.random.RandomState(2).randn(4,2,3,8,8).astype('float32'));job['sources'][split]=str(p)
                js[job['variant']]=job
            c=self.cfg('Z08')
            for key in ['history_mechanism','history_followup','history_plane_interaction']:c['model']['coe'].pop(key)
            m=self.model(c);ref=root/'init.pth';torch.save({'model':m.state_dict(),'sha256':state_hash(m.state_dict())},ref)
            prepare_initialization(suite,js,ref)
            for name,job in js.items():
                run=reserve_run(suite,job);result=suite/'results'/f'{name}.json'
                train(job,run,result,'cpu',stop_after=1);train(job,run,result,'cpu')
                with patch.object(base,'subprocess_run',side_effect=AssertionError('finished must skip')):base.launch(suite,job,0)
                other=root/(name+'_full');train(job,other,root/(name+'.json'),'cpu')
                aa=torch.load(run/'checkpoints/last.pth',weights_only=False);bb=torch.load(other/'checkpoints/last.pth',weights_only=False)
                for key in ['model','optimizer','scheduler','scaler','rng_states']:self.assertEqual(state_hash(aa[key]),state_hash(bb[key]),key)
                e=evaluate_sets(run/'checkpoints/best.pth',job['config']['experiment_plan']['protocol'],job['sources']['test'],suite/'evaluations'/f'{name}.json',device='cpu')
                self.assertEqual(e['sets']['in_distribution']['mask_spec']['rates'],[job['config']['data']['mask']['missing_rate']])
                self.assertEqual(collect(suite,job['method'],job)[0]['status'],'finished')

    def test_stage_report_includes_all_methods_and_limits(self):
        from report_history_mechanism import export_report
        parent=load(ROOT/PARENT/'plan.json')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);suite=root/'suite';suite.mkdir()
            refs=dict(parent['references'])
            refs.update({label:parent['jobs'][f'taxibj_{label}_seed7'] for label in ['Y08','Y12']})
            payload=dict(jobs=self.jobs,references=refs)
            p=export_report(root,suite,payload);s=p.read_text()
            self.assertIn('阶段报告',s);self.assertIn('Z08',s);self.assertIn('2048',s)
            self.assertLess(s.index('实验具体做法'),s.index('全部结果'))
            self.assertEqual(len(load(suite/'summary.json')),len(refs)+8)

if __name__=='__main__':unittest.main()
