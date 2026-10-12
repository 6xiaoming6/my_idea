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
from run_delta_dependence import jobs,reference_jobs,audit_parent_initialization
from run_history_plane_interaction import prepare_initialization
from run_history_followup import reserve_run
from run_b3_c3 import load
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.deterministic import state_hash


class DeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.jobs=jobs();cls.refs,cls.locations=reference_jobs()

    def cfg(self,label):
        c=copy.deepcopy(self.jobs[f'taxibj_{label}_seed7']['config'])
        c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
        return c

    def model(self,c):
        torch.manual_seed(7);return DualBranchSTImputer.from_config(c)

    def batch(self):
        g=torch.Generator().manual_seed(19);x=torch.randn(2,2,3,8,8,generator=g)*10
        mask=(torch.rand(x.shape,generator=g)>.4).float()
        return dict(x_f_obs=x*mask,x_f_gt=x,m_f=mask)

    def test_full_initialization_and_only_config_change(self):
        for label in ['Z09','Z10','Z11','Z12']:
            job=self.jobs[f'taxibj_{label}_seed7'];c=copy.deepcopy(job['config'])
            parent=c['model']['coe'].pop('delta_dependence')['parent']
            c['model']['coe']['core_validation']['communication']='conditional'
            c.pop('experiment_plan');expected=copy.deepcopy(self.refs[parent]['config']);expected.pop('experiment_plan')
            self.assertEqual(c,expected)
            model=self.model(job['config']);rng=torch.get_rng_state()
            initial=Path(self.locations[parent])/'initialization'/f'taxibj_{parent}_seed7.pth'
            old=torch.load(initial,map_location='cpu',weights_only=False)
            self.assertEqual(state_hash(model.state_dict()),state_hash(old['model']))
            parent_model=self.model(self.refs[parent]['config'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            gate_count=sum(p.numel() for p in parent_model.main_branch.history_gate.parameters())
            self.assertEqual(sum(p.numel() for p in parent_model.parameters() if p.requires_grad)-
                             sum(p.numel() for p in model.parameters() if p.requires_grad),gate_count)

    def test_disabled_branch_sparse_dispatch_and_active_gradients(self):
        for label in ['Z09','Z10','Z11','Z12']:
            m=self.model(self.cfg(label));main=m.main_branch;b=self.batch()
            with torch.no_grad():
                main.x_interaction[-1].weight.fill_(.01)
                if hasattr(main,'x_history'):main.x_history.inject[-1].weight.fill_(.01)
            calls=[];hooks=[e.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0]))) for e in main.routed_experts(0)]
            with patch.object(main,'rms_match',side_effect=AssertionError('difference RMS executed')), \
                 patch.object(main.history_gate,'forward',side_effect=AssertionError('difference gate executed')):
                out=m(b);out['x_hat_main'].square().mean().backward()
            for hook in hooks:hook.remove()
            self.assertEqual(sum(calls),8*len(b['m_f']))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in main.history_gate.parameters()))
            for name in ['x_interaction']+(['x_history'] if hasattr(main,'x_history') else []):
                params=list(getattr(main,name).parameters())
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in params))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in params))
            changed=dict(b,x_f_gt=b['x_f_gt']+1000)
            torch.testing.assert_close(out['x_hat_main'],m(changed)['x_hat_main'],rtol=0,atol=0)
            for mask in [torch.zeros_like(b['m_f']),torch.ones_like(b['m_f'])]:
                self.assertTrue(torch.isfinite(m(dict(b,m_f=mask,x_f_obs=b['x_f_gt']*mask))['x_hat_main']).all())
            torch.testing.assert_close(out['x_hat_main'],m(b)['x_hat_main'],rtol=0,atol=0)

    def test_parent_zero_gate_forward_backward_equivalence(self):
        for label in ['Z09','Z10','Z11','Z12']:
            c=self.cfg(label);parent=copy.deepcopy(c);parent['model']['coe'].pop('delta_dependence')
            parent['model']['coe']['core_validation']['communication']='conditional'
            left=self.model(c);right=self.model(parent)
            with torch.no_grad():
                for m in [left,right]:
                    m.main_branch.x_interaction[-1].weight.fill_(.01)
                    if hasattr(m.main_branch,'x_history'):m.main_branch.x_history.inject[-1].weight.fill_(.01)
            a=left(self.batch())['x_hat_main'];b=right(self.batch())['x_hat_main']
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            a.square().mean().backward();b.square().mean().backward()
            for (n,p),(nn,q) in zip(left.named_parameters(),right.named_parameters()):
                self.assertEqual(n,nn)
                if 'history_gate' not in n:
                    if p.grad is None:self.assertIsNone(q.grad)
                    else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_relative_history_mask_and_old_guards(self):
        m=self.model(self.cfg('Z10'));b=self.batch();inputs=[]
        hook=m.main_branch.x_history.inject.register_forward_pre_hook(lambda mod,a:inputs.append(a[0].detach().clone()))
        m(b);hook.remove()
        self.assertEqual(len(inputs),3);self.assertTrue((inputs[0][:,:2]==0).all())
        torch.testing.assert_close(inputs[0][:,2:],b['m_f'])
        for label in ['Z09','Z10']:
            c=self.cfg(label);c['model']['coe'].pop('delta_dependence')
            with self.assertRaisesRegex(ValueError,'require controlled difference'):self.model(c)
        c=self.cfg('Z11');c['model']['coe']['delta_dependence']['parent']='Y12'
        with self.assertRaises(ValueError):self.model(c)

    def test_cpu_resume_skip_and_evaluation(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        import run_four_direction_exploration as base
        from report_history_followup import collect
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);suite=root/'suite';js={}
            for label in ['Z09','Z10','Z11','Z12']:
                job=copy.deepcopy(self.jobs[f'taxibj_{label}_seed7']);c=self.cfg(label);job['config']=c
                c['output_dir']=str(root/'outputs/v24-COE');c['train'].update(epochs=2,val_epoch=1)
                c['train']['scheduler']['total_epochs']=2;c['data'].update(batch_size=2,pin_memory=False)
                c['experiment_plan']['protocol']={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
                for split in ['train','val','test']:
                    path=root/(split+'.npz');np.savez(path,x_f_gt=np.random.RandomState(2).randn(4,2,3,8,8).astype('float32'));job['sources'][split]=str(path)
                js[job['variant']]=job
            c=self.cfg('Z09')
            for key in ['delta_dependence','history_followup','history_plane_interaction']:c['model']['coe'].pop(key)
            m=self.model(c);ref=root/'init.pth';torch.save({'model':m.state_dict(),'sha256':state_hash(m.state_dict())},ref)
            prepare_initialization(suite,js,ref)
            for name,job in js.items():
                run=reserve_run(suite,job);result=suite/'results'/f'{name}.json'
                train(job,run,result,'cpu',stop_after=1);train(job,run,result,'cpu')
                with patch.object(base,'subprocess_run',side_effect=AssertionError('must skip')):base.launch(suite,job,0)
                other=root/(name+'_full');train(job,other,root/(name+'.json'),'cpu')
                a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(other/'checkpoints/last.pth',weights_only=False)
                for key in ['model','optimizer','scheduler','scaler','rng_states']:self.assertEqual(state_hash(a[key]),state_hash(b[key]),key)
                evaluate_sets(run/'checkpoints/best.pth',job['config']['experiment_plan']['protocol'],job['sources']['test'],suite/'evaluations'/f'{name}.json',device='cpu')
                self.assertEqual(collect(suite,job['method'],job)[0]['status'],'finished')

    def test_report_stages_and_control_coverage(self):
        from report_delta_dependence import export_report,mechanism_conclusions
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);suite=root/'suite';suite.mkdir()
            p=export_report(root,suite,dict(jobs=self.jobs,references=self.refs));s=p.read_text()
            self.assertIn('阶段报告',s);self.assertIn('Z12',s);self.assertIn('差中之差',s)
            self.assertLess(s.index('实验具体做法'),s.index('全部结果'))
            self.assertEqual(len(load(suite/'summary.json')),12)
            # Exercise completed interpretation and factorial calculations without training.
            records={k:{'status':'finished','evaluations':{pr:{'metrics':{'mae':10.,'rmse':20.}} for pr in
                       ['in_distribution','unseen_combinations','unseen_geometry','unseen_triple']}} for k in load(suite/'summary.json')}
            self.assertIn('绝对历史乘积差值','\n'.join(mechanism_conclusions(records)))

if __name__=='__main__':unittest.main()
