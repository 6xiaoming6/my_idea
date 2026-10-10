import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/v24')]
from run_post_fusion_ffn import jobs, prepare_initialization, REFERENCE_REL, REFERENCE_NAME
from run_b3_c3 import load
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash, common_initialization


class PostFFNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.jobs = jobs()

    def config(self, method='F01'):
        c = copy.deepcopy(self.jobs[f'taxibj_{method}_seed7']['config'])
        c['model']['main']['dim'] = 8
        c['model']['coe']['router_hidden_dim'] = 8
        return c

    def build(self, c):
        torch.manual_seed(7)
        return DualBranchSTImputer.from_config(c)

    def batch(self):
        g = torch.Generator().manual_seed(13)
        x = torch.randn(2,2,3,8,8,generator=g)
        mask = (torch.rand(x.shape,generator=g)>.4).float()
        return dict(x_f_obs=x*mask,x_f_gt=x,m_f=mask,mask_family=torch.tensor([0,1]))

    def test_plan_common_initialization_and_rng(self):
        self.assertEqual(len(self.jobs),2)
        old = load(ROOT/REFERENCE_REL/'plan.json')['jobs'][REFERENCE_NAME]['config']
        for j in self.jobs.values():
            c = copy.deepcopy(j['config']); c['model']['coe'].pop('post_fusion_ffn')
            c.pop('experiment_plan'); ref = copy.deepcopy(old); ref.pop('experiment_plan')
            self.assertEqual(c,ref)
        c = self.config(); basec = copy.deepcopy(c); basec['model']['coe'].pop('post_fusion_ffn')
        base = self.build(basec); rng = torch.get_rng_state()
        shared = self.build(c); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        separate = self.build(self.config('F02')); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in base.state_dict().items():
            torch.testing.assert_close(v,shared.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(v,separate.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(len(shared.main_branch.post_ffns),1)
        self.assertEqual(len(separate.main_branch.post_ffns),4)
        ptrs = [next(f.parameters()).data_ptr() for f in separate.main_branch.post_ffns]
        self.assertEqual(len(set(ptrs)),4)
        for f in separate.main_branch.post_ffns:
            self.assertEqual(state_hash(f.state_dict()),state_hash(shared.main_branch.post_ffns[0].state_dict()))

    def test_placement_sparse_calls_gradients_and_identity(self):
        batch = self.batch(); predictions = []
        for method in ('F01','F02'):
            c = self.config(method); model = self.build(c); main = model.main_branch
            expert_calls=[]; ffn_calls=[]; decoded=[]; memories=[]; transformed=[]
            hooks=[e.register_forward_pre_hook(lambda e,args:expert_calls.append((len(args[0]),args[0].shape[-1]))) for e in main.routed_experts(0)]
            hooks += [f.register_forward_hook(lambda f,args,out:(ffn_calls.append(tuple(args[0].shape)),transformed.append(out.detach().clone())) and None) for f in main.post_ffns]
            hooks.append(main.decoder.register_forward_pre_hook(lambda decoder,args:decoded.append(args[0].detach().clone())))
            original_memory=main._memory
            def memory(context):
                memories.append(context['h'].detach().clone());return original_memory(context)
            with patch.object(main,'_memory',side_effect=memory):
                out=model(batch)
            for hook in hooks:hook.remove()
            self.assertEqual(sum(n for n,w in expert_calls),16)
            self.assertEqual({w for n,w in expert_calls},{2,4,8})
            self.assertEqual(ffn_calls,[(2,8,3,8,8)]*4)
            # Decode and the next round consume the FFN output.
            for actual,expected in zip(decoded[-4:],transformed):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for actual,expected in zip(memories[1:],transformed[:-1]):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            loss,_=compute_coe_loss(out,batch,c);loss.backward()
            for block in list(main.post_ffns)+[main.history_gate]:
                grads=[p.grad for p in block.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads))
            predictions.append(out['x_hat_main'].detach())
            basec=copy.deepcopy(c);basec['model']['coe'].pop('post_fusion_ffn');base=self.build(basec)
            with torch.no_grad():
                for f in main.post_ffns:f.down.weight.zero_();f.down.bias.zero_()
            torch.testing.assert_close(model(batch)['x_hat_main'],base(batch)['x_hat_main'],rtol=0,atol=0)
        torch.testing.assert_close(*predictions,rtol=0,atol=0)

    def test_no_target_leak_or_persistent_state_and_invalid_spec(self):
        c=self.config();model=self.build(c).eval();batch=self.batch();pred=model(batch)['x_hat_main']
        torch.testing.assert_close(pred,model(dict(batch,x_f_gt=batch['x_f_gt']*100))['x_hat_main'],rtol=0,atol=0)
        model(dict(batch,x_f_obs=batch['x_f_obs']*2))
        torch.testing.assert_close(pred,model(batch)['x_hat_main'],rtol=0,atol=0)
        c['model']['coe']['post_fusion_ffn']['expansion']=2
        with self.assertRaises(ValueError):self.build(c)

    def test_templates_full_resume_evaluation_skip_and_stage_report(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from report_post_fusion_ffn import export_report
        import run_four_direction_exploration as base
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);suite=root/'suite';js=copy.deepcopy(self.jobs)
            for name,j in js.items():j['config']=self.config(j['method'])
            c=copy.deepcopy(next(iter(js.values()))['config']);c['model']['coe'].pop('post_fusion_ffn')
            initial=self.build(c).state_dict();refinit=root/'reference.pth'
            torch.save({'model':initial,'sha256':state_hash(initial)},refinit)
            prepare_initialization(suite,js,refinit);prepare_initialization(suite,js,refinit)
            for j in js.values():common_initialization(self.build(j['config']),j)
            j=js['taxibj_F02_seed7'];c=j['config']
            c['train'].update(epochs=2,val_epoch=1);c['train']['scheduler']['total_epochs']=2
            c['data'].update(batch_size=2,pin_memory=False);j['sources']={}
            protocol={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
            c['experiment_plan']['protocol']=protocol
            for split in ('train','val','test'):
                p=root/(split+'.npz');np.savez(p,x_f_gt=np.random.RandomState(2).randn(4,2,3,8,8).astype('float32'));j['sources'][split]=str(p)
            run=root/'resumed';result=suite/'results'/(j['variant']+'.json')
            train(j,run,result,'cpu',stop_after=1);r=train(j,run,result,'cpu');self.assertEqual(train(j,run,result,'cpu'),r)
            with patch.object(base,'subprocess_run',side_effect=AssertionError('completed run must skip')):base.launch(suite,j,0)
            train(j,root/'full',root/'full.json','cpu')
            a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(root/'full/checkpoints/last.pth',weights_only=False)
            for k in ('model','optimizer','scheduler','scaler','rng_states'):
                self.assertEqual(state_hash(a[k]),state_hash(b[k]),k)
            self.assertEqual(state_hash(a['training_state']['w_training']),state_hash(b['training_state']['w_training']))
            evaluate_sets(run/'checkpoints/best.pth',protocol,j['sources']['test'],suite/'evaluations'/(j['variant']+'.json'),device='cpu')
            ref=suite/'reference';ref.mkdir();original=ROOT/REFERENCE_REL
            for filename,source in {'config.json':original/'configs'/f'{REFERENCE_NAME}.json','result.json':original/'results'/f'{REFERENCE_NAME}.json','evaluations.json':original/'evaluations'/f'{REFERENCE_NAME}.json'}.items():shutil.copyfile(source,ref/filename)
            oldrun=Path(load(ref/'result.json')['run_dir'])
            for filename,source in {'metrics.jsonl':oldrun/'logs/metrics.jsonl','training_metadata.json':oldrun/'training_metadata.json'}.items():shutil.copyfile(source,ref/filename)
            path=export_report(root,suite,{'jobs':js,'reference':{'name':REFERENCE_NAME}})
            report=path.read_text();self.assertIn('阶段报告',report);self.assertLess(report.index('具体方法'),report.index('完整结果'))
            status=load(suite/'summary.json')
            self.assertEqual(status['taxibj_F02_seed7']['status'],'finished')
            self.assertEqual(status[REFERENCE_NAME]['status'],'finished')
            self.assertEqual(status['taxibj_F01_seed7']['status'],'pending')


if __name__=='__main__':unittest.main()
