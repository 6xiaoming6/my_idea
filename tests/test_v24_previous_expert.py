from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.coe_router import PreviousExpertRouter
from stmoe_imputer.losses import compute_coe_loss
from test_v24_coe import compact_config,make_batch

ROOT=Path(__file__).resolve().parents[1]


def candidate(context):
    return compact_config(num_steps=4,expert_pool=['T','S','TD','SD','TA','ST'],
                          previous_expert_context=context)


class PreviousExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_zero_initialization_preserves_all_common_weights_rng_and_predictions(self):
        torch.manual_seed(7);d=DualBranchSTImputer.from_config(candidate(False));rng_d=torch.get_rng_state()
        torch.manual_seed(7);e=DualBranchSTImputer.from_config(candidate(True));rng_e=torch.get_rng_state()
        torch.testing.assert_close(rng_d,rng_e,atol=0,rtol=0)
        for k,v in d.state_dict().items():torch.testing.assert_close(v,e.state_dict()[k],atol=0,rtol=0)
        extras=set(e.state_dict())-set(d.state_dict())
        self.assertEqual(len(extras),3);self.assertTrue(all(k.endswith('previous_expert_embedding') for k in extras))
        batch=make_batch()
        for training in [False,True]:
            d.train(training);e.train(training)
            torch.manual_seed(17);a=d(batch)
            torch.manual_seed(17);b=e(batch)
            torch.testing.assert_close(a['x_hat_main'],b['x_hat_main'],atol=0,rtol=0)
            torch.testing.assert_close(a['coe']['route_logits'],b['coe']['route_logits'],atol=0,rtol=0)
            torch.testing.assert_close(a['coe']['paths'],b['coe']['paths'],atol=0,rtol=0)

    def test_context_is_actual_previous_sampled_expert_and_resets_each_forward(self):
        model=DualBranchSTImputer.from_config(candidate(True)).train()
        for router in model.main_branch.routers:
            nn.init.zeros_(router[-1].weight);nn.init.zeros_(router[-1].bias)
        count=[0]
        def sampled(logits,**kwargs):
            count[0]+=1
            choices=(torch.arange(logits.shape[0])+count[0])%6
            return torch.nn.functional.one_hot(choices,6).to(logits)
        with patch('stmoe_imputer.models.temporal_spatial_coe.F.gumbel_softmax',side_effect=sampled):
            output=model(make_batch())['coe']
        context=output['previous_expert_choices']
        self.assertFalse(context.requires_grad)
        self.assertTrue(torch.equal(context[:,0],torch.zeros_like(context[:,0])))
        torch.testing.assert_close(context[:,1:],torch.nn.functional.one_hot(output['paths'][:,:-1],6).float())
        self.assertFalse(torch.equal(output['paths'],output['route_logits'].argmax(-1)))
        model.eval()
        with torch.no_grad():out=model(make_batch(seed=18))['coe']
        self.assertTrue(torch.equal(out['previous_expert_choices'][:,0],torch.zeros_like(context[:,0])))
        torch.testing.assert_close(out['previous_expert_choices'][:,1:],torch.nn.functional.one_hot(out['paths'][:,:-1],6).float())

    def test_context_can_affect_logits_but_does_not_backpropagate_through_choice(self):
        base=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,8),nn.GELU(),nn.Linear(8,6))
        router=PreviousExpertRouter(base,6)
        with torch.no_grad():router.previous_expert_embedding[1]=torch.arange(8).float()
        features=torch.randn(2,5);one=torch.nn.functional.one_hot(torch.tensor([0,0]),6).float().requires_grad_()
        two=torch.nn.functional.one_hot(torch.tensor([1,1]),6).float()
        first=router(features,one);second=router(features,two)
        self.assertFalse(torch.allclose(first,second))
        first.sum().backward();self.assertIsNone(one.grad)
        self.assertGreater(router.previous_expert_embedding.grad.abs().sum().item(),0)

    def test_real_loss_trains_context_and_cpu_amp_is_finite(self):
        cfg=candidate(True);model=DualBranchSTImputer.from_config(cfg).train();batch=make_batch()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            outputs=model(batch);loss,_=compute_coe_loss(outputs,batch,cfg)
        loss.backward()
        for router in model.main_branch.routers[1:]:
            grad=router.previous_expert_embedding.grad
            self.assertIsNotNone(grad);self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_unsupported_soft_history_fails_explicitly(self):
        for override in [{'routing_mode':'soft'},{'routing_warmup_epochs':5},{'use_routed':False},{'router_features':'grouped'}]:
            with self.assertRaises(ValueError):DualBranchSTImputer.from_config(deep_update(candidate(True),{'model':{'coe':override}}))

    def test_five_job_order_budgets_and_e_equals_d_plus_context(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/v24'))
        import run_experiments as runner
        plan,*_=runner.policy_plan(ROOT/'configs/v24/abcde_experiments.json','abcde');runner.validate_plan(plan)
        self.assertEqual([r['variant'] for r in plan['runs']],['abc_a','abc_b','abc_c','abc_d','abc_e'])
        for r in plan['runs']:
            self.assertEqual(r['config']['train']['epochs'],70)
            self.assertEqual(r['config']['train']['scheduler']['total_epochs'],70)
        d,e=[copy.deepcopy(r['config']) for r in plan['runs'][-2:]]
        self.assertTrue(e['model']['coe'].pop('previous_expert_context'))
        for c in (d,e):c.pop('experiment_plan')
        self.assertEqual(d,e)
        with tempfile.TemporaryDirectory() as tmp:
            manifest=copy.deepcopy(plan);manifest['suite_fingerprint']='test'
            def result(run,*_):
                i=['abc_a','abc_b','abc_c','abc_d','abc_e'].index(run['variant'])
                return {'best_epoch':2,'best_val_mae':10-i,'test':{'mae':10-i,'rmse':20-i},'total_time_sec':1,'run_dir':'test'}
            with patch.object(runner,'result_for',side_effect=result):runner.summarize(manifest,Path(tmp))
            comparisons=json.loads((Path(tmp)/'comparison.json').read_text())
            pair=next(x for x in comparisons['paired'] if x['reference']=='abc_d' and x['variant']=='abc_e')
            self.assertEqual(pair['test_mae_delta_vs_abc_d'],-1)

    def test_single_gpu_launcher_rejects_work_on_any_gpu(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/v24'))
        import run_abcde
        import subprocess
        for records,expected in [('123, GPU-other, python\n',RuntimeError),('',None)]:
            answers=[subprocess.CompletedProcess([],0,stdout='0\n1\n'),subprocess.CompletedProcess([],0,stdout=records)]
            with patch.object(run_abcde.subprocess,'run',side_effect=answers):
                if expected:
                    with self.assertRaises(expected):run_abcde.check_gpu_idle(0)
                else:run_abcde.check_gpu_idle(0)

if __name__=='__main__':unittest.main()
