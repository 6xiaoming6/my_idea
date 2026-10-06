import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/v24'))
from run_r_scale_followup import jobs
from run_r_exploration import jobs as r_jobs
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


def cfg(name):
    c=jobs(variants=(name,))[0]['config']
    c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
    c['train']['amp']=False
    return c


def model(name):
    torch.manual_seed(7)
    return DualBranchSTImputer.from_config(cfg(name))


def batch(n=2):
    x=torch.randn(n,2,3,8,8);mask=(torch.rand_like(x)>.4).float()
    return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x}


def force_experts(m):
    for router in m.main_branch.routers:
        router[-1].weight.data.zero_();router[-1].bias.data.copy_(torch.arange(8,0,-1))


class ScaleFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_only_intended_config_changes(self):
        for dataset in ('taxibj','bikenyc'):
            base=r_jobs(dataset,61,13,('R7',))[0]['config']
            for job in jobs(dataset,61,13):
                c=copy.deepcopy(job['config']);b=copy.deepcopy(base)
                c['model']['coe']['spatial_scale']=b['model']['coe']['spatial_scale']
                c.pop('experiment_plan');b.pop('experiment_plan')
                self.assertEqual(c,b)
                self.assertEqual(c['train']['epochs'],61);self.assertEqual(c['data']['batch_size'],13)

    def test_soft_start_schedule_execution_and_gradients(self):
        m=model('R11').train();force_experts(m);data=batch();back=m.main_branch
        for epoch,s in [(1,1),(5,1),(6,5/6),(8,.5),(10,1/6),(11,0),(100,0)]:
            back.set_routing_epoch(epoch);calls=[]
            handles=[e.register_forward_pre_hook(lambda m,args,i=i:calls.append((i,args[0].shape[0],args[0].shape[-1]))) for i,e in enumerate(back.routed_experts())]
            m.zero_grad();out=m(data)
            for h in handles:h.remove()
            coe=out['coe'];p=coe['scale_probabilities'];hard=torch.nn.functional.one_hot(coe['selected_scales'],2).float()
            torch.testing.assert_close(coe['scale_weights'],s*p+(1-s)*hard)
            torch.testing.assert_close(coe['scale_weights'].sum(-1),torch.ones(2,4))
            self.assertEqual(sum(n for _,n,_ in calls),32 if s>0 else 16)
            self.assertEqual(set(i for i,_,_ in calls),{0,1})
            if s>0:self.assertEqual(sum(n for _,n,w in calls if w==4),16)
            acc=CoERoutingMetricAccumulator();acc.update(coe);metrics=acc.compute()
            self.assertEqual(metrics['coe_expert_execution_count'],16 if s>0 else 8)
            self.assertEqual(metrics['coe_expert_grid_equivalents'],10 if s>0 else 5)
            loss=(out['x_hat_final']-data['x_f_gt']).square().mean();loss.backward()
            for head in back.scale_routers:
                grad=head[-1].bias.grad
                self.assertIsNotNone(grad);self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum(),0)
            for e in back.routed_experts()[2:]:self.assertTrue(all(p.grad is None for p in e.parameters()))
        back.set_routing_epoch(1);m.eval()
        with torch.no_grad():out=m(data)
        self.assertTrue((out['coe']['scale_executed'].sum(-1)==1).all())

    def test_soft_fusion_both_scales_eval_and_extreme_logits(self):
        m=model('R12');force_experts(m);data=batch()
        for training in (True,False):
            m.train(training)
            for logits in ([0.,0.],[1.,2.],[1000.,-1000.]):
                for h in m.main_branch.scale_routers:h[-1].bias.data.copy_(torch.tensor(logits))
                calls=[];handles=[e.register_forward_pre_hook(lambda m,args:calls.append(args[0].shape[0])) for e in m.main_branch.routed_experts()]
                out=m(data)
                for h in handles:h.remove()
                self.assertEqual(sum(calls),32)
                self.assertTrue(out['coe']['scale_executed'].all())
                torch.testing.assert_close(out['coe']['scale_weights'],out['coe']['scale_probabilities'])
                self.assertTrue(torch.isfinite(out['x_hat_final']).all())

    def test_dense_fusion_is_weighted_sum_not_hard_dominant(self):
        # Isolate actual dispatch: known constant outputs at each resolution.
        m=model('R12').eval();back=m.main_branch;force_experts(m)
        data=batch();original=back._dispatch_weighted;seen=[]
        from unittest.mock import patch
        from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE
        def branch(self,unified,weights,step=0):
            return torch.ones_like(unified)*(2 if unified.shape[-1]==8 else 6)
        def wrapped(unified,weights,step=0):
            result=original(unified,weights,step)
            w=back._scale_weight_history[-1]
            expected=(2*w[:,0]+6*w[:,1])[:,None,None,None,None].expand_as(result)
            torch.testing.assert_close(result,expected)
            seen.append(step)
            return result
        back._dispatch_weighted=wrapped
        with patch.object(TemporalSpatialCoE,'_dispatch_weighted',branch):m(data)
        self.assertEqual(seen,list(range(4)))

    def test_exact_mixed_batch_metrics_and_merge(self):
        m=model('R11');acc=CoERoutingMetricAccumulator();parts=[]
        for n,training in [(3,True),(1,False)]:
            m.train(training);m.main_branch.set_routing_epoch(1)
            out=m(batch(n));out['coe']['mask_family']=torch.zeros(n,dtype=torch.long)
            acc.update(out['coe']);part=CoERoutingMetricAccumulator();part.update(out['coe']);parts.append(part)
        merged=CoERoutingMetricAccumulator()
        for a in parts:merged.merge(a)
        a=acc.compute();self.assertEqual(a,merged.compute())
        self.assertEqual(a['coe_expert_execution_count'],14.)
        self.assertEqual(a['coe_expert_grid_equivalents'],8.75)
        self.assertEqual(a['coe_condition_family_random_point_expert_grid_equivalents'],8.75)
        self.assertEqual(a['coe_step1_both_scales_fraction'],.75)
        self.assertEqual(a['coe_step1_coarse_execution_fraction'],1.)
        self.assertEqual(a['coe_step1_fine_execution_fraction'],.75)

    def test_checkpoint_no_target_leak_and_eval_isolation(self):
        data=batch();changed={**data,'x_f_gt':torch.full_like(data['x_f_gt'],1e8),'x_f_obs':torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))}
        for name in ('R11','R12'):
            m=model(name).train();m.main_branch.set_routing_epoch(8)
            state=copy.deepcopy(m.state_dict());out=m(data)
            restored=model(name).train();restored.load_state_dict(state);restored.main_branch.set_routing_epoch(8)
            torch.testing.assert_close(out['x_hat_final'],restored(data)['x_hat_final'],rtol=0,atol=0)
            m.eval()
            with torch.no_grad():a=m(data);b=m(changed)
            torch.testing.assert_close(a['x_hat_final'],b['x_hat_final'],rtol=0,atol=0)
            m.train();after=m(data)
            torch.testing.assert_close(out['x_hat_final'],after['x_hat_final'],rtol=0,atol=0)

    def test_legacy_forward_unchanged_against_frozen_source(self):
        path=ROOT/'outputs/v24-COE/experiments/r_exploration/taxibj/b21d7928e67f4565/source_snapshot/src/stmoe_imputer/models/spatial_scale_coe.py'
        if not path.exists():self.skipTest('Historical frozen source unavailable')
        spec=importlib.util.spec_from_file_location('stmoe_imputer.models._legacy_scale_check',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        data=batch()
        for name in ('R2','R3','R4','R5','R6','R7'):
            c=r_jobs(variants=(name,))[0]['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
            torch.manual_seed(7);current=DualBranchSTImputer.from_config(c)
            previous=copy.deepcopy(current);previous.main_branch.__class__=old.SpatialScaleCoE
            for training in (False,True):
                current.train(training);previous.train(training)
                a=current(data);b=previous(data)
                torch.testing.assert_close(a['x_hat_final'],b['x_hat_final'],rtol=0,atol=0)
                aa=CoERoutingMetricAccumulator();bb=CoERoutingMetricAccumulator();aa.update(a['coe']);bb.update(b['coe'])
                self.assertEqual(aa.compute(),bb.compute())

if __name__=='__main__':unittest.main()
