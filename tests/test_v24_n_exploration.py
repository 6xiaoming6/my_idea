import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/v24'))
from run_n_exploration import jobs
from run_b3_c3 import jobs as baseline_jobs
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.spatial_scale_coe import observed_pool
from stmoe_imputer.data.diverse_masks import FAMILIES,ALL_FAMILIES,COMPOSITION_PARTS,make_diverse_mask,DiverseMaskSchedule


def config(variant):
    cfg=jobs(variants=(variant,))[0]['config']
    cfg['model']['main']['dim']=8
    cfg['model']['coe']['router_hidden_dim']=8
    cfg['train']['amp']=False
    return cfg


def batch():
    x=torch.randn(2,2,3,8,8)
    mask=(torch.rand_like(x)>.4).float()
    return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x}


class NExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_n1_to_n3_keep_baseline_structure_and_protocol_is_shared(self):
        plan=jobs()
        for job in plan:
            cfg=job['config'];families=cfg['data']['train_mask_diversity']['families']
            self.assertEqual(families,cfg['data']['eval_mask_diversity']['families'])
            self.assertEqual(len(families),4)
            self.assertFalse(set(families)&set(COMPOSITION_PARTS))
            self.assertEqual(cfg['train']['val_epoch'],5)
        for n,b in zip(('N1','N2','N3'),('B1','B2','B3')):
            self.assertEqual(jobs(variants=(n,))[0]['config']['model'],baseline_jobs(variants=(b,))[0]['config']['model'])

    def test_compositions_exact_reproducible_and_legacy_schedule_stays_nine(self):
        self.assertEqual(len(FAMILIES),9)
        self.assertEqual(DiverseMaskSchedule(90,{}).families,FAMILIES)
        for family in COMPOSITION_PARTS:
            for shape in ((12,32,32),(12,16,8),(3,8,8)):
                a=make_diverse_mask(shape,.4,family,np.random.default_rng(9))
                b=make_diverse_mask(shape,.4,family,np.random.default_rng(9))
                np.testing.assert_array_equal(a,b)
                self.assertEqual(int((a==0).sum()),round(np.prod(shape)*.4))
                self.assertTrue(np.isin(a,[0,1]).all())
        schedule=DiverseMaskSchedule(9,{'families':list(COMPOSITION_PARTS),'resample_each_epoch':False})
        a=schedule.sample(0,(12,8,8));schedule.set_epoch(9);b=schedule.sample(0,(12,8,8))
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
        self.assertIn(ALL_FAMILIES[a[1]],COMPOSITION_PARTS)

    def test_n4_router_inputs_frozen_but_expert_states_evolve(self):
        model=DualBranchSTImputer.from_config(config('N4'))
        routed=[];projected=[];handles=[]
        for router in model.main_branch.routers:
            handles.append(router.register_forward_pre_hook(lambda m,args:routed.append(args[0].detach().clone())))
        handles.append(model.main_branch.state_projection.register_forward_pre_hook(lambda m,args:projected.append(args[0].detach().clone())))
        model(batch())
        for handle in handles:handle.remove()
        self.assertEqual(len(routed),4)
        for features in routed[1:]:torch.testing.assert_close(features,routed[0],rtol=0,atol=0)
        self.assertFalse(torch.equal(projected[0][:,:8],projected[1][:,:8]))

    def test_observed_pool_never_uses_hidden_values_and_handles_empty(self):
        x=torch.tensor([2.,1000.,float('nan'),1000.]).reshape(1,1,1,2,2)
        mask=torch.tensor([1.,0.,0.,0.]).reshape_as(x)
        values,coverage=observed_pool(x,mask)
        self.assertEqual(values.item(),2.);self.assertEqual(coverage.item(),.25)
        values,coverage=observed_pool(x,torch.zeros_like(mask))
        self.assertEqual(values.item(),0.);self.assertEqual(coverage.item(),0.)

    def test_zero_scale_heads_match_fixed_n5_and_no_target_leakage(self):
        torch.manual_seed(7);fixed=DualBranchSTImputer.from_config(config('N5')).eval()
        torch.manual_seed(7);adaptive=DualBranchSTImputer.from_config(config('N6')).eval()
        for key,value in fixed.state_dict().items():torch.testing.assert_close(value,adaptive.state_dict()[key])
        data=batch()
        with torch.no_grad():
            a=fixed(data);b=adaptive(data)
            altered={**data,'x_f_gt':torch.full_like(data['x_f_gt'],1e8),
                     'x_f_obs':torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))}
            c=adaptive(altered)
        torch.testing.assert_close(a['x_hat_main'],b['x_hat_main'],rtol=0,atol=0)
        torch.testing.assert_close(b['x_hat_main'],c['x_hat_main'],rtol=0,atol=0)
        torch.testing.assert_close(b['coe']['selected_scales'],torch.tensor([[1,1,0,0],[1,1,0,0]]))

    def test_adaptive_scales_exact_budget_sparse_execution_and_gradient(self):
        for prefer_coarse in (True,False):
            torch.manual_seed(7);model=DualBranchSTImputer.from_config(config('N6')).train()
            backbone=model.main_branch
            for head in backbone.scale_routers:
                head[-1].bias.data.copy_(torch.tensor([0.,.1] if prefer_coarse else [.1,0.]))
            for router in backbone.routers:
                torch.nn.init.zeros_(router[-1].weight)
                router[-1].bias.data.copy_(torch.arange(8,0,-1).float())
            calls=[];handles=[]
            for eid,expert in enumerate(backbone.routed_experts()):
                def hook(m,args,out,eid=eid):calls.append((eid,args[0].shape[0],args[0].shape[-2:]))
                handles.append(expert.register_forward_hook(hook))
            data=batch();out=model(data)
            for h in handles:h.remove()
            scales=out['coe']['selected_scales']
            self.assertTrue((scales.sum(1)==2).all())
            self.assertEqual(sum(n for _,n,_ in calls),2*4*2)
            self.assertEqual(sum(n for _,n,shape in calls if shape==(4,4)),8)
            self.assertEqual(sum(n for _,n,shape in calls if shape==(8,8)),8)
            self.assertEqual(set(eid for eid,_,_ in calls),{0,1})
            loss=(out['x_hat_main']-data['x_f_gt']).square().mean();loss.backward()
            grad=backbone.scale_routers[0][-1].bias.grad
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)
            for expert in backbone.routed_experts()[2:]:
                self.assertTrue(all(p.grad is None for p in expert.parameters()))


if __name__=='__main__':unittest.main()
