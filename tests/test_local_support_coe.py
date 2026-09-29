from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
import torch
from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.local_support_coe import split_regions, merge_regions, transport_support
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator

ROOT = Path(__file__).resolve().parents[1]

def config(local=True, region=4):
    cfg=json.loads((ROOT/'configs/v24/coe_main_s4_e8_taxibj_base.json').read_text())
    patch=json.loads((ROOT/f'configs/v24/b3_c3/{"C3" if local else "B3"}.json').read_text())
    cfg=deep_update(cfg,patch)
    cfg['model']['main']['dim']=8
    cfg['model']['coe']['router_hidden_dim']=8
    cfg['model']['coe']['local_routing']['region_size']=[region,region]
    cfg['train']['amp']=False
    return cfg

class LocalSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_single_region_zero_support_correction_matches_b3(self):
        torch.manual_seed(7);base=DualBranchSTImputer.from_config(config(False)).eval()
        torch.manual_seed(7);local=DualBranchSTImputer.from_config(config(True,8)).eval()
        x=torch.randn(2,2,3,8,8);mask=(torch.rand_like(x)>.4).float()
        batch={'x_f_obs':x*mask,'m_f':mask}
        with torch.no_grad():a,b=base(batch),local(batch)
        torch.testing.assert_close(a['x_hat_main'],b['x_hat_main'],atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.equal(a['coe']['selected_experts'],b['coe']['selected_experts']))

    def test_no_feedback_keeps_initial_completion_but_updates_hidden(self):
        for local in (False, True):
            with self.subTest(local=local):
                cfg = config(local)
                self.assertFalse(cfg['model']['coe']['completion_feedback'])
                model = DualBranchSTImputer.from_config(cfg).eval()
                backbone = model.main_branch
                projections, features = [], []
                handles = [backbone.state_projection.register_forward_pre_hook(
                    lambda module, args: projections.append(args[0].detach().clone()))]
                for router in backbone.routers:
                    handles.append(router.register_forward_pre_hook(
                        lambda module, args: features.append(args[0].detach().clone())))
                x = torch.randn(2, 2, 3, 8, 8)
                mask = (torch.rand_like(x) > .4).float()
                with torch.no_grad():
                    outputs = model({'x_f_obs': x * mask, 'm_f': mask})
                for handle in handles:
                    handle.remove()
                dim, channels = backbone.dim, backbone.c_in
                initial = outputs['coe']['initial_completion']
                if local:
                    initial = backbone._split(initial)
                self.assertEqual(len(projections), 4)
                for projected, routed in zip(projections, features):
                    torch.testing.assert_close(projected[:, dim:dim+channels], initial, rtol=0, atol=0)
                    torch.testing.assert_close(routed[:, 2*dim:2*dim+2*channels],
                                               features[0][:, 2*dim:2*dim+2*channels], rtol=0, atol=0)
                    change_start = 2*dim + 2*channels + 4*10*channels
                    self.assertEqual(routed[:, change_start:change_start+2*channels].abs().max(), 0)
                self.assertFalse(torch.equal(projections[0][:, :dim], projections[1][:, :dim]))
                if local:
                    self.assertEqual(len(outputs['coe']['support_history']), 4)

    def test_halo_execution_matches_each_full_grid_expert_including_edges(self):
        model=DualBranchSTImputer.from_config(config()).main_branch.eval()
        model._layout=(2,8,8)
        x=torch.randn(2,8,3,8,8);tiles=model._split(x)
        model._original_mask=torch.zeros(2,2,3,8,8)
        model._reach=model._original_mask.clone();model._density=model._original_mask.clone()
        for eid,expert in enumerate(model.routed_experts()):
            model._execution=[];model._support_history=[]
            model._candidates=[transport_support(model._reach,model._density,e) for e in model.routed_experts()]
            weights=torch.zeros(8,8);weights[:,eid]=1
            with torch.no_grad():
                actual=model._merge(model._dispatch_weighted(tiles,weights))
                expected=expert(x)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6,msg=model.expert_names[eid])

    def test_selected_only_forward_backward_metrics_and_no_hidden_target(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg)
        for router in model.main_branch.routers:
            torch.nn.init.zeros_(router.base[-1].weight)
            router.base[-1].bias.data.copy_(torch.tensor([8.,7.,-8.,-9.,-10.,-11.,-12.,-13.]))
        calls={name:0 for name in model.main_branch.expert_names}
        handles=[]
        for name,expert in model.main_branch.pattern_experts.items():
            def hook(module,args,out,name=name):calls[name]+=args[0].shape[0]
            handles.append(expert.register_forward_hook(hook))
        x=torch.randn(2,2,3,8,8);mask=(torch.rand_like(x)>.4).float()
        batch={'x_f_obs':x*mask,'x_f_gt':x,'m_f':mask,'mask_family':torch.tensor([0,1])}
        outputs=model(batch)
        loss,logs=compute_coe_loss(outputs,batch,cfg);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(calls['T'],32);self.assertEqual(calls['S'],32)
        self.assertTrue(all(calls[n]==0 for n in ('TD','SD','TA','ST','TL','SL')))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.main_branch.routers[0].support_head.parameters()))
        coe=outputs['coe'];coe['mask_family']=batch['mask_family'].repeat_interleave(4)
        metrics=CoERoutingMetricAccumulator();metrics.update(coe)
        self.assertEqual(metrics.compute()['coe_routing_sample_count'],8)
        self.assertTrue(torch.equal(outputs['x_comp'][mask.bool()],x[mask.bool()]))
        model.eval()
        with torch.no_grad():
            before=model(batch)['x_hat_main']
            changed=dict(batch,x_f_gt=x+1000,x_f_obs=torch.where(mask.bool(),x,torch.full_like(x,999.)))
            after=model(changed)['x_hat_main']
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        for handle in handles:handle.remove()

    def test_support_uses_dilated_offsets_and_selected_region_only(self):
        model=DualBranchSTImputer.from_config(config()).main_branch
        mask=torch.zeros(1,2,5,8,8);mask[:,:,2,1,1]=1
        td=model.pattern_experts['TD'];r,d=transport_support(mask,mask,td)
        self.assertEqual(r[0,0,0,1,1],1);self.assertEqual(r[0,0,1,1,1],0)
        model._layout=(1,8,8);model._original_mask=mask
        model._reach=mask.clone();model._density=mask.clone()
        model._candidates=[transport_support(mask,mask,e) for e in model.routed_experts()]
        model._execution=[];model._support_history=[]
        weights=torch.zeros(4,8);weights[:,0]=.5;weights[:,1]=.5
        with torch.no_grad():model._dispatch_weighted(model._split(torch.randn(1,8,5,8,8)),weights)
        self.assertEqual(model._reach[0,0,0,1,1],0) # TD was not selected.
        self.assertEqual(model._reach[0,0,1,1,1],1) # T was selected.
        self.assertTrue((model._density[mask.bool()]==1).all())
        self.assertTrue((model._reach[:,:,:,4:,4:]==0).all())

if __name__=='__main__':unittest.main()
