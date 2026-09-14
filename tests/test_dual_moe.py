from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from stmoe_imputer.config import deep_update, load_config
from stmoe_imputer.engine import build_optimizer, evaluate, train_one_epoch
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer, DualMoEBackbone
from stmoe_imputer.models.dual_moe import PointRouter, RegionInteraction, restore_regions
from stmoe_imputer.routing_metrics import DualMoEMetricAccumulator
from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def config(channels=2):
    cfg = deep_update(load_config(ROOT/'configs/presets/default.json'),
                      load_config(ROOT/'configs/presets/dual_moe_smoke.json'))
    cfg['model']['c_in'] = channels
    return cfg


def batch(channels=2, t=3, h=8, w=12, n=2):
    x = torch.randn(n, channels, t, h, w)
    m = (torch.rand(n, 1, t, h, w) > .4).float()
    return {'x_f_gt': x, 'x_f_obs': x*m, 'm_f': m}


class DualMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_is_new_backbone_and_does_not_build_legacy_model(self):
        model = DualBranchSTImputer.from_config(config())
        self.assertIsInstance(model.main_branch, DualMoEBackbone)
        self.assertFalse(model.alpha.requires_grad)
        self.assertFalse(any('controller' in n or 'refiner' in n for n, _ in model.named_parameters()))

    def test_invalid_configuration_and_no_silent_synthetic_training(self):
        for patch in [{'strides':[4,2]},{'strides':[2,3.5]},{'completion_mode':'topk'},{'min_std':0}]:
            cfg=config();cfg['model']['dual_moe'].update(patch)
            with self.assertRaises(ValueError):DualBranchSTImputer.from_config(cfg)
        cfg=config();cfg['model']['aux']['enabled']=True
        with self.assertRaises(ValueError):DualBranchSTImputer.from_config(cfg)
        result=subprocess.run([sys.executable,'scripts/train.py','--no_plot'],cwd=ROOT,
                              capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('no silent fallback',result.stderr)

    def test_support_communication_ablation_does_not_change_aggregation(self):
        cfg=config();first=DualBranchSTImputer.from_config(cfg).eval()
        cfg['model']['dual_moe']['completion_use_support']=False
        second=DualBranchSTImputer.from_config(cfg).eval();second.load_state_dict(first.state_dict())
        with torch.no_grad():
            first.main_branch.completion_router.head.weight[0,-6:]=1.
            second.main_branch.completion_router.head.weight.copy_(first.main_branch.completion_router.head.weight)
        data=batch();a,b=first(data),second(data)
        for scale in ['mid','coarse']:
            torch.testing.assert_close(a['aggregation_mass'][scale],b['aggregation_mass'][scale],rtol=0,atol=0)
            torch.testing.assert_close(a['scale_predictions'][scale],b['scale_predictions'][scale],rtol=0,atol=0)
        self.assertGreater(float((a['completion_gates']-b['completion_gates']).abs().max()),0)

    def test_dataset_shapes_and_odd_or_single_cell_grids(self):
        for c,t,h,w in [(2,12,32,32),(2,12,24,12),(1,7,32,32),(1,2,7,11),(1,1,1,1)]:
            with self.subTest(shape=(c,t,h,w)):
                cfg=config(c); model=DualBranchSTImputer.from_config(cfg)
                data=batch(c,t,h,w,n=1); out=model(data)
                self.assertEqual(out['x_hat_final'].shape,data['x_f_gt'].shape)
                self.assertTrue(torch.isfinite(out['x_hat_final']).all())
                for s,k in zip(['mid','coarse'],cfg['model']['dual_moe']['coarse_nodes']):
                    k=min(k,h*w)
                    self.assertEqual(out['aggregation_mass'][s].shape,(1,3,t,k))
                    a=out['region_assignments'][s]
                    self.assertEqual(a.shape,(1,3,t,h*w,k))
                    torch.testing.assert_close(a.sum(-1),torch.ones_like(a[...,0]))
                loss,_=compute_main_stage_loss(out,data,cfg);loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))

    def test_no_hidden_ground_truth_or_external_scale_input_leak(self):
        model=DualBranchSTImputer.from_config(config()).eval(); data=batch()
        first=model(data)
        changed=copy.deepcopy(data)
        changed['x_f_obs']=torch.where(data['m_f'].bool(),data['x_f_obs'],torch.full_like(data['x_f_obs'],float('nan')))
        changed['x_f_gt']=torch.full_like(data['x_f_gt'],float('nan'))
        for key in ['x_m_obs','x_c_obs','m_m','m_c','r_m','r_c']:
            changed[key]=torch.tensor(float('nan'))
        second=model(changed)
        for key in ['x_hat_final','x_comp','completion_gates']:
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        changed.pop('x_f_gt'); self.assertTrue(torch.isfinite(model(changed)['x_comp']).all())

    def test_missing_source_gates_cannot_change_pooling_or_readout(self):
        model=DualBranchSTImputer.from_config(config()).eval(); data=batch()
        reference=model(data)['x_hat_final']
        def hook(module, inputs, output):
            replacement=torch.zeros_like(output);replacement[:,2]=1
            return torch.where(data['m_f'].bool(),output,replacement)
        hooks=[a.router.register_forward_hook(hook) for a in model.main_branch.aggregation.values()]
        try:torch.testing.assert_close(reference,model(data)['x_hat_final'],rtol=0,atol=0)
        finally:
            for h in hooks:h.remove()

    def test_mass_conservation_and_effective_count_deduplicates_experts(self):
        model=DualBranchSTImputer.from_config(config()).eval(); data=batch(t=1,h=8,w=8,n=1)
        data['m_f'].zero_();data['m_f'][...,2,2]=1;data['x_f_obs']=data['x_f_gt']*data['m_f']
        out=model(data)
        for scale in ['mid','coarse']:
            mass=out['aggregation_mass'][scale];neff=out['aggregation_effective_count'][scale]
            torch.testing.assert_close(mass.sum(),torch.tensor(1.),atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(neff[mass>1e-5],torch.ones_like(neff[mass>1e-5]),atol=1e-5,rtol=1e-5)

    def test_two_routers_receive_gradients_and_predictions_mix_exactly(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg);data=batch()
        out=model(data);gate=out['completion_gates']
        pred=sum(gate[:,i:i+1]*out['scale_predictions'][s] for i,s in enumerate(['fine','mid','coarse']))
        torch.testing.assert_close(out['x_hat_final'],pred)
        torch.testing.assert_close(gate.sum(1),torch.ones_like(gate[:,0]))
        loss,_=compute_main_stage_loss(out,data,cfg);loss.backward()
        for name in ['mid','coarse']:
            self.assertGreater(float(model.main_branch.aggregation[name].router.head.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.main_branch.completion_router.head.weight.grad.abs().sum()),0)

    def test_router_ablations_keep_common_initial_parameters(self):
        reference=None
        for agg in ['learned','static','uniform']:
            for completion in ['learned','static','uniform']:
                cfg=config();cfg['model']['dual_moe'].update(aggregation_mode=agg,completion_mode=completion)
                torch.manual_seed(17);model=DualBranchSTImputer.from_config(cfg)
                state=model.state_dict()
                if reference is None:reference=state
                for k,v in state.items():torch.testing.assert_close(v,reference[k],rtol=0,atol=0)
                data=batch();out=model(data);loss,_=compute_main_stage_loss(out,data,cfg);loss.backward()
                self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_all_observed_all_missing_and_reject_bad_inputs(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg)
        for value in [0.,1.]:
            data=batch();data['m_f'].fill_(value);data['x_f_obs']=data['x_f_gt']*value
            out=model(data);loss,_=compute_main_stage_loss(out,data,cfg)
            self.assertTrue(torch.isfinite(loss));loss.backward()
            if value==1:self.assertEqual(float(loss),0.)
        data=batch();data['m_f'].fill_(.5)
        with self.assertRaisesRegex(ValueError,'binary'):model(data)
        data=batch();data['m_f'].fill_(1);data['x_f_obs'].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'finite'):model(data)

    def test_normalized_loss_not_legacy_auxiliary_loss(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg);data=batch();out=model(data)
        expected,_=compute_main_stage_loss(out,data,cfg)
        cfg['loss'].update(lambda_cross=100.,lambda_v14_regret=100.,lambda_balance=100.)
        actual,_=compute_main_stage_loss(out,data,cfg);torch.testing.assert_close(actual,expected)
        cfg['loss']['dual_moe_expert_weight']=.1
        extra,logs=compute_main_stage_loss(out,data,cfg)
        torch.testing.assert_close(extra,expected+.1*logs['l_dual_expert'])

    def test_exact_gate_and_expert_metrics_independent_of_batch_partition(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg).eval();data=batch(n=3)
        out=model(data);whole=DualMoEMetricAccumulator();whole.update(out,data)
        split=DualMoEMetricAccumulator()
        for lo,hi in [(0,2),(2,3)]:
            part={k:v[lo:hi] for k,v in data.items()}
            sliced={k:({n:x[lo:hi] for n,x in v.items()} if isinstance(v,dict) else v[lo:hi] if torch.is_tensor(v) and v.ndim else v) for k,v in out.items()}
            split.update(sliced,part)
        for k,v in whole.compute().items():self.assertAlmostEqual(v,split.compute()[k],delta=1e-5*max(1,abs(v)))
        self.assertEqual(whole.compute()['completion_missing_count'],float((1-data['m_f']).sum()))
        self.assertEqual(whole.compute()['aggregation_mid_observed_count'],float(data['m_f'].sum()))

    def test_training_eval_and_checkpoint_roundtrip(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg);optim=build_optimizer(model,cfg);data=batch()
        train=train_one_epoch(model,[data],optim,torch.device('cpu'),cfg,1)
        val=evaluate(model,[data],torch.device('cpu'),cfg)
        self.assertTrue(all(math.isfinite(v) for v in train.values()))
        self.assertIn('mae_expert_mid',val);self.assertIn('completion_missing_fine_mean',val)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'best.pt';save_checkpoint(path,model,optim,1,val,cfg)
            clone=DualBranchSTImputer.from_config(cfg).eval();load_checkpoint(path,clone,map_location='cpu')
            torch.testing.assert_close(model(data)['x_hat_final'],clone(data)['x_hat_final'])

    def test_region_labels_are_not_spatial_grid_positions(self):
        model=DualBranchSTImputer.from_config(config()).eval(); data=batch()
        reference=model(data)['x_hat_final'].detach()
        with torch.no_grad():
            for aggregation in model.main_branch.aggregation.values():
                for expert in aggregation.experts:
                    expert.slots.copy_(expert.slots[torch.randperm(expert.slots.shape[0])])
        torch.testing.assert_close(reference,model(data)['x_hat_final'],rtol=1e-5,atol=1e-5)
        block=RegionInteraction(8,0).eval(); x=torch.randn(2,3,4,5,8); p=torch.randperm(5)
        torch.testing.assert_close(block(x)[:,:,:,p],block(x[:,:,:,p]),rtol=1e-5,atol=1e-5)

    def test_readout_aligns_each_experts_own_regions(self):
        a=torch.tensor([[[[[1.,0.],[0.,1.]]],[[[0.,1.],[1.,0.]]]]])
        regions=torch.tensor([[[[[10.],[20.]]],[[[30.],[40.]]]]])
        actual=restore_regions(a,regions).mean(1).flatten()
        torch.testing.assert_close(actual,torch.tensor([25.,25.]))

    def test_memberships_are_input_conditioned_and_receive_gradients(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg); data=batch()
        out=model(data); changed=copy.deepcopy(data)
        changed['x_f_obs'][...,0,0]+=10*changed['m_f'][...,0,0]
        changed['m_f'][...,1,1]=1-changed['m_f'][...,1,1]
        other=model(changed)
        for name in ['mid','coarse']:
            a=out['region_assignments'][name]
            self.assertTrue((a>0).all())  # No fixed support-radius mask.
            self.assertGreater(float((a-other['region_assignments'][name]).abs().max()),0)
            self.assertGreater(float((a[:,0]-a[:,1]).abs().max()),0)
        loss,_=compute_main_stage_loss(out,data,cfg);loss.backward()
        for aggregation in model.main_branch.aggregation.values():
            for expert in aggregation.experts:
                self.assertGreater(float(expert.slots.grad.abs().sum()),0)
                self.assertGreater(float(expert.query[0].weight.grad.abs().sum()),0)

    def test_single_or_multiple_learned_aggregators(self):
        for e in [1,2,3,4,6,8]:
            cfg=config();cfg['model']['dual_moe']['aggregation_experts']=e
            model=DualBranchSTImputer.from_config(cfg);data=batch();out=model(data)
            loss,_=compute_main_stage_loss(out,data,cfg);loss.backward()
            acc=DualMoEMetricAccumulator();acc.update(out,data);logs=acc.compute()
            self.assertIn(f'aggregation_mid_observed_e{e-1}_mean',logs)
            self.assertNotIn(f'aggregation_mid_observed_e{e}_mean',logs)
            self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_uniform_router_remains_uniform_after_loading_static_logits(self):
        router=PointRouter(8,3,'uniform')
        with torch.no_grad():router.logits[:,0]=10
        gates=router(torch.randn(2,8,3,4,5))
        torch.testing.assert_close(gates,torch.full_like(gates,1/3))

    def test_partition_regularizer_is_optional_and_safe_without_observations(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg);data=batch();out=model(data)
        cfg['loss']['dual_moe_partition_weight']=0
        base,_=compute_main_stage_loss(out,data,cfg)
        cfg['loss']['dual_moe_partition_weight']=.01
        regularized,_=compute_main_stage_loss(out,data,cfg)
        torch.testing.assert_close(regularized,base+.01*out['partition_loss'])
        data['m_f'].zero_();data['x_f_obs'].zero_()
        self.assertEqual(float(model(data)['partition_loss']),0.)

    @unittest.skipUnless(os.environ.get('DUAL_MOE_CUDA_TEST')=='1' and torch.cuda.is_available(),'Opt-in CUDA smoke')
    def test_cuda_amp_real_grid_shapes(self):
        for channels,t,h,w in [(2,12,32,32),(2,12,24,12),(1,7,32,32)]:
            cfg=config(channels);cfg['train']['amp']=True
            model=DualBranchSTImputer.from_config(cfg).cuda();optim=build_optimizer(model,cfg)
            data={k:v.cuda() for k,v in batch(channels,t,h,w,n=1).items()}
            result=train_one_epoch(model,[data],optim,torch.device('cuda'),cfg,1)
            self.assertTrue(all(math.isfinite(v) for v in result.values()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    @unittest.skipUnless(os.environ.get('DUAL_MOE_REAL_DATA_TEST')=='1' and torch.cuda.is_available(),'Opt-in local real-data smoke')
    def test_real_npz_and_csv_train_val_best_reload_test(self):
        import numpy as np
        for folder,prefix,channels in [('TaxiBJ','taxibj',2),('BikeNYC','bikenyc',2),('CHAP/beijing','chap_beijing',1)]:
            # Explicit opt-in test: read first actual window of each original
            # split, never substitute val/test windows for training windows.
            values={}
            for split in ['train','val','test']:
                with np.load(ROOT/f'data/{folder}/{prefix}_{split}.npz',allow_pickle=False) as f:
                    key='x_f_gt' if 'x_f_gt' in f else 'x_f'
                    values[split]=torch.from_numpy(f[key][:1].copy()).float()
            for pattern in ['fixed','random']:
                with self.subTest(dataset=folder,pattern=pattern):
                    cfg=load_config(ROOT/'configs/presets/default.json')
                    cfg['model']['c_in']=channels;cfg['train']['amp']=True
                    model=DualBranchSTImputer.from_config(cfg).cuda();optim=build_optimizer(model,cfg)
                    splits={}
                    for split,x in values.items():
                        csv=ROOT/f'data/{folder}/{pattern}_mask/0.4/{split}.csv'
                        a=np.loadtxt(csv,delimiter=',',max_rows=1,ndmin=2)
                        _,_,t,h,w=x.shape
                        m=torch.from_numpy(a.copy()).float()
                        m=m.reshape(1,1,1 if m.numel()==h*w else t,h,w).expand(1,1,t,h,w)
                        splits[split]={k:v.cuda() for k,v in {'x_f_gt':x,'x_f_obs':x*m,'m_f':m}.items()}
                    train=train_one_epoch(model,[splits['train']],optim,torch.device('cuda'),cfg,1)
                    val=evaluate(model,[splits['val']],torch.device('cuda'),cfg)
                    with tempfile.TemporaryDirectory() as tmp:
                        ckpt=Path(tmp)/'best.pt';save_checkpoint(ckpt,model,optim,1,val,cfg)
                        load_checkpoint(ckpt,model,map_location='cuda')
                        test=evaluate(model,[splits['test']],torch.device('cuda'),cfg)
                    self.assertTrue(all(math.isfinite(v) for logs in [train,val,test] for v in logs.values()))


if __name__=='__main__':
    unittest.main()
