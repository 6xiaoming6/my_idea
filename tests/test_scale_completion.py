import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import torch

from test_dual_moe import batch,config,ROOT
from stmoe_imputer.config import load_config,deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.dual_moe import AnchoredScaleMoEBackbone,GridObservationAggregation,aligned_readout
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.engine import train_one_epoch,evaluate,build_optimizer
from stmoe_imputer.routing_metrics import DualMoEMetricAccumulator
from stmoe_imputer.utils.checkpoint import save_checkpoint,load_checkpoint


def optimized(kind='learned_regions',channels=2):
    cfg=config(channels)
    cfg['model']['dual_moe'].update(design='anchored_scale_moe',aggregation_mode='uniform',
                                   aggregation_kind=kind,residual_bound=1.,grid_strides=[4,8])
    cfg['loss']['dual_moe_expert_weight']=.01
    if kind=='regular_grid':cfg['loss']['dual_moe_partition_weight']=0
    return cfg


class ScaleCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_starts_at_fine_and_corrections_stay_bounded(self):
        for kind in ('learned_regions','regular_grid'):
            cfg=optimized(kind);m=DualBranchSTImputer.from_config(cfg);data=batch()
            self.assertIsInstance(m.main_branch,AnchoredScaleMoEBackbone)
            out=m(data)
            torch.testing.assert_close(out['x_hat_final'],out['base_prediction'],rtol=0,atol=0)
            with torch.no_grad():
                m.main_branch.scale_experts['mid'].head.bias.fill_(100.)
                m.main_branch.scale_experts['coarse'].head.bias.fill_(-100.)
            out=m(data);scale=out['normalization']['scale']
            self.assertLessEqual(float(out['normalized_correction'].abs().max()),1.)
            self.assertTrue(((out['x_hat_final']-out['base_prediction']).abs()<=scale+1e-5).all())
            combined=sum(out['completion_gates'][:,i:i+1]*out['scale_predictions'][s] for i,s in enumerate(('fine','mid','coarse')))
            torch.testing.assert_close(out['x_hat_final'],combined)

    def test_grid_exact_observed_means_edge_bins_and_readout(self):
        agg=GridObservationAggregation(2)
        x=torch.arange(15.).reshape(1,1,1,3,5);mask=torch.ones_like(x);mask[...,0,0]=0
        out=agg(x,mask,x*mask)
        self.assertIsNone(out['assignment'])
        self.assertEqual(tuple(out['features'].shape),(1,1,1,6,1))
        expected=torch.tensor([4.,5.,6.5,10.5,12.5,14.])
        torch.testing.assert_close(out['features'].flatten(),expected)
        torch.testing.assert_close(out['mass'].sum(),mask.sum())
        torch.testing.assert_close(out['effective_count'],out['mass'])
        restored=aligned_readout(out,out['features']).reshape(3,5)
        torch.testing.assert_close(restored[0],torch.tensor([4.,4.,5.,5.,6.5]))
        torch.testing.assert_close(restored[2],torch.tensor([10.5,10.5,12.5,12.5,14.]))
        mask.zero_();out=agg(x,mask,x*mask)
        self.assertEqual(float(out['features'].abs().max()),0.)
        self.assertEqual(float(out['support'].abs().max()),0.)

    def test_shapes_gradients_empty_masks_and_no_ground_truth_leak(self):
        for kind in ('learned_regions','regular_grid'):
            for c,t,h,w in [(2,12,32,32),(2,12,24,12),(1,7,32,32),(1,2,7,11),(1,1,1,1)]:
                cfg=optimized(kind,c);m=DualBranchSTImputer.from_config(cfg);opt=build_optimizer(m,cfg)
                data=batch(c,t,h,w,n=1)
                for _ in range(2):
                    opt.zero_grad();out=m(data);loss,_=compute_main_stage_loss(out,data,cfg);loss.backward();opt.step()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
                self.assertEqual(out['x_hat_final'].shape,data['x_f_gt'].shape)
                changed=copy.deepcopy(data);changed['x_f_gt'].fill_(float('nan'))
                changed['x_f_obs']=torch.where(data['m_f'].bool(),data['x_f_obs'],torch.full_like(data['x_f_obs'],float('nan')))
                torch.testing.assert_close(m(data)['x_hat_final'],m(changed)['x_hat_final'])
            cfg=optimized(kind);m=DualBranchSTImputer.from_config(cfg)
            for value in [0.,1.]:
                data=batch();data['m_f'].fill_(value);data['x_f_obs']=data['x_f_gt']*value
                out=m(data);loss,_=compute_main_stage_loss(out,data,cfg)
                self.assertTrue(torch.isfinite(loss))
                if value==1:self.assertEqual(float(loss),0.)
                acc=DualMoEMetricAccumulator();acc.update(out,data)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in acc.compute().values()))

    def test_common_initial_parameters_and_new_legacy_dispatch(self):
        torch.manual_seed(42);learned=DualBranchSTImputer.from_config(optimized())
        torch.manual_seed(42);grid=DualBranchSTImputer.from_config(optimized('regular_grid'))
        a,b=learned.state_dict(),grid.state_dict()
        for k in b:
            torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        old=DualBranchSTImputer.from_config(config())
        self.assertNotIsInstance(old.main_branch,AnchoredScaleMoEBackbone)
        for patch in [{'design':'bad'},{'residual_bound':0},{'grid_strides':[8,4]},{'aggregation_mode':'learned'}]:
            cfg=optimized();cfg['model']['dual_moe'].update(patch)
            with self.assertRaises(ValueError):DualBranchSTImputer.from_config(cfg)

    def test_metric_partition_invariance_and_checkpoint(self):
        for kind in ('learned_regions','regular_grid'):
            cfg=optimized(kind);m=DualBranchSTImputer.from_config(cfg).eval();data=batch(n=3)
            whole=DualMoEMetricAccumulator();whole.update(m(data),data)
            pieces=DualMoEMetricAccumulator()
            for lo,hi in [(0,2),(2,3)]:
                part={k:v[lo:hi] for k,v in data.items()};pieces.update(m(part),part)
            for k,v in whole.compute().items():self.assertAlmostEqual(v,pieces.compute()[k],delta=1e-4*max(1,abs(v)))
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'best.pt';save_checkpoint(p,m,None,1,{},cfg)
                clone=DualBranchSTImputer.from_config(cfg);load_checkpoint(p,clone,map_location='cpu')
                torch.testing.assert_close(m(data)['x_hat_final'],clone(data)['x_hat_final'])

    def test_existing_training_entry_best_checkpoint_and_logs(self):
        for kind in ('learned_regions','regular_grid'):
            cfg=optimized(kind)
            with tempfile.TemporaryDirectory() as tmp:
                cfg['output_dir']=tmp;path=Path(tmp)/'input.json';path.write_text(json.dumps(cfg))
                result=subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'-c',str(path),'--synthetic','--no_plot','--quiet','--name','smoke_anchored'],cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout[-1000:]+result.stderr[-1000:])
                checkpoints=list(Path(tmp).rglob('best.pt'));self.assertEqual(len(checkpoints),1)
                run=checkpoints[0].parent.parent
                entries=[json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()]
                self.assertEqual(sum('epoch'in e for e in entries),2)
                self.assertEqual(sum(e.get('stage')=='test' for e in entries),1)
                self.assertIn('bounded correction', (run/'logs/train.log').read_text())

    @unittest.skipUnless(os.environ.get('SCALE_COMPLETION_REAL_TEST')=='1' and torch.cuda.is_available(),'Opt-in real-data CUDA check')
    def test_three_real_datasets_both_aggregations_train_val_reload_test(self):
        import numpy as np
        for folder,prefix,c in [('TaxiBJ','taxibj',2),('BikeNYC','bikenyc',2),('CHAP/beijing','chap_beijing',1)]:
            values={}
            for split in ('train','val','test'):
                with np.load(ROOT/f'data/{folder}/{prefix}_{split}.npz',allow_pickle=False) as f:values[split]=torch.from_numpy(f['x_f_gt'][:1].copy()).float()
            for kind in ('learned_regions','regular_grid'):
                for pattern in ('fixed','random'):
                    preset='scale_completion_grid.json' if kind=='regular_grid' else 'scale_completion.json'
                    cfg=deep_update(load_config(ROOT/'configs/presets/default.json'),load_config(ROOT/'configs/presets'/preset));cfg['model']['c_in']=c
                    m=DualBranchSTImputer.from_config(cfg).cuda();opt=build_optimizer(m,cfg);splits={}
                    for split,x in values.items():
                        a=np.loadtxt(ROOT/f'data/{folder}/{pattern}_mask/0.4/{split}.csv',delimiter=',',max_rows=1,ndmin=2)
                        _,_,t,h,w=x.shape;mask=torch.from_numpy(a.copy()).float().reshape(1,1,1 if a.size==h*w else t,h,w).expand(1,1,t,h,w)
                        splits[split]={k:v.cuda() for k,v in {'x_f_gt':x,'x_f_obs':x*mask,'m_f':mask}.items()}
                    for epoch in (1,2):train=train_one_epoch(m,[splits['train']],opt,torch.device('cuda'),cfg,epoch)
                    val=evaluate(m,[splits['val']],torch.device('cuda'),cfg)
                    with tempfile.TemporaryDirectory() as tmp:
                        p=Path(tmp)/'best.pt';save_checkpoint(p,m,opt,2,val,cfg);load_checkpoint(p,m,map_location='cuda')
                        test=evaluate(m,[splits['test']],torch.device('cuda'),cfg)
                    for logs in (train,val,test):self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in logs.values()))


if __name__=='__main__':unittest.main()
