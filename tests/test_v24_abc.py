from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import unittest
import torch
import numpy as np
from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.coe_router import observed_pattern_features
from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.engine import build_optimizer, build_scheduler
from stmoe_imputer.data import build_loader
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator
from test_v24_coe import compact_config, make_batch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/v24'))
from prepare_clean_taxibj import retained_windows


def candidate(stable=False):
    cfg=compact_config(num_steps=4,expert_pool=['T','S','TD','SD','TA','ST'],routing_mode='hard',
                       router_features='grouped',value_mean=[100.,120.],value_std=[120.,140.])
    cfg['loss']['lambda_coe_balance']=.01
    if stable:
        cfg=deep_update(cfg,{'model':{'coe':{'router_fp32':True,'router_init_std':.01,
            'routing_warmup_epochs':5,'routing_transition_epochs':5,'sampling_temperature_start':2.,'uniform_mix_start':.5}},
            'loss':{'lambda_coe_z':.001},'train':{'lr_router':1e-4,'router_grad_diagnostic_every':50}})
    return cfg


class ABCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_cleaning_rejects_time_gaps_and_cross_split_frames(self):
        times=np.array([0,1,2,5,6,7,8,9])*1800*10**9
        ids=np.array([[0,1,2],[1,2,3],[3,4,5],[5,6,7]])
        kept,bad=retained_windows(ids,times,{0})
        self.assertEqual(kept.tolist(),[2,3])
        self.assertEqual([r['reason'] for r in bad],['cross_split_overlap','time_discontinuity'])

    def test_shuffle_is_independent_of_model_initialization_rng(self):
        cfg=candidate();cfg['data']['loader_seed']=7
        dataset=torch.utils.data.TensorDataset(torch.arange(37))
        torch.manual_seed(7)
        a=build_loader(dataset,cfg,shuffle=True)
        first=torch.cat([batch[0] for batch in a])
        torch.randn(10000)
        b=build_loader(dataset,cfg,shuffle=True)
        second=torch.cat([batch[0] for batch in b])
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        self.assertFalse(torch.equal(first,torch.cat([batch[0] for batch in a])))

    def test_observed_features_ignore_hidden_payload_and_report_no_pairs(self):
        x=torch.randn(2,2,3,4,5);m=torch.rand_like(x)>.5;std=torch.ones(1,2,1,1,1)
        expected=observed_pattern_features(x,m,std)
        x[~m]=float('nan')
        torch.testing.assert_close(expected,observed_pattern_features(x,m,std))
        self.assertTrue(torch.equal(observed_pattern_features(x,torch.zeros_like(m),std),torch.zeros(2,18)))

    def test_full_grouped_forward_does_not_read_hidden_targets(self):
        model=DualBranchSTImputer.from_config(candidate()).eval();batch=make_batch()
        with torch.no_grad():
            a=model(batch)
            mask=batch['m_f'].bool().expand_as(batch['x_f_gt'])
            batch['x_f_gt'][~mask]=1e8;batch['x_f_obs'][~mask]=float('nan')
            b=model(batch)
        torch.testing.assert_close(a['x_hat_main'],b['x_hat_main'],atol=0,rtol=0)
        torch.testing.assert_close(a['coe']['route_logits'],b['coe']['route_logits'],atol=0,rtol=0)

    def test_schedule_soft_gradients_and_hard_eval(self):
        cfg=candidate(True);model=DualBranchSTImputer.from_config(cfg).train();batch=make_batch()
        outputs=model(batch)
        self.assertEqual(outputs['coe']['routing_mode'],'soft')
        self.assertTrue((outputs['coe']['route_weights']>=.5/6).all())
        loss,logs=compute_coe_loss(outputs,batch,cfg);loss.backward()
        self.assertGreater(logs['l_coe_z_weighted'],0)
        for expert in model.main_branch.routed_experts():
            self.assertGreater(sum(p.grad.abs().sum() for p in expert.parameters() if p.grad is not None),0)
        accumulator=CoERoutingMetricAccumulator();accumulator.update(outputs['coe'])
        self.assertIn('coe_step1_T_weight',accumulator.compute())
        self.assertNotIn('coe_path_max_fraction',accumulator.compute())
        branch=model.main_branch
        for epoch,hard,temp in [(5,0.,2.),(6,.2,1.8),(9,.8,1.2),(10,1.,1.),(11,1.,1.)]:
            branch.set_routing_epoch(epoch);a,_,c=branch.routing_schedule()
            self.assertAlmostEqual(a,hard);self.assertAlmostEqual(c,temp)
        for epoch in [1,6,11]:
            branch.set_routing_epoch(epoch);model.eval()
            with torch.no_grad(): result=model(batch)
            self.assertEqual(result['coe']['routing_mode'],'hard')
            self.assertTrue(((result['coe']['route_weights']==0)|(result['coe']['route_weights']==1)).all())
            model.train()

    def test_fp32_router_under_cpu_autocast_and_transition_backward(self):
        cfg=candidate(True);model=DualBranchSTImputer.from_config(cfg).train();model.main_branch.set_routing_epoch(7)
        batch=make_batch()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            outputs=model(batch);loss,_=compute_coe_loss(outputs,batch,cfg)
        self.assertEqual(outputs['coe']['route_logits'].dtype,torch.float32)
        self.assertEqual(outputs['coe']['route_weights'].dtype,torch.float32)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_router_group_and_80_epoch_schedule_prefix(self):
        cfg=candidate(True);cfg['train']['epochs']=20;cfg['train']['scheduler']={'type':'cosine','total_epochs':80,'eta_min':1e-6}
        model=DualBranchSTImputer.from_config(cfg);opt=build_optimizer(model,cfg)
        expected={id(p) for p in model.main_branch.routers.parameters()}
        group=next(g for g in opt.param_groups if g['name']=='router')
        self.assertEqual({id(p) for p in group['params']},expected);self.assertEqual(group['lr'],1e-4)
        self.assertFalse(expected & {id(p) for g in opt.param_groups if g['name']!='router' for p in g['params']})
        scheduler=build_scheduler(opt,cfg);self.assertEqual(scheduler.T_max,80)

    def test_z_loss_averages_rounds_and_empty_supervision(self):
        cfg=candidate(True);model=DualBranchSTImputer.from_config(cfg);batch=make_batch();outputs=model(batch)
        _,logs=compute_coe_loss(outputs,batch,cfg)
        expected=outputs['coe']['route_logits'].float().logsumexp(-1).square().mean()
        torch.testing.assert_close(logs['l_coe_z'],expected)
        batch['m_f']=torch.ones_like(batch['m_f'])
        loss,logs=compute_coe_loss(outputs,batch,cfg)
        self.assertEqual(loss.item(),0.);self.assertEqual(logs['l_coe_z_weighted'].item(),0.)

    def test_legacy_parameters_predictions_and_routes_match_original(self):
        # Frozen pre-ablation implementation: independent of runtime workspace
        # location and of subsequent edits to the current model.
        origin=Path(__file__).resolve().parent/'fixtures/v24_coe_before_router_ablation.py'
        spec=importlib.util.spec_from_file_location('stmoe_imputer.models._original_coe',origin)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cfg=compact_config(num_steps=4,expert_pool=['T','S','TD','SD','TA','ST'])
        torch.manual_seed(7);old=module.TemporalSpatialCoE.from_config(cfg)
        torch.manual_seed(7);new=TemporalSpatialCoE.from_config(cfg)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for k,v in old.state_dict().items(): torch.testing.assert_close(v,new.state_dict()[k],atol=0,rtol=0)
        batch=make_batch()
        for training in [False,True]:
            old.train(training);new.train(training)
            torch.manual_seed(123);a=old(batch['x_f_obs'],batch['m_f'])
            torch.manual_seed(123);b=new(batch['x_f_obs'],batch['m_f'])
            for key in ['route_logits','route_probs','route_weights']:
                torch.testing.assert_close(a['coe'][key],b['coe'][key],atol=0,rtol=0)
            torch.testing.assert_close(a['x_hat_main'],b['x_hat_main'],atol=0,rtol=0)

if __name__=='__main__': unittest.main()
