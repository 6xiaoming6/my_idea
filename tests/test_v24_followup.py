from __future__ import annotations
import copy,json,sys
from pathlib import Path
import unittest
import numpy as np
from stmoe_imputer.data.diverse_masks import DiverseMaskSchedule
from stmoe_imputer.models import DualBranchSTImputer
from test_v24_coe import compact_config,make_batch
ROOT=Path(__file__).resolve().parents[1]

class FollowupTests(unittest.TestCase):
 def test_static_diverse_schedule_does_not_change_between_epochs(self):
  cfg={'families':['random_point','temporal_gap','spatial_region'],'rates':[.4],'seed':7,'resample_each_epoch':False}
  s=DiverseMaskSchedule(13,cfg);a=[s.sample(i,(5,3,5))[0] for i in range(13)];s.set_epoch(2);b=[s.sample(i,(5,3,5))[0] for i in range(13)]
  for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
  self.assertFalse(s.resample_each_epoch)
 def test_dynamic_schedule_remains_distinct(self):
  s=DiverseMaskSchedule(2452,{'families':['random_point','temporal_gap'],'rates':[.4],'seed':7});a=[s.sample(i,(5,3,5))[0] for i in range(40)];s.set_epoch(2);b=[s.sample(i,(5,3,5))[0] for i in range(40)];self.assertTrue(any(not np.array_equal(x,y) for x,y in zip(a,b)))
 def test_fixed_path_and_f_shapes(self):
  base=compact_config(num_steps=4,expert_pool=['T','S','TD','SD','TA','ST'],routing_mode='fixed',fixed_path=['TA','ST','S','TA'])
  model=DualBranchSTImputer.from_config(base);out=model(make_batch());self.assertEqual(out['coe']['paths'].shape[-1],4)
  f=compact_config(num_steps=2,expert_pool=['T','S','ST'],routing_mode='hard');model=DualBranchSTImputer.from_config(f);out=model(make_batch());self.assertEqual(out['coe']['route_logits'].shape[-1],3);self.assertEqual(out['coe']['route_logits'].shape[1],2)
 def test_router_input_noise_only_affects_training(self):
  cfg=compact_config(num_steps=4,expert_pool=['T','S','TD','SD','TA','ST'])
  cfg['model']['coe'].update(router_input_noise_std=.1,router_input_noise_steps=2)
  model=DualBranchSTImputer.from_config(cfg);batch=make_batch()
  model.eval()
  with __import__('torch').no_grad():
   a=model(batch)['coe']['route_logits'];b=model(batch)['coe']['route_logits']
  __import__('torch').testing.assert_close(a,b,atol=0,rtol=0)
  model.train()
  with __import__('torch').no_grad():
   a=model(batch)['coe']['route_logits'];b=model(batch)['coe']['route_logits']
  self.assertFalse(__import__('torch').equal(a[:,:2],b[:,:2]))
  # Noise injected into early router inputs can also change the later state;
  # only the early-router stochasticity is part of this unit-level contract.
 def test_followup_plan_has_seven_matched_jobs(self):
  sys.path.insert(0,str(ROOT/'scripts/v24'));from run_experiments import policy_plan,validate_plan
  plan,*_=policy_plan(ROOT/'configs/v24/followup_experiments.json','followup');validate_plan(plan)
  names=['route20_base','route20_warmup','route20_grouped','route20_previous','route20_noise','route20_fixed','route20_small']
  self.assertEqual([r['variant'] for r in plan['runs']],names)
  configs={r['variant']:r['config'] for r in plan['runs']}
  baseline=configs['route20_base']
  self.assertTrue(plan['comparison_policy']['same_dataset_and_masks_across_variants'])
  self.assertTrue(plan['comparison_policy']['same_evaluation_masks_across_variants'])
  for r in plan['runs']:
   c=r['config'];self.assertEqual(c['train']['epochs'],20)
   self.assertEqual(c['train']['scheduler']['total_epochs'],20)
   self.assertEqual(c['data'],baseline['data'])
   self.assertIn('mixed9',r['protocol'])
   self.assertEqual(c['data']['train_mask_diversity']['families'],c['data']['eval_mask_diversity']['families'])
   self.assertEqual(len(c['data']['eval_mask_diversity']['families']),9)
  self.assertEqual(configs['route20_warmup']['model']['coe']['routing_warmup_epochs'],3)
  self.assertEqual(configs['route20_warmup']['model']['coe']['routing_transition_epochs'],3)
  self.assertTrue(configs['route20_previous']['model']['coe']['previous_expert_context'])
  self.assertEqual(configs['route20_noise']['model']['coe']['router_input_noise_steps'],2)
  self.assertAlmostEqual(configs['route20_noise']['model']['coe']['router_input_noise_std'],0.1)
  self.assertEqual(configs['route20_fixed']['model']['coe']['routing_mode'],'fixed')
  self.assertEqual(configs['route20_small']['model']['coe']['expert_pool'],['T','S','ST'])
  self.assertEqual(configs['route20_small']['model']['coe']['num_steps'],2)
 def test_next_plan_keeps_noise_paired_and_forces_only_step_two(self):
  sys.path.insert(0,str(ROOT/'scripts/v24'));from run_experiments import policy_plan,validate_plan
  plan,*_=policy_plan(ROOT/'configs/v24/followup_next_experiments.json','followup_next');validate_plan(plan)
  self.assertEqual([r['variant'] for r in plan['runs']],['route20_next_noise','route20_next_warmup','route20_next_warmup_fixed2'])
  for run in plan['runs']:
   coe=run['config']['model']['coe']; self.assertEqual(coe['num_steps'],4)
   self.assertEqual(coe['expert_pool'],['T','S','TD','SD','TA','ST'])
   self.assertEqual(run['config']['loss']['lambda_coe_balance'],0.01)
   self.assertFalse(run['config']['train']['save_best_checkpoint'])
  noise=plan['runs'][0]['config']['model']['coe']
  self.assertEqual(noise['router_input_noise_steps'],2);self.assertAlmostEqual(noise['router_input_noise_std'],0.1)
  fixed=plan['runs'][2]['config']['model']['coe']
  self.assertEqual(fixed['fixed_expert_steps'],[None,'TA',None,None])
if __name__=='__main__':unittest.main()
