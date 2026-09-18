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
  s=DiverseMaskSchedule(12,cfg);a=[s.sample(i,(5,3,5))[0] for i in range(12)];s.set_epoch(2);b=[s.sample(i,(5,3,5))[0] for i in range(12)]
  for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
  self.assertFalse(s.resample_each_epoch)
 def test_dynamic_schedule_remains_distinct(self):
  s=DiverseMaskSchedule(2452,{'families':['random_point','temporal_gap'],'rates':[.4],'seed':7});a=[s.sample(i,(5,3,5))[0] for i in range(40)];s.set_epoch(2);b=[s.sample(i,(5,3,5))[0] for i in range(40)];self.assertTrue(any(not np.array_equal(x,y) for x,y in zip(a,b)))
 def test_fixed_path_and_f_shapes(self):
  base=compact_config(num_steps=4,expert_pool=['T','S','TD','SD','TA','ST'],routing_mode='fixed',fixed_path=['TA','ST','S','TA'])
  model=DualBranchSTImputer.from_config(base);out=model(make_batch());self.assertEqual(out['coe']['paths'].shape[-1],4)
  f=compact_config(num_steps=2,expert_pool=['T','S','ST'],routing_mode='hard');model=DualBranchSTImputer.from_config(f);out=model(make_batch());self.assertEqual(out['coe']['route_logits'].shape[-1],3);self.assertEqual(out['coe']['route_logits'].shape[1],2)
 def test_followup_plan_has_three_jobs_and_expected_interventions(self):
  sys.path.insert(0,str(ROOT/'scripts/v24'));from run_experiments import policy_plan,validate_plan
  plan,*_=policy_plan(ROOT/'configs/v24/followup_experiments.json','followup');validate_plan(plan)
  self.assertEqual([x['variant'] for x in plan['runs']],['abc_d_eval_mixed','fixed4_ta_st_s_ta','abc_d_static','abc_f'])
  configs={x['variant']:x['config'] for x in plan['runs']}
  self.assertIn('eval_mask_diversity',configs['abc_d_eval_mixed']['data'])
  self.assertEqual(configs['abc_d_eval_mixed']['experiment_plan']['evaluation_mask_source'],'diverse_fixed_per_split')
  self.assertFalse(plan['comparison_policy']['same_evaluation_masks_across_variants'])
  self.assertEqual(configs['fixed4_ta_st_s_ta']['model']['coe']['routing_mode'],'fixed')
  self.assertEqual(configs['fixed4_ta_st_s_ta']['model']['coe']['fixed_path'],['TA','ST','S','TA'])
  self.assertFalse(configs['abc_d_static']['data']['train_mask_diversity']['resample_each_epoch'])
  self.assertEqual(configs['abc_f']['model']['coe']['num_steps'],2);self.assertEqual(len(configs['abc_f']['model']['coe']['expert_pool']),3)
if __name__=='__main__':unittest.main()
