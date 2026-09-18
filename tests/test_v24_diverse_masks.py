from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from stmoe_imputer.data.diverse_masks import FAMILIES, DiverseMaskSchedule, make_diverse_mask
from stmoe_imputer.data import build_datasets, build_loader, build_test_dataset
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.engine import train_one_epoch, build_optimizer
from test_v24_coe import compact_config

ROOT=Path(__file__).resolve().parents[1]

class DiverseMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_families_exact_budget_binary_and_reproducible(self):
        for shape in [(12,32,32),(1,2,5),(3,1,4)]:
            for family in FAMILIES:
                for rate in [.2,.4,.6]:
                    a=make_diverse_mask(shape,rate,family,np.random.default_rng(123))
                    b=make_diverse_mask(shape,rate,family,np.random.default_rng(123))
                    np.testing.assert_array_equal(a,b)
                    self.assertTrue(np.isin(a,[0,1]).all())
                    self.assertEqual(int((a==0).sum()),round(np.prod(shape)*rate))

    def test_balanced_families_and_rates_change_across_epochs(self):
        schedule=DiverseMaskSchedule(2452,{'rates':[.2,.4,.6],'seed':7})
        for epoch in [1,2,20]:
            schedule.set_epoch(epoch);counts=np.bincount(schedule.assignments,minlength=27)
            self.assertLessEqual(counts.max()-counts.min(),1)
        schedule.set_epoch(1);a=schedule.sample(0,(12,32,32))[0]
        schedule.set_epoch(2);b=schedule.sample(0,(12,32,32))[0]
        self.assertFalse(np.array_equal(a,b))
        schedule.set_epoch(1);np.testing.assert_array_equal(a,schedule.sample(0,(12,32,32))[0])

    def test_geometry_really_changes_temporal_vs_spatial_support(self):
        # Exact node budget: all timestamps share the same observed nodes.
        node=make_diverse_mask((10,10,10),.4,'node_outage',np.random.default_rng(1))
        self.assertTrue(np.all(node==node[0]))
        gap=make_diverse_mask((10,10,10),.4,'temporal_gap',np.random.default_rng(1))
        self.assertTrue(np.isin(gap.mean((1,2)),[0,1]).all())
        missing=np.flatnonzero(gap.mean((1,2))==0)
        self.assertTrue(np.all(np.diff(missing)==1))
        random=make_diverse_mask((10,10,10),.4,'random_point',np.random.default_rng(1))
        self.assertGreater(np.mean(random[1:]!=random[:-1]),.2)

    def make_data(self,directory):
        cfg=compact_config(num_steps=4,expert_pool=['T','S','TD','SD','TA','ST'])
        cfg['data']['train_mask_diversity']={'seed':7,'rates':[.4]}
        cfg['data']['loader_seed']=7;cfg['data']['batch_size']=3;cfg['data']['num_workers']=0
        values=np.random.default_rng(9).normal(size=(9,2,5,3,5)).astype('float32')
        sources={}
        for split in ['train','val','test']:
            p=directory/f'{split}.npz';np.savez(p,x_f_gt=values);sources[split]=str(p)
            mask=np.ones((9,75));mask[:,::3]=0
            csv=directory/f'{split}.csv';np.savetxt(csv,mask,delimiter=',',fmt='%d');cfg['data']['mask'][f'{split}_csv']=str(csv)
        return cfg,sources

    def test_only_training_masks_change_and_no_target_or_family_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,paths=self.make_data(Path(temp));train,val=build_datasets(cfg,paths['train'],paths['val'])
            test=build_test_dataset(cfg,paths['test']);self.assertIsNone(val.diverse_masks);self.assertIsNone(test.diverse_masks)
            before=val[0]['m_f'].clone();a=train[0];train.arrays[train.x_key][:]+=100
            b=train[0];torch.testing.assert_close(a['m_f'],b['m_f'],atol=0,rtol=0)
            self.assertTrue((b['x_f_obs'][~b['m_f'].bool()]==0).all())
            train.set_epoch(2);self.assertFalse(torch.equal(a['m_f'],train[0]['m_f']))
            torch.testing.assert_close(before,val[0]['m_f'],atol=0,rtol=0)
            batch=next(iter(build_loader(train,cfg,False)));model=DualBranchSTImputer.from_config(cfg).eval()
            with torch.no_grad():
                first=model(batch)['coe']['route_logits'];batch['mask_family'].fill_(999)
                second=model(batch)['coe']['route_logits'];torch.testing.assert_close(first,second,atol=0,rtol=0)

    def test_evaluation_mask_intervention_applies_to_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,paths=self.make_data(Path(temp))
            cfg['data']['eval_mask_diversity']={'families':list(FAMILIES),'rates':[.4],'seed':7}
            train,val=build_datasets(cfg,paths['train'],paths['val'])
            test=build_test_dataset(cfg,paths['test'])
            self.assertIsNotNone(train.diverse_masks)
            self.assertIsNotNone(val.diverse_masks)
            self.assertIsNotNone(test.diverse_masks)
            self.assertIsNone(val.loaded_masks)
            self.assertIsNone(test.loaded_masks)
            self.assertEqual(set(int(val[i]['mask_family']) for i in range(len(val))),set(range(len(FAMILIES))))
            self.assertEqual(set(int(test[i]['mask_family']) for i in range(len(test))),set(range(len(FAMILIES))))
            before=val[0]['m_f'].clone(); val.set_epoch(2)
            torch.testing.assert_close(before,val[0]['m_f'],atol=0,rtol=0)

    def test_worker_independent_and_train_engine_sets_epoch_and_logs_family(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,paths=self.make_data(Path(temp));train,_=build_datasets(cfg,paths['train'],paths['val'])
            a=torch.cat([b['m_f'] for b in build_loader(train,cfg,False)])
            worker_cfg=copy.deepcopy(cfg);worker_cfg['data']['num_workers']=2
            b=torch.cat([batch['m_f'] for batch in build_loader(train,worker_cfg,False)])
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            model=DualBranchSTImputer.from_config(cfg);opt=build_optimizer(model,cfg)
            logs=train_one_epoch(model,build_loader(train,cfg,True),opt,torch.device('cpu'),cfg,2)
            self.assertEqual(train.diverse_masks.epoch,2)
            for family in FAMILIES:self.assertEqual(logs[f'coe_condition_family_{family}_sample_count'],1)

    def test_d_matches_a_except_training_mask_policy_and_pairing_is_honest(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/v24'))
        from run_experiments import policy_plan,validate_plan
        plan,*_=policy_plan(ROOT/'configs/v24/abcd_experiments.json','ad');validate_plan(plan)
        a,d=[copy.deepcopy(r['config']) for r in plan['runs']]
        self.assertEqual(plan['comparison_policy']['same_evaluation_masks_across_variants'],True)
        self.assertEqual(plan['comparison_policy']['same_dataset_and_masks_across_variants'],False)
        d['data'].pop('train_mask_diversity')
        for c in (a,d):c.pop('experiment_plan')
        self.assertEqual(a,d)

    def test_invalid_config_fails(self):
        for config in [{'families':[]},{'families':['x']},{'rates':[0]},{'rates':[float('nan')]},{'seed':-1}]:
            with self.assertRaises(ValueError):DiverseMaskSchedule(10,config)

if __name__=='__main__':unittest.main()
