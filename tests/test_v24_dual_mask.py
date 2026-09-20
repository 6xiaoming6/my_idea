import copy
import sys
import unittest
from pathlib import Path
import torch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_main_stage_loss
from test_v24_coe import make_batch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/v24'))
from run_experiments import policy_plan, validate_plan

class DualMaskTests(unittest.TestCase):
    def test_generated_configs_construct_and_train(self):
        plan, *_ = policy_plan(ROOT / 'configs/v24/dual_mask_experiments.json', 'coe_dual_mask')
        validate_plan(plan)
        self.assertEqual(len(plan['runs']), 16)
        for run in plan['runs']:
            with self.subTest(run=run['name']):
                cfg = run['config']
                coe = cfg['model']['coe']
                self.assertEqual(coe['num_steps'], 4)
                self.assertEqual(coe['expert_pool'], ['T', 'S', 'TD', 'SD', 'TA', 'ST'])
                model = DualBranchSTImputer.from_config(cfg)
                batch = make_batch()
                for epoch in (1, 6):
                    model.main_branch.set_routing_epoch(epoch)
                    model.zero_grad(set_to_none=True)
                    out = model(batch)
                    loss, _ = compute_main_stage_loss(out, batch, cfg)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                model.eval()
                with torch.no_grad():
                    out = model(batch)
                self.assertEqual(tuple(out['coe']['paths'].shape), (2, 4))
                if coe['routing_mode'] == 'fixed':
                    expected = [coe['expert_pool'].index(e) for e in coe['fixed_path']]
                    self.assertEqual(out['coe']['paths'].tolist(), [expected, expected])
                original = run['protocol'].startswith('original')
                self.assertEqual(cfg['experiment_plan']['evaluation_mask_source'], 'protocol' if original else 'diverse_fixed_per_split')
                self.assertEqual(cfg['data'].get('eval_mask_diversity') is None, original)
                self.assertFalse(cfg['train']['save_best_checkpoint'])
        broken = copy.deepcopy(plan)
        broken['runs'][0]['config']['model']['coe']['num_steps'] = 2
        with self.assertRaisesRegex(ValueError, 'four steps'):
            validate_plan(broken)
