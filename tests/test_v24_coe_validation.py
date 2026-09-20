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

class CoEValidationTests(unittest.TestCase):
    def test_final_configs_match_declared_interventions_and_backpropagate(self):
        plan, *_ = policy_plan(ROOT / 'configs/v24/coe_validation_experiments.json', 'coe_validation')
        validate_plan(plan)
        configs = {r['variant']: r['config'] for r in plan['runs']}
        self.assertEqual(len(configs), 6)
        base = configs['coe_main']
        expected = {
            'coe_main': ({}, {}),
            'coe_initial_router': ({'router_state': 'initial'}, {}),
            'coe_initial_expert': ({'expert_state': 'initial'}, {}),
            'coe_fixed_chain': ({'routing_mode': 'fixed', 'fixed_path': ['TA','ST','S','TA'], 'routing_warmup_epochs': 0, 'routing_transition_epochs': 0}, {'lambda_coe_balance': 0.0}),
            'coe_no_balance': ({}, {'lambda_coe_balance': 0.0}),
            'coe_original_mask': ({}, {}),
        }
        for name, cfg in configs.items():
            with self.subTest(name=name):
                want = copy.deepcopy(base)
                want['model']['coe'].update(expected[name][0])
                want['loss'].update(expected[name][1])
                for key in ['model', 'loss', 'data', 'train']:
                    if name != 'coe_original_mask' or key != 'data':
                        self.assertEqual(cfg[key], want[key])
                self.assertEqual(cfg['model']['coe']['num_steps'], 4)
                self.assertEqual(len(cfg['model']['coe']['expert_pool']), 6)
                self.assertEqual(cfg['train']['epochs'], 20)
                self.assertFalse(cfg['train']['save_best_checkpoint'])
                tiny = copy.deepcopy(cfg)
                tiny['model']['main']['dim'] = 8
                model = DualBranchSTImputer.from_config(tiny)
                batch = make_batch()
                for epoch in [1, 6]:
                    model.zero_grad(set_to_none=True)
                    model.main_branch.set_routing_epoch(epoch)
                    out = model(batch)
                    loss, _ = compute_main_stage_loss(out, batch, tiny)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                model.eval()
                with torch.no_grad():
                    out = model(batch)
                self.assertEqual(tuple(out['coe']['paths'].shape), (2, 4))
        original = configs['coe_original_mask']
        self.assertIsNone(original['data']['train_mask_diversity'])
        self.assertIsNone(original['data']['eval_mask_diversity'])
        self.assertTrue(original['data']['mask']['train_csv'].endswith('random_point/0.4/train.csv'))
