"""One-epoch CPU train/validation/test and gloo DDP smoke for v25."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from stmoe_imputer.config import deep_update

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v25_ablation_runner', ROOT / 'scripts/v25/run_ablations.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _config(output_dir: str, batch_size: int) -> dict:
    cfg = json.loads((ROOT / 'configs/v24/smoke.json').read_text())
    patch = json.loads((ROOT / 'configs/v25/A5_full_ras_coe.json').read_text())
    cfg = deep_update(cfg, patch)
    cfg['output_dir'] = output_dir
    cfg['device'] = 'cpu'
    cfg['model']['architecture'] = 'v25_ras_coe'
    cfg['model']['main'].update(dim=8, max_t=3, h=4, w=4, use_multiscale=False)
    cfg['model']['coe'].update(
        num_steps=2, expert_pool=['T', 'S', 'TD', 'SD', 'TA', 'ST', 'TL', 'SL'],
        fixed_expert_steps=[None, None], routing_mode='hard', top_k=2,
        pair_mode='partner_residual', partner_fusion='corrected',
        partner_aux_head_only=True, expert_sharing='shared',
        router_state='dynamic', expert_state='dynamic', acceptance='none',
        repair_accept_aux_head_only=True, repair_accept_init_prob=0.99,
        routing_warmup_epochs=0, routing_transition_epochs=0,
    )
    cfg['data']['synthetic'].update(num_train=8, num_val=4, t=3, h=4, w=4)
    cfg['data']['batch_size'] = batch_size
    cfg['train'].update(
        epochs=1, val_epoch=1, amp=False, save_best_checkpoint=True,
        best_checkpoint_name='best.pth', early_stopping={'enabled': False},
        partner_probe={'interval_batches': 2, 'weight': 0.1, 'min_scale': 0.5,
                       'grad_diagnostic_interval_batches': 0},
    )
    cfg['loss'].update(lambda_coe_mid=0., lambda_coe_balance=0.01,
                       balance_importance='candidate', lambda_coe_accept=0.1)
    return cfg


class V25TrainingSmokeTests(unittest.TestCase):
    def _run(self, world_size: int) -> None:
        with tempfile.TemporaryDirectory(prefix='v25_train_smoke_') as tmp:
            root = Path(tmp)
            cfg = _config(str(root / 'outputs'), 2 * world_size)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(cfg))
            receipt_path = root / 'receipt.json'
            if world_size == 2:
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                           '--nproc_per_node=2', 'scripts/v25/train.py']
            else:
                command = [sys.executable, '-u', 'scripts/v25/train.py']
            command.extend(['-c', str(config_path), '--synthetic', '--no_plot',
                            '--result-file', str(receipt_path), '--name', 'smoke_v25'])
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1')
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-3000:])
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt['world_size'], world_size)
            self.assertEqual(receipt['completed_epochs'], 1)
            run_dir = Path(receipt['run_dir'])
            self.assertTrue((run_dir / 'checkpoints/best.pth').is_file())
            last_path = run_dir / 'checkpoints/last.pth'
            self.assertEqual(receipt['last_checkpoint'], str(last_path))
            last = torch.load(last_path, map_location='cpu', weights_only=True)
            self.assertEqual(last['epoch'], 1)
            self.assertTrue(last['optimizer']['state'])
            self.assertEqual(len(last['rng_states']), world_size)
            self.assertEqual(last['training_state']['next_epoch'], 2)
            self.assertTrue(runner._receipt_valid(receipt_path, cfg))
            self.assertIn('coe_step2_accepted_harm_rate', receipt['test'])
            self.assertIn('coe_all_steps_monotonic_sample_rate', receipt['test'])
            self.assertIn('coe_accept_f1', receipt['test'])

    def test_single_process(self):
        self._run(1)

    def test_two_process_gloo_ddp(self):
        self._run(2)


if __name__ == '__main__':
    unittest.main()
