from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from stmoe_imputer.engine import build_grad_scaler, build_optimizer, train_one_epoch
from stmoe_imputer.models import DualBranchSTImputer
from test_v24_coe import compact_config, make_batch


class PersistentScalerTests(unittest.TestCase):
    def exercise(self, explicit):
        cfg = compact_config()
        model = DualBranchSTImputer.from_config(cfg)
        optimizer = build_optimizer(model, cfg)
        # CPU GradScaler keeps a real growth tracker without requiring a GPU.
        scaler = torch.amp.GradScaler('cpu', init_scale=2., growth_interval=2)
        kwargs = {'scaler': scaler} if explicit else {}
        with patch('stmoe_imputer.engine.build_grad_scaler', return_value=scaler) as constructor:
            first = train_one_epoch(model, [make_batch()], optimizer, torch.device('cpu'), cfg, 1, **kwargs)
            second = train_one_epoch(model, [make_batch()], optimizer, torch.device('cpu'), cfg, 2, **kwargs)
        self.assertEqual(constructor.call_count, 0 if explicit else 1)
        self.assertEqual(first['train_amp_scale_start'], 2.)
        self.assertEqual(first['train_amp_scale_end'], 2.)
        self.assertEqual(second['train_amp_scale_start'], 2.)
        self.assertEqual(second['train_amp_scale_end'], 4.)
        self.assertEqual(second['train_optimizer_steps'], 1.)
        self.assertEqual(second['train_skipped_amp_steps'], 0.)

    def test_explicit_scaler_keeps_growth_history_across_epochs(self):
        self.exercise(True)

    def test_legacy_caller_keeps_scaler_on_its_optimizer(self):
        self.exercise(False)

    def test_cpu_training_default_does_not_enable_amp_scaling(self):
        cfg = compact_config(); cfg['train']['amp'] = True
        self.assertFalse(build_grad_scaler(torch.device('cpu'), cfg).is_enabled())


if __name__ == '__main__':
    unittest.main()
