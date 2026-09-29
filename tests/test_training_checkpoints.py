"""Final-vs-best checkpoints and the adopted cosine training protocol."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import unittest
import torch

from stmoe_imputer.engine import build_scheduler
from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('checkpoint_train', ROOT / 'scripts/train.py')
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def schedule_config(epochs=100, endpoint=True):
    return {'train': {'epochs': epochs, 'scheduler': {
        'type': 'cosine', 'total_epochs': epochs, 'eta_min': 3e-4,
        'reach_min_at_last_epoch': endpoint}}}


def _check_cosine_uses_initial_and_final_lr_and_never_rebounds(epochs):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    scheduler = build_scheduler(optimizer, schedule_config(epochs))
    used = []
    for _ in range(epochs + 3):
        used.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    assert abs(used[0] - 1e-3) < 1e-14
    if epochs > 1:
        assert abs(used[epochs - 1] - 3e-4) < 1e-14
    assert abs(used[-1] - 3e-4) < 1e-14
    assert all(a >= b for a, b in zip(used, used[1:]))


def _check_legacy_cosine_still_uses_original_period():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cfg = schedule_config()
    del cfg['train']['scheduler']['reach_min_at_last_epoch']
    scheduler = build_scheduler(optimizer, cfg)
    assert type(scheduler) is torch.optim.lr_scheduler.CosineAnnealingLR
    assert scheduler.T_max == 100


def _check_checkpoint_restores_identical_next_optimizer_and_scheduler_step(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = schedule_config()
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler('cpu')
    x = torch.ones(2, 2)

    def step(m, opt, sched, amp):
        opt.zero_grad()
        amp.scale(m(x).square().sum()).backward()
        amp.step(opt)
        amp.update()
        sched.step()

    for _ in range(4):
        step(model, optimizer, scheduler, scaler)
    path = tmp_path / 'last.pth'
    save_checkpoint(path, model, optimizer, 4, {}, cfg, scheduler=scheduler, scaler=scaler)
    restored = torch.nn.Linear(2, 1)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=9e-2)
    new_scheduler = build_scheduler(new_optimizer, cfg)
    new_scaler = torch.amp.GradScaler('cpu', init_scale=256.)
    checkpoint = load_checkpoint(path, restored, new_optimizer, scheduler=new_scheduler, scaler=new_scaler)
    assert checkpoint['epoch'] == 4
    assert new_scheduler.state_dict() == scheduler.state_dict()
    assert new_scaler.state_dict() == scaler.state_dict()
    step(model, optimizer, scheduler, scaler)
    step(restored, new_optimizer, new_scheduler, new_scaler)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert new_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']


def _check_last_is_actual_final_model_before_restoring_best(tmp_path, early_stop, save_best):
    cfg = json.loads((ROOT / 'configs/v24/smoke.json').read_text())
    cfg['output_dir'] = str(tmp_path / 'outputs')
    cfg['train'].update(epochs=3, val_epoch=1, save_best_checkpoint=save_best,
                        best_checkpoint_name='best.pth', save_last_checkpoint=True,
                        early_stopping={'enabled': early_stop, 'patience': 1})
    cfg['train']['scheduler'] = schedule_config(3)['train']['scheduler']
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    receipt_path = tmp_path / 'receipt.json'
    model = torch.nn.Linear(1, 1)

    def train_epoch(model, loader, optimizer, device, cfg, epoch, **kwargs):
        with torch.no_grad():
            model.weight.fill_(epoch)
        optimizer.zero_grad()
        model.bias.square().sum().backward()
        optimizer.step()
        return {'loss': 1., 'mae': 1., 'rmse': 1.}

    def evaluate(model, *args, **kwargs):
        epoch = kwargs['epoch']
        if kwargs['desc'].startswith('test'):
            assert model.weight.item() == 1.  # Test still uses best, not last.
        return {'loss': float(epoch), 'mae': float(epoch), 'rmse': float(epoch)}

    with patch('sys.argv', ['train.py', '-c', str(path), '--synthetic', '--no_plot',
                            '--result-file', str(receipt_path)]), \
            patch.object(trainer.DualBranchSTImputer, 'from_config', return_value=model), \
            patch.object(trainer, 'build_optimizer', side_effect=lambda m, c: torch.optim.AdamW(m.parameters(), lr=1e-3)), \
            patch.object(trainer, 'train_one_epoch', side_effect=train_epoch), \
            patch.object(trainer, 'evaluate', side_effect=evaluate), \
            patch.object(trainer, '_git_metadata', return_value={'git_commit': 'test'}):
        trainer.main()
    receipt = json.loads(receipt_path.read_text())
    final_epoch = 2 if early_stop else 3
    last = torch.load(receipt['last_checkpoint'], map_location='cpu', weights_only=True)
    assert last['epoch'] == final_epoch
    assert last['model']['weight'].item() == final_epoch
    assert last['training_state']['next_epoch'] == final_epoch + 1
    assert last['training_state']['best_epoch'] == 1
    assert last['scheduler']['last_epoch'] == final_epoch
    assert last['optimizer']['state']
    assert len(last['rng_states']) == 1
    assert 'train_dataset' in last['rng_states'][0]
    assert receipt['best_epoch'] == 1
    if save_best:
        best = torch.load(Path(receipt['run_dir']) / 'checkpoints/best.pth', weights_only=True)
        assert best['model']['weight'].item() == 1
    assert not list(Path(receipt['run_dir']).rglob('*.tmp'))


class TrainingCheckpointTests(unittest.TestCase):
    def test_endpoint_schedule(self):
        for epochs in (1, 20, 100):
            with self.subTest(epochs=epochs):
                _check_cosine_uses_initial_and_final_lr_and_never_rebounds(epochs)

    def test_historical_schedule(self):
        _check_legacy_cosine_still_uses_original_period()

    def test_restore_next_step(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_checkpoint_restores_identical_next_optimizer_and_scheduler_step(Path(directory))

    def test_final_vs_best_and_early_stop(self):
        for early_stop, save_best in ((False, True), (True, True), (False, False)):
            with self.subTest(early_stop=early_stop, save_best=save_best):
                with tempfile.TemporaryDirectory() as directory:
                    _check_last_is_actual_final_model_before_restoring_best(Path(directory), early_stop, save_best)


if __name__ == '__main__':
    unittest.main()
