#!/usr/bin/env python3
"""One full-size AMP update and checkpoint roundtrip; not a training experiment."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from release_utils import load, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=1)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.data import build_datasets, build_loader
    from stmoe_imputer.losses import compute_main_stage_loss
    from stmoe_imputer.engine import build_optimizer, build_scheduler, build_grad_scaler
    from stmoe_imputer.utils.deterministic import configure, state_hash
    from stmoe_imputer.utils.gpu_monitor import GPUMonitor
    from stmoe_imputer.utils.checkpoint import save_checkpoint
    from train_core import rng_snapshot, restore_rng
    a.run_dir.mkdir(parents=True, exist_ok=False)
    cfg = load(ROOT / 'configs/i37_taxibj.json'); configure(cfg)
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda'); started = time.monotonic()
    with GPUMonitor(a.run_dir, enabled=True):
        ds = build_datasets(cfg, a.data_dir / 'taxibj_train.npz', a.data_dir / 'taxibj_val.npz')[0]
        loader = build_loader(ds, cfg, shuffle=True)
        batch = {k: v.to(device) for k, v in next(iter(loader)).items()}
        assert batch['x_f_obs'].shape == (32, 2, 12, 32, 32)
        model = DualBranchSTImputer.from_config(cfg).to(device)
        optimizer = build_optimizer(model, cfg); scheduler = build_scheduler(optimizer, cfg)
        scaler = build_grad_scaler(device, cfg)
        calls = []
        hooks = [e.register_forward_hook(lambda m, args, out: calls.append(args[0].shape[0]))
                 for e in model.main_branch.pattern_experts.values()]
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(batch); loss, _ = compute_main_stage_loss(out, batch, cfg)
        for h in hooks: h.remove()
        assert sum(calls) == 32 * 8
        assert (out['coe']['triscale_choices'][:, -1] == 0).all()
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip_norm'])
        scaler.step(optimizer); scaler.update(); scheduler.step()
        rng = rng_snapshot({'train': loader}, device)
        cp = a.run_dir / 'roundtrip.pth'
        save_checkpoint(cp, model, optimizer, 1, {'loss': float(loss.detach())}, cfg,
                        scheduler=scheduler, scaler=scaler, rng_states=[rng], training_state={'preflight': True})
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        expected = state_hash(model.state_dict())
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng_states'][0], {'train': loader}, device)
        assert state_hash(model.state_dict()) == expected
        assert state_hash(rng_snapshot({'train': loader}, device)) == state_hash(rng)
        torch.cuda.synchronize()
        result = {'status': 'passed', 'physical_gpu': a.gpu, 'gpu': torch.cuda.get_device_name(),
                  'batch_shape': list(batch['x_f_obs'].shape), 'amp': True, 'expert_executions': sum(calls),
                  'loss': float(loss.detach()), 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'seconds': time.monotonic() - started, 'checkpoint_and_rng_roundtrip': True,
                  'torch': str(torch.__version__)}
        write(a.run_dir / 'preflight.json', result)
        print(result, flush=True)


if __name__ == '__main__': main()
