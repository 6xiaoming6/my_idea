#!/usr/bin/env python3
"""Validate each frozen Y configuration on real batch32 before training."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).resolve().parent)]
import torch
from run_b3_c3 import load, write, digest
from run_delta_dependence import validate
from run_team_accept_v4 import check_gpu_idle
from stmoe_imputer.data import build_datasets, build_loader
from stmoe_imputer.engine import build_optimizer, build_grad_scaler, move_batch_to_device
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import configure


def check_case(job, batch, heavy):
    c = job['config']; configure(c); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    model = DualBranchSTImputer.from_config(c).cuda().train()
    main = model.main_branch
    if main.communication != 'none' or any(p.requires_grad for p in main.history_gate.parameters()):
        raise RuntimeError('Delta is not disabled/frozen')
    def forbidden(*args, **kwargs):
        raise RuntimeError('Disabled difference branch executed')
    main.rms_match = forbidden
    main.history_gate.register_forward_pre_hook(forbidden)
    if heavy:
        with torch.no_grad():
            for router in main.routers:
                router[-1].weight.zero_(); router[-1].bias.zero_(); router[-1].bias[4:6] = 10
    optimizer = build_optimizer(model, c); scaler = build_grad_scaler(torch.device('cuda'), c)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic(); updates = 0
    for attempt in range(24):
        optimizer.zero_grad(set_to_none=True); calls = []
        hooks = [e.register_forward_pre_hook(lambda m,a:calls.append(len(a[0]))) for e in main.routed_experts(0)]
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(batch); loss, logs = compute_coe_loss(out, batch, c)
        finally:
            for hook in hooks:
                hook.remove()
        if sum(calls) != 8 * len(batch['m_f']) or not torch.isfinite(loss):
            raise RuntimeError('Sparse dispatch budget or finite loss failed')
        if c['loss']['lambda_coe_mid'] and not bool(logs['l_coe_mid'] > 0):
            raise RuntimeError('Intermediate supervision inactive')
        scale = scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        groups = [getattr(main, name) for name in ('x_history', 'x_global_plane', 'x_interaction') if hasattr(main, name)]
        if main.core_spec['communication'] != 'none':
            groups.append(main.history_gate)
        valid = all(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in block.parameters())
                    and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in block.parameters()) for block in groups)
        if any(p.grad is not None for p in main.history_gate.parameters()):
            raise RuntimeError('Disabled gate received gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() >= scale:
            if not valid:
                raise RuntimeError('No finite nonzero gradient in active extension/gate')
            updates += 1
        if updates >= 3:
            break
    if updates < 3:
        raise RuntimeError('Insufficient AMP optimizer updates')
    torch.cuda.synchronize()
    return dict(variant=job['variant'], case='forced_TA_ST' if heavy else 'native',
                rate=c['data']['mask']['missing_rate'], mid=c['loss']['lambda_coe_mid'],
                batch_size=len(batch['m_f']), input_grid=list(batch['m_f'].shape[2:]),
                measured_missing_fraction=float((1-batch['m_f']).float().mean()),
                status='passed', delta_bypassed=True, gate_frozen=True, attempts=attempt+1, updates=updates, expert_calls_per_window=8,
                peak_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-started)

