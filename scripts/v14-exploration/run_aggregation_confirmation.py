#!/usr/bin/env python3
"""Predeclared, multi-seed aggregation confirmation; preserve the earlier probe.

Five core controls, budgeted training windows and FULL validation/test splits. A
deadline is admission control, not permission to truncate or cherry-pick epochs.
Use --calibrate to estimate the WHOLE queue before launching it.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

if __name__ == '__main__':
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--gpu', default='0')
    early.add_argument('--cpu', action='store_true')
    early.add_argument('--cpu-threads', type=int, default=2)
    a, _ = early.parse_known_args()
    if not a.gpu.isdigit() or a.cpu_threads < 1:
        early.error('One numeric GPU and positive threads required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if a.cpu else a.gpu
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(a.cpu_threads)

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import run_aggregation_probe as base

ROOT = base.ROOT
OriginalProbe = base.Probe
original_masks = base.masks
original_pass = base.epoch_pass
VARIANTS = ('single_wide', 'uniform', 'static_moe', 'geometry_moe', 'fine_wide', 'fixed', 'context_moe')


class StaticRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))

    def forward(self, x):
        return self.logits.expand(x.shape[0], 3, *x.shape[2:])


class GeometryRouter(nn.Module):
    """Same 1x1 trainable head as the old point router; add NO trainable layers.

Neighbourhood observed-feature means and mask densities give missing queries
different inputs. Statistics only use observed input; optional context_moe is a
matched-capacity control without explicit density/gradient channel additions.
"""
    def __init__(self, head, with_geometry=True):
        super().__init__()
        self.head = head
        self.with_geometry = with_geometry

    def inputs(self, x):
        features, mask = x[:, :-1], x[:, -1:]
        def pool(z, k):
            # BikeNYC width=12: torch AvgPool3d rejects kernel 15 even with
            # padding. Use the largest fitting odd neighbourhood on each axis.
            kh = min(k, z.shape[-2] if z.shape[-2] % 2 else z.shape[-2]-1)
            kw = min(k, z.shape[-1] if z.shape[-1] % 2 else z.shape[-1]-1)
            return F.avg_pool3d(z, (1, kh, kw), stride=1, padding=(0, kh//2, kw//2), count_include_pad=False)
        d3, d7, d15 = (pool(mask, k) for k in (3, 7, 15))
        context = pool(features*mask, 7)/d7.clamp_min(1e-6)
        if self.with_geometry:
            # Parameter-free geometry injection keeps the original head capacity.
            dx = F.pad(d7[..., 1:]-d7[..., :-1], (0, 1, 0, 0))
            dy = F.pad(d7[..., 1:, :]-d7[..., :-1, :], (0, 0, 0, 1))
            stats = torch.cat((d3, d7, d15, dx, dy), 1)
            context = torch.cat((context[:, :5]+stats, context[:, 5:]), 1)
        return torch.cat((context, mask), 1)

    def forward(self, x):
        return self.head(self.inputs(x))


class ConfirmationProbe(OriginalProbe):
    def __init__(self, channels, variant, options):
        if variant not in VARIANTS:
            raise ValueError(variant)
        original = variant if variant in ('fine_wide', 'fixed', 'uniform') else 'uniform' if variant == 'single_wide' else 'moe'
        super().__init__(channels, original, options)
        self.confirmation_variant = variant
        if variant == 'static_moe':
            self.router = nn.ModuleList([StaticRouter() for _ in self.strides])
        elif variant in ('geometry_moe', 'context_moe'):
            self.router = nn.ModuleList([GeometryRouter(head, variant == 'geometry_moe') for head in self.router])
        self.route_totals = {}
        self._hooks = []
        if variant in ('static_moe', 'geometry_moe', 'context_moe'):
            for stride, router in zip(self.strides, self.router):
                self._hooks.append(router.register_forward_hook(self.route_hook(stride)))

    def route_hook(self, stride):
        def hook(module, inputs, logits):
            if self.training:
                return
            with torch.no_grad():
                p = logits.detach().float().softmax(1)
                missing = 1-inputs[0][:, -1:]
                count = missing.sum()
                moment = torch.cat((count.reshape(1), (p*missing).sum((0, 2, 3, 4)),
                                    (p.square()*missing).sum((0, 2, 3, 4))))
                self.route_totals[stride] = self.route_totals.get(stride, torch.zeros_like(moment))+moment
        return hook

    def assignments(self, features, mask, stride, scale_index):
        weights, indices, hc, wc = super().assignments(features, mask, stride, scale_index)
        if self.confirmation_variant == 'single_wide':
            # Parent uniform assignment has [1,9,25] edges and a 1/3 factor.
            # Remove the first two experts entirely from the actual aggregation.
            return weights[..., -25:]*3, indices[:, -25:], hc, wc
        return weights, indices, hc, wc


def epoch_pass(model, loader, device, optimizer=None, grad_clip=1.):
    model.route_totals = {}
    result = original_pass(model, loader, device, optimizer, grad_clip)
    if optimizer is None:
        result['routing_missing'] = {}
        for stride, moments in model.route_totals.items():
            count, *rest = moments.cpu().tolist()
            mean = [v/max(count, 1) for v in rest[:3]]
            std = [math.sqrt(max(v/max(count, 1)-m*m, 0)) for v, m in zip(rest[3:], mean)]
            result['routing_missing'][str(stride)] = {'mean': mean, 'std': std, 'count': count}
    return result


def fixed_mask_function(mask_seed, run_seed):
    def make(shape, pattern, rate, seed, indices):
        # Original trainer supplies run_seed + split_offset; remove ONLY run seed.
        return original_masks(shape, pattern, rate, mask_seed+seed-run_seed, indices)
    return make


def identity(policy, dataset, variant, pattern, seed, device_type):
    record, paths = base.identity(policy, dataset, variant, pattern, seed, device_type)
    record.update(protocol='aggregation-confirmation-v1; original V14 blocks, NOT full V14',
                  mask_seed=policy['mask_seed'],
                  extension_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime={'torch': torch.__version__, 'cuda': torch.version.cuda})
    record['fingerprint'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record, paths


def load_data(paths, policy):
    data, indices = {}, {}
    for split, path in paths.items():
        data[split], indices[split] = base.read_windows(path, policy['samples'][split])
    return data, indices


def calibration(record, data, device):
    """Disposable real-data warmup+timing. NEVER a completed accuracy experiment."""
    torch.manual_seed(record['seed'])
    batch = record['train']['batch_size']
    x = data['train'][:batch]
    mask = original_masks(x.shape, record['pattern'], record['missing_rate'], record['mask_seed'], list(range(len(x))))
    # Repeat ONLY for timing; formal training uses the full selected sample set.
    ds = TensorDataset(x, mask)
    loader = DataLoader(ds, batch_size=batch)
    model = ConfirmationProbe(x.shape[1], record['variant'], record['model']).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=record['train']['lr'])
    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None
    original_pass(model, loader, device, optimizer)
    train_times, eval_times = [], []
    for _ in range(3):
        sync(); start = time.monotonic()
        epoch_pass(model, loader, device, optimizer)
        sync(); train_times.append(time.monotonic()-start)
        start = time.monotonic()
        epoch_pass(model, loader, device)
        sync(); eval_times.append(time.monotonic()-start)
    train_step, eval_step = statistics.median(train_times), statistics.median(eval_times)
    n = record['train']['epochs']
    checks = n//record['train']['val_epoch'] + int(n % record['train']['val_epoch'] != 0)
    estimate = train_step*math.ceil(len(data['train'])/batch)*n
    estimate += eval_step*(math.ceil(len(data['val'])/batch)*checks+math.ceil(len(data['test'])/batch))
    return {'seconds': estimate+checks*.1+3, 'train_step': train_step, 'eval_step': eval_step,
            'actual_samples': {s: len(x) for s, x in data.items()}}


def summary(rows):
    base.summarize(rows)
    lookup = {(r['dataset'], r['pattern'], r['seed'], r['variant']): r['result'] for r in rows if r['result']}
    print('\nCONFIRMATION: validation-only paired decisions; negative=better; no raw cross-dataset averaging.')
    for candidate, ref in [('uniform', 'single_wide'), ('static_moe', 'uniform'),
                           ('geometry_moe', 'static_moe'), ('geometry_moe', 'uniform'), ('geometry_moe', 'context_moe')]:
        for d, m in sorted({(r['dataset'], r['pattern']) for r in rows}):
            pairs = [(seed, lookup[d, m, seed, candidate], lookup[d, m, seed, ref])
                     for seed in sorted({r['seed'] for r in rows})
                     if (d, m, seed, candidate) in lookup and (d, m, seed, ref) in lookup]
            if not pairs:
                continue
            mae = [(a['val']['mae']/max(b['val']['mae'], 1e-12)-1)*100 for _, a, b in pairs]
            rmse = [(a['val']['rmse']/max(b['val']['rmse'], 1e-12)-1)*100 for _, a, b in pairs]
            print(f'{d}/{m} {candidate} vs {ref}: n={len(mae)} '
                  f'MAE={statistics.mean(mae):+.3f}% SD={statistics.stdev(mae) if len(mae)>1 else None} '
                  f'RMSE={statistics.mean(rmse):+.3f}% wins={sum(v<0 for v in mae)}/{len(mae)}')
    print('Gate std>0 shows differing inputs/weights, NOT proof that routing helps. Read paired metrics too.')


def save_summary(rows, policy, profile):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summary(rows)
    content = buffer.getvalue()
    print(content, end='', flush=True)
    root = base.resolve(policy['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    name = f'summary_{profile or "budgeted"}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    (root/f'{name}.log').write_text(content)
    (root/f'{name}.json').write_text(json.dumps({'policy': policy, 'rows': rows}, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/v14-exploration/aggregation_confirmation.json')
    p.add_argument('--profile', choices=['full'], help='All original train/val/test windows; not a 13:00 promise')
    p.add_argument('--gpu', default='0')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--calibrate', action='store_true', help='Time real batches only; no formal training')
    p.add_argument('--ignore-deadline', action='store_true', help='Explicitly finish remaining jobs even after configured deadline')
    p.add_argument('--max-blocks', type=int, help='Limit newly trained complete variant blocks')
    args = p.parse_args()
    policy = json.loads(base.resolve(args.config).read_text())
    if args.profile:
        for key, patch in policy['profiles'][args.profile].items():
            if isinstance(patch, dict):
                policy[key] = dict(policy.get(key, {}), **patch)
            else:
                policy[key] = patch
    if not policy['variants'] or any(v not in VARIANTS for v in policy['variants']):
        p.error('Invalid variants')
    if len(set(policy['seeds'])) != len(policy['seeds']) or not policy['seeds'] or any(s<0 for s in policy['seeds']):
        p.error('Distinct nonnegative seeds required')
    if any(policy['train'][k]<1 for k in ('epochs', 'val_epoch', 'batch_size')) or any(n<1 for n in policy['samples'].values()):
        p.error('Positive budgets required')
    if args.max_blocks is not None and args.max_blocks < 1:
        p.error('Positive --max-blocks required')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    kind = 'cpu' if args.cpu else 'cuda'
    if not (args.summary or args.dry_run) and kind=='cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; activate difftdi or explicitly use --cpu')
    device = torch.device('cpu' if args.cpu else 'cuda:0')
    deadline = datetime.fromisoformat(policy['deadline']).timestamp()
    plans = []
    # Whole paired variant blocks, rotating datasets for each seed. No selection
    # based on validation/test outcomes or preferential scheduling of winners.
    for seed in policy['seeds']:
        for d in policy['datasets']:
            for m in policy['patterns']:
                block = []
                for v in policy['variants']:
                    record, paths = identity(policy, d, v, m, seed, kind)
                    root = base.resolve(policy['output_dir'])/d/m/v/record['fingerprint'][:16]
                    block.append((record, paths, root, base.complete(root, record)))
                plans.append(block)
    print(f'CONFIRMATION jobs={sum(map(len,plans))}, completed={sum(old is not None for b in plans for *_,old in b)}, '
          f'epochs={policy["train"]["epochs"]} seeds={policy["seeds"]} '
          f'train_cap={policy["samples"]["train"]} val/test=full (subject to configured caps)', flush=True)
    rows = []
    if args.summary or args.dry_run:
        for block in plans:
            for c, _, _, old in block:
                rows.append(dict(dataset=c['dataset'],pattern=c['pattern'],variant=c['variant'],seed=c['seed'],result=old))
                if args.dry_run:
                    print('SKIP' if old else 'PENDING', c['dataset'],c['pattern'],c['variant'],c['seed'])
        if args.summary:
            save_summary(rows, policy, args.profile)
        return
    if not args.ignore_deadline and time.time() >= deadline:
        p.error('Deadline passed. No training launched; use --ignore-deadline for remaining jobs.')
    for block in plans:
        for _, paths, _, _ in block:
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
    timing, cache, cached = {}, None, None
    print('[calibration] real-data disposable timing on selected GPU; no accuracy results saved', flush=True)
    for block in plans:
        for record, paths, _, old in block:
            key = (record['dataset'], record['variant'])
            if key in timing or old:
                continue
            if cached != record['dataset']:
                cache = load_data(paths, policy)
                cached = record['dataset']
            timing[key] = calibration(record, cache[0], device)
            print('[timing]', key, timing[key], flush=True)
    def estimate(block):
        return sum(timing[(c['dataset'],c['variant'])]['seconds'] for c,_,_,old in block if old is None)
    estimate_all = sum(estimate(block) for block in plans)*policy['timing_safety_factor']
    remaining = deadline-time.time()-policy['deadline_reserve_seconds']
    print(f'[estimate] pending_with_margin={estimate_all/3600:.2f}h available={remaining/3600:.2f}h; '
          'estimate only, no guarantee under load changes', flush=True)
    timing_root = base.resolve(policy['output_dir'])/'calibration'
    timing_root.mkdir(parents=True, exist_ok=True)
    timing_path = timing_root/f'{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    timing_path.write_text(json.dumps({'samples': policy['samples'], 'train': policy['train'],
        'variants': policy['variants'], 'seeds': policy['seeds'], 'timings': {f'{d}/{v}': x for (d,v),x in timing.items()},
        'estimated_seconds_with_margin': estimate_all, 'available_seconds': remaining,
        'device': str(device), 'deadline': policy['deadline']}, indent=2))
    if args.calibrate:
        return
    if not args.ignore_deadline and estimate_all > remaining:
        p.error('Whole fixed protocol is estimated to exceed deadline. No formal runs launched. '
                'Do not silently shrink seeds/epochs; explicitly revise deadline/budget or use --ignore-deadline.')
    # Worker-local adapters leave original source and historical identities intact.
    base.Probe, base.epoch_pass = ConfirmationProbe, epoch_pass
    blocks_done = 0
    for block in plans:
        needed = any(old is None for *_, old in block)
        allowed = args.max_blocks is None or blocks_done < args.max_blocks
        seconds = estimate(block)*policy['timing_safety_factor']
        allowed = allowed and (args.ignore_deadline or time.time()+seconds+policy['deadline_reserve_seconds'] < deadline)
        if needed and allowed:
            c, paths, _, _ = block[0]
            if cached != c['dataset']:
                cache = load_data(paths, policy); cached = c['dataset']
            for c, _, root, old in block:
                if old is None:
                    print('[RUN]',c['dataset'],c['pattern'],c['variant'],c['seed'],flush=True)
                    base.masks = fixed_mask_function(policy['mask_seed'], c['seed'])
                    old = base.train_job(c, root, cache[0], cache[1], device)
                rows.append(dict(dataset=c['dataset'],pattern=c['pattern'],variant=c['variant'],seed=c['seed'],result=old))
            blocks_done += 1
        else:
            for c,_,_,old in block:
                rows.append(dict(dataset=c['dataset'],pattern=c['pattern'],variant=c['variant'],seed=c['seed'],result=old))
            if needed:
                print('[PENDING BLOCK]',block[0][0]['dataset'],block[0][0]['pattern'],block[0][0]['seed'],flush=True)
    save_summary(rows, policy, args.profile)
    print('Remaining blocks must be reported as incomplete, not omitted.', flush=True)


if __name__ == '__main__':
    main()
