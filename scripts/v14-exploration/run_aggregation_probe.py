#!/usr/bin/env python3
"""Fast controlled aggregation assay on V14 blocks, NOT the full V14 model.

Subsample within the original train/val/test splits, generate value-independent
matched-count spatial masks, select one best validation checkpoint, then test.
The base stage tests scale necessity and learned aggregation. MoE is optional.
No edits to V14, offline masks, datasets, historical outputs, or GPU settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

# Configure CPU threads/GPU visibility before importing torch, even for direct CLI.
if __name__ == '__main__':
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--gpu', default='0')
    early.add_argument('--cpu', action='store_true')
    early.add_argument('--cpu-threads', type=int, default=2)
    early_args, _ = early.parse_known_args()
    if early_args.cpu_threads < 1 or not early_args.gpu.isdigit():
        early.error('Positive CPU threads and one numeric GPU ID required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if early_args.cpu else early_args.gpu
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(early_args.cpu_threads)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stmoe_imputer.models.blocks import ResidualSTBlock
from stmoe_imputer.metrics import MaskedMetricAccumulator

DATASETS = {
    'BikeNYC': ('data/BikeNYC', 'bikenyc'),
    'TaxiBJ': ('data/TaxiBJ', 'taxibj'),
    'CHAP': ('data/CHAP/beijing', 'chap_beijing'),
}
VARIANTS = ('fine_local', 'fine_wide', 'fixed', 'geometry', 'adaptive', 'uniform', 'moe')


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT/path


def read_windows(path, cap):
    """Stream evenly spaced C-order NCTHW windows; never materialize full NPZ.

Compressed streams still require decompression up to selected offsets. Memory is
bounded by the requested subset, not by the full training array.
"""
    if cap < 1:
        raise ValueError('Sample caps must be positive')
    with zipfile.ZipFile(path) as archive:
        key = 'x_f_gt.npy' if 'x_f_gt.npy' in archive.namelist() else 'x_f.npy'
        with archive.open(key) as stream:
            version = np.lib.format.read_magic(stream)
            readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
            if version not in readers:
                raise ValueError(f'Unsupported NPY version {version}')
            shape, fortran, dtype = readers[version](stream)
            if fortran or len(shape) != 5 or dtype.hasobject or shape[0] < 1 or shape[1] not in (1, 2):
                raise ValueError(f'Expected nonempty numeric C-order [N,C,T,H,W], got {shape}')
            indices = np.linspace(0, shape[0]-1, min(cap, shape[0]), dtype=np.int64)
            size = int(np.prod(shape[1:])) * dtype.itemsize
            windows, position = [], 0
            for index in indices:
                skip = int(index)*size-position
                while skip:
                    chunk = stream.read(min(skip, 1024*1024))
                    if not chunk:
                        raise EOFError(path)
                    skip -= len(chunk)
                    position += len(chunk)
                raw = stream.read(size)
                if len(raw) != size:
                    raise EOFError(path)
                position += size
                windows.append(np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).astype(np.float32))
    x = torch.from_numpy(np.stack(windows))
    if not torch.isfinite(x).all():
        raise ValueError(f'Nonfinite targets in selected windows: {path}; no silent replacement')
    return x, indices.tolist()


def masks(shape, pattern, rate, seed, indices):
    """Both patterns remove EXACTLY round(rate*H*W) cells, fixed over time.

scattered != previous temporal random protocol. block is a compact spatial hole,
not the project's old fixed CSV. Independent masks for each sample/split.
"""
    n, _, t, h, w = shape
    k = round(rate*h*w)
    if not 0 < k < h*w or pattern not in ('scattered', 'block'):
        raise ValueError('Need a nontrivial missing count and scattered/block pattern')
    result = torch.ones(n, 1, t, h, w)
    yy, xx = np.mgrid[:h, :w]
    for row, index in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index)]))
        if pattern == 'scattered':
            hidden = rng.choice(h*w, k, replace=False)
        else:
            y, x = rng.integers(h), rng.integers(w)
            # Compact square-like region clipped by domain boundaries; exact count.
            distance = np.maximum(abs(yy-y), abs(xx-x)).astype(float)
            hidden = np.argsort((distance + rng.random((h, w))*.01).ravel())[:k]
        result[row].view(t, h*w)[:, hidden] = 0
    return result


def flat(x):
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b*t, h*w, c)


def grid(x, b, t, h, w):
    return x.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()


def scatter(values, weights, indices, n):
    result = values.new_zeros(values.shape[0], n, values.shape[-1])
    for j in range(indices.shape[1]):
        idx = indices[:, j][None, :, None].expand(values.shape[0], -1, values.shape[-1])
        result.scatter_add_(1, idx, values*weights[..., j:j+1])
    return result


def prolong(values, weights, indices):
    return sum(values[:, indices[:, j]]*weights[..., j:j+1] for j in range(indices.shape[1]))


class Probe(nn.Module):
    """Identical completion blocks across controls; modify only context construction.

Fine controls spend BOTH context blocks on the original grid (not disabled dummy
parameters). fixed uses the same blocks at lower resolution. Extra q/k/router
parameters in adaptive controls are explicitly counted, not claimed exactly equal.
"""
    def __init__(self, channels, variant, options):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.strides = variant, options['strides']
        dim = options['dim']
        if dim < 4 or not self.strides or any(s < 2 or int(s) != s for s in self.strides):
            raise ValueError('dim>=4 and nonempty integer strides>=2 required')
        if len(options['wide_dilations']) != len(self.strides) or any(d < 1 or int(d) != d for d in options['wide_dilations']):
            raise ValueError('One positive integer wide dilation per context branch required')
        self.stem = nn.Sequential(nn.Conv3d(channels+1, dim, 1), nn.GELU())
        self.fine = ResidualSTBlock(dim, 4, 0.)
        self.context = nn.ModuleList([ResidualSTBlock(dim, 4, 0.) for _ in self.strides])
        self.fuse = nn.Conv3d(dim*(1+len(self.strides)), dim, 1)
        self.head = nn.Conv3d(dim, channels, 1)
        # All shared parameters initialized BEFORE optional aggregation modules.
        self.query = nn.ModuleList([nn.Linear(dim, 8, bias=False) for _ in self.strides])
        self.key = nn.ModuleList([nn.Linear(dim, 8, bias=False) for _ in self.strides])
        self.router = nn.ModuleList([nn.Conv3d(dim+1, 3, 1) for _ in self.strides])
        for router in self.router:
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)
        if variant not in ('adaptive', 'uniform', 'moe'):
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)
        if variant != 'moe':
            self.router.requires_grad_(False)
        if variant == 'fine_wide':
            for block, dilation in zip(self.context, options['wide_dilations']):
                for layer in (block.conv1, block.conv2):
                    layer.dilation = (1, dilation, dilation)
                    layer.padding = (1, dilation, dilation)

    def assignments(self, features, mask, stride, scale_index):
        b, d, t, h, w = features.shape
        hc, wc = math.ceil(h/stride), math.ceil(w/stride)
        y, x = torch.meshgrid(torch.arange(h, device=features.device), torch.arange(w, device=features.device), indexing='ij')
        radii = [0] if self.variant == 'fixed' else [1] if self.variant in ('adaptive', 'geometry') else [0, 1, 2]
        if self.variant == 'moe':
            gates = flat(self.router[scale_index](torch.cat((features, mask), 1)).softmax(1))
        else:
            gates = features.new_full((b*t, h*w, len(radii)), 1/len(radii))
        if self.variant not in ('fixed', 'geometry'):
            anchor = F.avg_pool2d(features.permute(0, 2, 1, 3, 4).reshape(b*t, d, h, w),
                                  stride, stride, ceil_mode=True, count_include_pad=False).flatten(2).transpose(1, 2)
            q, key = self.query[scale_index](flat(features)), self.key[scale_index](anchor)
        weights, indices = [], []
        for e, radius in enumerate(radii):
            offset = torch.arange(-radius, radius+1, device=features.device)
            dy, dx = torch.meshgrid(offset, offset, indexing='ij')
            ay, ax = y.flatten()[:, None]//stride+dy.flatten(), x.flatten()[:, None]//stride+dx.flatten()
            valid = (ay >= 0) & (ay < hc) & (ax >= 0) & (ax < wc)
            idx = ay.clamp(0, hc-1)*wc+ax.clamp(0, wc-1)
            if radius:
                dist = ((y.flatten()[:, None]+.5)/stride-ay-.5).square()+((x.flatten()[:, None]+.5)/stride-ax-.5).square()
                logits = -dist[None].expand(b*t, -1, -1) if self.variant == 'geometry' else torch.stack(
                    [(q*key[:, idx[:, j]]).sum(-1)/math.sqrt(q.shape[-1]) for j in range(idx.shape[1])], -1)-dist
                a = logits.masked_fill(~valid[None], -torch.inf).softmax(-1)
            else:
                a = features.new_ones(b*t, h*w, 1)
            weights.append(a*gates[..., e:e+1])
            indices.append(idx)
        return torch.cat(weights, -1), torch.cat(indices, -1), hc, wc

    def forward(self, x, mask):
        observed = torch.where(mask.bool(), x, torch.zeros_like(x))
        count = mask.sum((2, 3, 4), keepdim=True).clamp_min(1)
        center = (observed.sum((2, 3, 4), keepdim=True)/count).detach()
        scale = (((observed-center).square()*mask).sum((2, 3, 4), keepdim=True)/count).sqrt().clamp_min(1.).detach()
        z = (observed-center)/scale*mask
        features = self.stem(torch.cat((z, mask), 1))
        fine = self.fine(features)
        b, _, t, h, w = features.shape
        contexts = []
        for i, (stride, block) in enumerate(zip(self.strides, self.context)):
            if self.variant.startswith('fine_'):
                contexts.append(block(features))
            else:
                a, indices, hc, wc = self.assignments(features, mask, stride, i)
                m = flat(mask)
                mass = scatter(torch.ones_like(m), a*m, indices, hc*wc)
                pooled = scatter(flat(features), a*m, indices, hc*wc)/mass.clamp_min(1e-6)
                coarse = block(grid(pooled, b, t, hc, wc))
                contexts.append(grid(prolong(flat(coarse), a, indices), b, t, h, w))
        pred = self.head(F.gelu(fine+self.fuse(torch.cat([fine, *contexts], 1))))*scale+center
        return pred, center, scale


def epoch_pass(model, loader, device, optimizer=None, grad_clip=1.):
    model.train(optimizer is not None)
    metrics = MaskedMetricAccumulator()
    numerator, denominator = 0., 0.
    for x, mask in loader:
        x, mask = x.to(device), mask.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction, center, scale = model(x, mask)
            missing = (1-mask).expand_as(x)
            count = missing.sum()
            errors = F.smooth_l1_loss((prediction-center)/scale, (x-center)/scale, reduction='none')
            loss = (errors*missing).sum()/count.clamp_min(1)
            if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite loss/prediction; job is not marked complete')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
        metrics.update(prediction, x, mask)
        numerator += float(loss.detach())*float(count)
        denominator += float(count)
    if denominator <= 0:
        raise ValueError('No missing targets for evaluation')
    result = metrics.compute()
    return {'loss': numerator/denominator, 'mae': result['mae'], 'rmse': result['rmse'], 'missing_count': denominator}


def identity(policy, dataset, variant, pattern, seed, device_type):
    folder, prefix = DATASETS[dataset]
    paths = {s: resolve(f'{folder}/{prefix}_{s}.npz') for s in ('train', 'val', 'test')}
    files = {s: {'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} if p.is_file() else None for s, p in paths.items()}
    sources = [Path(__file__), *sorted((ROOT/'src/stmoe_imputer').rglob('*.py'))]
    source_hash = hashlib.sha256()
    for p in sources:
        source_hash.update(str(p.relative_to(ROOT)).encode())
        source_hash.update(p.read_bytes())
    record = {'dataset': dataset, 'variant': variant, 'pattern': pattern, 'seed': seed,
              'device_type': device_type, 'samples': policy['samples'], 'model': policy['model'],
              'train': policy['train'], 'missing_rate': policy['missing_rate'],
              'data_identity': files, 'source_hash': source_hash.hexdigest(),
              'protocol': 'v14-blocks-probe-v1; spatial matched-count masks; NOT full V14'}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return dict(record, fingerprint=digest), paths


def complete(root, record):
    for path in sorted(root.glob('*/result.json'), reverse=True):
        try:
            result = json.loads(path.read_text())
            if result['fingerprint'] != record['fingerprint'] or result['epochs'] != record['train']['epochs']:
                continue
            events = [json.loads(line) for line in (path.parent/'logs/metrics.jsonl').read_text().splitlines()]
            trains = [e for e in events if e['stage'] == 'train']
            tests = [e for e in events if e['stage'] == 'test']
            best = [e for e in events if e['stage'] == 'val' and e.get('is_best')]
            if len(trains) != result['epochs'] or len(tests) != 1 or not best or best[-1]['epoch'] != result['best_epoch']:
                continue
            if [e['epoch'] for e in trains] != list(range(1, result['epochs']+1)) or tests[0]['epoch'] != result['best_epoch']:
                continue
            if not all(math.isfinite(float(e[k])) for e in events for k in ('loss', 'mae', 'rmse')):
                continue
            if not (path.parent/'best.pt').is_file():
                continue
            if not all(math.isfinite(float(result[s][k])) for s in ('val', 'test') for k in ('mae', 'rmse')):
                continue
            return dict(result, run=str(path.parent))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None


def train_job(record, root, data, indices, device):
    run = root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logs = run/'logs'
    logs.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(dict(record, sample_indices=indices), indent=2))
    torch.manual_seed(record['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(record['seed'])
        torch.cuda.reset_peak_memory_stats()
    model = Probe(data['train'].shape[1], record['variant'], record['model']).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loaders = {}
    for split, x in data.items():
        offset = {'train': 0, 'val': 100000, 'test': 200000}[split]
        mask = masks(x.shape, record['pattern'], record['missing_rate'], record['seed']+offset, indices[split])
        loaders[split] = DataLoader(TensorDataset(x, mask), batch_size=record['train']['batch_size'],
            shuffle=split == 'train', num_workers=0,
            generator=torch.Generator().manual_seed(record['seed']+offset))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=record['train']['lr'], weight_decay=record['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, record['train']['epochs'], eta_min=1e-6)
    best_mae, best_epoch, best_val = math.inf, 0, None
    start = time.monotonic()

    def log(stage, epoch, values, **extra):
        event = dict(stage=stage, epoch=epoch, **values, **extra)
        with (logs/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False)+'\n')
        with (logs/f'{stage}.log').open('a') as f:
            f.write(f'epoch={epoch} loss={values["loss"]:.6f} mae={values["mae"]:.6f} rmse={values["rmse"]:.6f} '+json.dumps(extra)+'\n')

    for epoch in range(1, record['train']['epochs']+1):
        epoch_start = time.monotonic()
        train = epoch_pass(model, loaders['train'], device, optimizer, record['train']['grad_clip'])
        log('train', epoch, train, lr=optimizer.param_groups[0]['lr'])
        val = None
        if epoch % record['train']['val_epoch'] == 0 or epoch == record['train']['epochs']:
            val = epoch_pass(model, loaders['val'], device)
            improved = val['mae'] < best_mae
            if improved:
                best_mae, best_epoch, best_val = val['mae'], epoch, val
                temporary = run/'best.pt.tmp'
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'fingerprint': record['fingerprint']}, temporary)
                os.replace(temporary, run/'best.pt')
            log('val', epoch, val, is_best=improved)
        scheduler.step()
        print(f'  epoch {epoch}/{record["train"]["epochs"]} train_mae={train["mae"]:.4f} '
              f'val_mae={val["mae"] if val else "-"} time={time.monotonic()-epoch_start:.1f}s', flush=True)
    checkpoint = torch.load(run/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test = epoch_pass(model, loaders['test'], device)
    log('test', best_epoch, test, checkpoint='best.pt')
    result = dict(fingerprint=record['fingerprint'], epochs=record['train']['epochs'],
                  best_epoch=best_epoch, val=best_val, test=test, trainable_parameters=parameters,
                  seconds=time.monotonic()-start,
                  peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else None)
    (run/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return dict(result, run=str(run))


def summarize(rows):
    print('\nRESULTS: incomplete/mismatched runs are NOT scores; selection uses VAL only.')
    for r in rows:
        if not r['result']:
            print(f'{r["dataset"]:8} {r["pattern"]:9} {r["variant"]:10} seed={r["seed"]} MISSING')
            continue
        v = r['result']
        print(f'{r["dataset"]:8} {r["pattern"]:9} {r["variant"]:10} seed={r["seed"]} '
              f'best={v["best_epoch"]:3} val={v["val"]["mae"]:.5f}/{v["val"]["rmse"]:.5f} '
              f'test={v["test"]["mae"]:.5f}/{v["test"]["rmse"]:.5f} params={v["trainable_parameters"]}')
    lookup = {(r['dataset'], r['pattern'], r['seed'], r['variant']): r['result'] for r in rows if r['result']}
    print('\nPAIRED validation relative changes (negative=better), no pooled cross-dataset raw MAE:')
    for candidate, reference in [('fixed', 'fine_local'), ('fixed', 'fine_wide'), ('adaptive', 'fixed'), ('adaptive', 'geometry'),
                                 ('uniform', 'adaptive'), ('moe', 'uniform')]:
        diffs = []
        for (d, m, seed, v), result in lookup.items():
            if v == candidate and (d, m, seed, reference) in lookup:
                ref = lookup[d, m, seed, reference]
                diffs.append((result['val']['mae']/max(ref['val']['mae'], 1e-12)-1)*100)
        if diffs:
            print(f'{candidate} vs {reference}: n={len(diffs)}, mean={statistics.mean(diffs):+.3f}%, '
                  f'wins={sum(x<0 for x in diffs)}/{len(diffs)}; single-seed screening is NOT significance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/v14-exploration/aggregation_probe.json')
    p.add_argument('--gpu', default='0')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--stage', choices=['base', 'moe', 'all'], default='base')
    p.add_argument('--datasets', nargs='+', choices=list(DATASETS))
    p.add_argument('--variants', nargs='+', choices=VARIANTS)
    p.add_argument('--epochs', type=int)
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()
    policy = json.loads(resolve(args.config).read_text())
    if args.epochs is not None:
        policy['train']['epochs'] = args.epochs
    if any(int(policy['train'][k]) < 1 for k in ('epochs', 'val_epoch', 'batch_size')) or any(int(x) < 1 for x in policy['samples'].values()):
        p.error('Budgets/batch/sample caps must be positive')
    if not 0 < policy['missing_rate'] < 1 or args.cpu_threads < 1 or (args.max_jobs is not None and args.max_jobs < 1):
        p.error('Invalid missing rate / threads / max-jobs')
    if not policy['seeds'] or any(int(s) != s or s < 0 for s in policy['seeds']) or len(set(policy['seeds'])) != len(policy['seeds']):
        p.error('Use distinct nonnegative integer seeds')
    if not policy['patterns'] or any(m not in ('scattered', 'block') for m in policy['patterns']):
        p.error('Patterns must be scattered/block')
    if not policy['datasets'] or any(d not in DATASETS for d in policy['datasets']):
        p.error('Unsupported dataset')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    kind = 'cpu' if args.cpu else 'cuda'
    # Summary/dry-run can inspect GPU-result identities even on a CPU machine.
    if not (args.summary or args.dry_run) and kind == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable. Activate difftdi or explicitly use --cpu; no silent CPU fallback')
    work = [(d, m, v, s) for d in args.datasets or policy['datasets'] for m in policy['patterns']
            for v in args.variants or policy['stages'][args.stage] for s in policy['seeds']]
    plans = []
    for d, m, v, seed in work:
        record, paths = identity(policy, d, v, m, seed, kind)
        root = resolve(policy['output_dir'])/d/m/v/record['fingerprint'][:16]
        plans.append((record, paths, root, complete(root, record)))
    print(f'Controlled probe, NOT full V14: jobs={len(plans)} completed={sum(old is not None for *_, old in plans)} '
          f'epochs={policy["train"]["epochs"]}, samples={policy["samples"]}, single {kind}', flush=True)
    if not (args.summary or args.dry_run):
        for _, paths, _, _ in plans:
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
    cached_dataset, data, indices, launched, rows = None, None, None, 0, []
    for number, (record, paths, root, old) in enumerate(plans, 1):
        label = f'{record["dataset"]}/{record["pattern"]}/{record["variant"]}/seed{record["seed"]}'
        if not (args.summary or args.dry_run) and old is None and (args.max_jobs is None or launched < args.max_jobs):
            if cached_dataset != record['dataset']:
                data, indices = {}, {}
                for split, path in paths.items():
                    data[split], indices[split] = read_windows(path, policy['samples'][split])
                cached_dataset = record['dataset']
                print(f'[data] {cached_dataset}: '+str({s: list(x.shape) for s, x in data.items()}), flush=True)
            print(f'[{number}/{len(plans)}] RUN {label}', flush=True)
            old = train_job(record, root, data, indices, torch.device('cpu' if args.cpu else 'cuda:0'))
            launched += 1
        elif not args.summary:
            print(f'[{number}/{len(plans)}] {"SKIP" if old else "PENDING"} {label}', flush=True)
        rows.append(dict(dataset=record['dataset'], pattern=record['pattern'], variant=record['variant'], seed=record['seed'], result=old))
    if not args.dry_run:
        summarize(rows)


if __name__ == '__main__':
    main()
