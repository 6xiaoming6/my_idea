#!/usr/bin/env python3
"""Matched 2x2 V23 comparison; single GPU, full validation/test, bounded train set.

Run in the intended PyTorch environment. --calibrate estimates rather than
launching the experiment queue. No automatic truncation or unequal epoch budgets.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'TaxiBJ': ('TaxiBJ', 'taxibj', 'taxibj'),
    'BikeNYC': ('BikeNYC', 'bikenyc', 'bikenyc'),
    'CHAP': ('CHAP/beijing', 'chap_beijing', 'chap_beijing'),
}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT/p


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def merge(base, patch):
    result = dict(base)
    for k, v in patch.items():
        result[k] = merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
    return result


def stamp(path):
    s = path.stat()
    return {'path': str(path.relative_to(ROOT)), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def identity(policy):
    sources = []
    for dataset in policy['datasets']:
        folder, prefix, base = SPECS[dataset]
        sources.append(stamp(ROOT/f'configs/datasets/{base}.json'))
        for split in ('train', 'val', 'test'):
            sources.append(stamp(ROOT/f'data/{folder}/{prefix}_{split}.npz'))
            for pattern in policy['patterns']:
                for rate in policy.get('rates', [policy.get('rate')]):
                    sources.append(stamp(ROOT/f'data/{folder}/{pattern}_mask/{rate:g}/{split}.csv'))
    code = list((ROOT/'src/stmoe_imputer').rglob('*.py')) + [Path(__file__), ROOT/'scripts/train.py', ROOT/'configs/presets/dual_moe.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code)}
    return {'protocol': 'v23-learned-regions-2x2-v1', 'policy': policy, 'sources': sources, 'code': hashes}


def read_windows(path, cap):
    """Bounded-memory C-order NCTHW selection, spanning the original split."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        name = 'x_f_gt.npy' if 'x_f_gt.npy' in archive.namelist() else 'x_f.npy'
        with archive.open(name) as f:
            version = np.lib.format.read_magic(f)
            readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
            shape, fortran, dtype = readers[version](f)
            if fortran or dtype.hasobject or len(shape) != 5 or shape[1] not in (1, 2) or shape[0] < 1:
                raise ValueError(f'Expected numeric C-order NCTHW: {path}: {shape}')
            ids = np.linspace(0, shape[0]-1, min(cap, shape[0]), dtype=np.int64)
            size = math.prod(shape[1:])*dtype.itemsize
            values, pos = [], 0
            for index in ids:
                skip = int(index)*size-pos
                while skip:
                    raw = f.read(min(skip, 1024*1024))
                    if not raw:
                        raise EOFError(path)
                    skip -= len(raw); pos += len(raw)
                raw = f.read(size)
                if len(raw) != size:
                    raise EOFError(path)
                values.append(np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).copy())
                pos += size
    return np.stack(values), ids.tolist(), shape[0]


def prepare(policy, suite):
    """Subset only TRAIN. CSV rows follow exactly the selected NPZ row IDs."""
    import numpy as np
    result = {}
    for dataset in policy['datasets']:
        folder, prefix, _ = SPECS[dataset]
        dest = suite/'data'/dataset
        manifest = dest/'selection.json'
        masks = [(p, r, dest/(f'{p}_rate{r:g}_train.csv' if 'rates' in policy else f'{p}_train.csv'))
                 for p in policy['patterns'] for r in policy.get('rates', [policy.get('rate')])]
        required = [dest/'train.npz']+[path for _, _, path in masks]
        if not manifest.exists() or not all(p.is_file() for p in required):
            x, ids, original_count = read_windows(ROOT/f'data/{folder}/{prefix}_train.npz', policy['train_windows'])
            dest.mkdir(parents=True, exist_ok=True)
            np.savez(dest/'train.npz', x_f_gt=x)
            for pattern, rate, target in masks:
                source = ROOT/f'data/{folder}/{pattern}_mask/{rate:g}/train.csv'
                selected, rows = set(ids), 0
                with source.open() as src, target.open('w') as dst:
                    for i, line in enumerate(src):
                        if not line.strip():
                            raise ValueError(f'Blank mask row: {source}:{i+1}')
                        rows += 1
                        if pattern == 'fixed' or i in selected:
                            dst.write(line if line.endswith('\n') else line+'\n')
                if rows != (1 if pattern == 'fixed' else original_count):
                    raise ValueError(f'Mask row count mismatch: {source}')
            write_json(manifest, {'indices': ids, 'original_train_windows': original_count, 'selected_train_windows': len(ids)})
            print(f'[prepare] {dataset}: train {len(ids)}/{original_count}, full val/test', flush=True)
        result[dataset] = dest
    return result


def job_config(policy, suite, data, dataset, pattern, variant):
    folder, prefix, base = SPECS[dataset]
    cfg = merge(load(ROOT/f'configs/datasets/{base}.json'), load(ROOT/'configs/presets/dual_moe.json'))
    cfg = merge(cfg, {
        'seed': policy['seed'], 'output_dir': str(suite/'runs'), 'device': 'cuda:0',
        'data': {'batch_size': policy['batch_size'], 'num_workers': 0, 'drop_last': False,
                 'mask': {'pattern': pattern, 'missing_rate': policy['rate'],
                          'train_csv': str(data/f'{pattern}_train.csv'),
                          'val_csv': str(ROOT/f'data/{folder}/{pattern}_mask/{policy["rate"]:g}/val.csv'),
                          'test_csv': str(ROOT/f'data/{folder}/{pattern}_mask/{policy["rate"]:g}/test.csv')}},
        'model': {'dual_moe': policy['variants'][variant]},
        'train': {'epochs': policy['epochs'], 'val_epoch': policy['val_epoch'], 'early_stopping': {'enabled': False}},
    })
    paths = {'train': data/'train.npz', **{s: ROOT/f'data/{folder}/{prefix}_{s}.npz' for s in ('val', 'test')}}
    return cfg, paths


def completed_run(cfg, variant):
    """Require full finite best-model testing, with disk or explicit memory provenance."""
    base = Path(cfg['output_dir'])/cfg['data']['dataset_name']/'ablation'/variant/cfg['data']['mask']['pattern']/f'rate{cfg["data"]["mask"]["missing_rate"]:g}'
    for p in sorted(base.glob('*/config.json'), reverse=True):
        run = p.parent
        try:
            save_best = cfg['train'].get('save_best_checkpoint', True)
            if type(save_best) is not bool or load(p) != cfg:
                continue
            if save_best and not (run/'checkpoints/best.pt').is_file():
                continue
            train = (run/'logs/train.log').read_text()
            entries = [json.loads(line) for line in (run/'logs/metrics.jsonl').read_text().splitlines()]
            records = [r for r in entries if 'epoch' in r]
            tests = [r for r in entries if r.get('stage') == 'test']
            if 'Training finished normally:' not in train or len(records) != cfg['train']['epochs']:
                continue
            if [r['epoch'] for r in records] != list(range(1, cfg['train']['epochs']+1)):
                continue
            valid = [r for r in records if r.get('val') is not None]
            if not valid or not all(math.isfinite(r['val']['mae']) for r in valid):
                continue
            if len(tests) != 1 or 'Testing finished:' not in (run/'logs/test.log').read_text():
                continue
            test = {**tests[0]['metrics'], 'best_epoch': tests[0]['extra']['best_epoch']}
            if not all(k in test and math.isfinite(test[k]) for k in ('mae','rmse','best_epoch')):
                continue
            best = min(valid, key=lambda r:r['val']['mae'])
            if test['best_epoch'] != best['epoch']:
                continue
            if not save_best:
                extra = tests[0]['extra']
                if (extra.get('best_model_source') != 'memory' or extra.get('best_weights_restored') is not True
                        or extra.get('checkpoint') != 'not_saved'
                        or extra.get('best_val_mae') != best['val']['mae']):
                    continue
            return {'run_dir': str(run), 'val_mae': min(r['val']['mae'] for r in valid),
                    'test_mae': test['mae'], 'test_rmse': test['rmse'], 'best_epoch': int(test['best_epoch'])}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def calibration(policy, suite, prepared):
    import torch
    from torch.utils.data import Subset
    sys.path.insert(0, str(ROOT/'src'))
    from stmoe_imputer.data import build_datasets, build_test_dataset, build_loader
    from stmoe_imputer.engine import build_optimizer, train_one_epoch, evaluate
    from stmoe_imputer.models import DualBranchSTImputer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: activate difftdi first')
    torch.set_num_threads(policy['cpu_threads'])
    timings = {}
    for dataset in policy['datasets']:
        cfg, paths = job_config(policy, suite, prepared[dataset], dataset, policy['patterns'][0], 'A11')
        train, val = build_datasets(cfg, str(paths['train']), str(paths['val']))
        test = build_test_dataset(cfg, str(paths['test']))
        # Include Dataset.__getitem__, CPU pooling/collation and H2D transfer in
        # timing, rather than timing pre-materialized batches optimistically.
        train_batches = build_loader(Subset(train, range(min(len(train),8*policy['batch_size']))), cfg, False)
        val_batches = build_loader(Subset(val, range(min(len(val),8*policy['batch_size']))), cfg, False)
        model = DualBranchSTImputer.from_config(cfg).cuda(); optimizer = build_optimizer(model, cfg)
        def timed(training):
            torch.cuda.synchronize(); start = time.perf_counter()
            batches = train_batches if training else val_batches
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                if training:
                    train_one_epoch(model, batches, optimizer, torch.device('cuda'), cfg, 1)
                else:
                    evaluate(model, batches, torch.device('cuda'), cfg)
            torch.cuda.synchronize()
            return (time.perf_counter()-start)/len(batches)
        timed(True); timed(False)
        train_s, eval_s = timed(True), timed(False)
        counts = {s: math.ceil(len(ds)/policy['batch_size']) for s, ds in [('train',train),('val',val),('test',test)]}
        raw = policy['epochs']*counts['train']*train_s + math.ceil(policy['epochs']/policy['val_epoch'])*counts['val']*eval_s + counts['test']*eval_s
        seconds = raw*policy['timing_safety_factor']+30
        timings[dataset] = {'seconds_per_run': seconds, 'train_batch_seconds': train_s, 'eval_batch_seconds': eval_s, 'steps': counts}
        print(f'[timing] {dataset}: estimated {seconds/60:.1f} min/run (safety included)', flush=True)
        del model, optimizer, train, val, test, train_batches, val_batches
        torch.cuda.empty_cache()
    write_json(suite/'timing.json', timings)
    return timings


def validate(policy):
    for key in ('epochs','val_epoch','batch_size','train_windows','cpu_threads'):
        if not isinstance(policy[key], int) or policy[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    expected = {'A00':('uniform','uniform'), 'A10':('learned','uniform'), 'A01':('uniform','learned'), 'A11':('learned','learned')}
    if set(policy['variants']) != set(expected):
        raise ValueError('All four predeclared variants are required')
    for key, pair in expected.items():
        if policy['variants'][key] != dict(zip(('aggregation_mode','completion_mode'),pair)):
            raise ValueError('Only the two routing modes may differ across variants')
    if not policy['datasets'] or len(set(policy['datasets'])) != len(policy['datasets']) or any(d not in SPECS for d in policy['datasets']):
        raise ValueError('Invalid datasets')
    if not policy['patterns'] or len(set(policy['patterns'])) != len(policy['patterns']) or any(p not in ('fixed','random') for p in policy['patterns']):
        raise ValueError('Invalid patterns')
    if not 0 < policy['rate'] < 1 or not math.isfinite(policy['timing_safety_factor']) or policy['timing_safety_factor'] < 1:
        raise ValueError('Invalid rate/timing safety factor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/presets/dual_moe_comparison.json')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--deadline', help='Local YYYY-MM-DD HH:MM; admission check, never truncate a run')
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error('Use one GPU index; this runner deliberately does not use dual GPU')
    policy = load(resolve(args.config)); validate(policy)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONUNBUFFERED'] = '1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(policy['cpu_threads'])
    record = identity(policy)
    suite = resolve(policy['output_dir'])/digest(record)[:16]
    deadline = datetime.strptime(args.deadline, '%Y-%m-%d %H:%M') if args.deadline else None
    print(f'[suite] {suite}\n[budget] epochs={policy["epochs"]}, train≤{policy["train_windows"]}; full val/test', flush=True)
    if args.dry_run:
        for d in policy['datasets']:
            for p in policy['patterns']:
                print(f'{d} {p}@{policy["rate"]}: '+', '.join(policy['variants']))
        return
    # OS lock prevents two terminals launching duplicate jobs for this suite.
    import fcntl
    suite.mkdir(parents=True, exist_ok=True)
    with (suite/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(suite/'protocol.json', record)
        prepared = {d:suite/'data'/d for d in policy['datasets']} if args.summary_only else prepare(policy,suite)
        jobs = []
        for dataset in policy['datasets']:
            for pattern in policy['patterns']:
                for variant in policy['variants']:
                    cfg, paths = job_config(policy,suite,prepared[dataset],dataset,pattern,variant)
                    jobs.append((dataset,pattern,variant,cfg,paths))
        if args.calibrate:
            timing = calibration(policy,suite,prepared)
        elif deadline and not args.summary_only:
            timing = calibration(policy,suite,prepared)
        elif (suite/'timing.json').exists():
            timing = load(suite/'timing.json')
        else:
            timing = None
        remaining = [j for j in jobs if not completed_run(j[3],j[2])]
        if timing:
            seconds = sum(timing[j[0]]['seconds_per_run'] for j in remaining)
            eta = datetime.now()+timedelta(seconds=seconds)
            print(f'[ETA] remaining={len(remaining)}, estimated {seconds/3600:.2f} h, finish {eta:%Y-%m-%d %H:%M} (not guaranteed)', flush=True)
            if deadline and eta > deadline and not args.summary_only:
                raise SystemExit('Budget exceeds deadline: reduce the SAME epochs/train_windows in JSON for ALL four groups, then recalibrate.')
        if args.calibrate:
            return
        rows = []
        for i,(dataset,pattern,variant,cfg,paths) in enumerate(jobs,1):
            result = completed_run(cfg,variant)
            key = f'{dataset}_{pattern}_{variant}'
            if result:
                print(f'[{i}/{len(jobs)}] SKIP complete {key}', flush=True)
            elif not args.summary_only:
                if deadline and datetime.now() >= deadline:
                    print('[deadline] no new jobs; incomplete jobs are NOT scores', flush=True)
                    break
                path = suite/'configs'/f'{key}.json'; write_json(path,cfg)
                cmd = [sys.executable,'-u',str(ROOT/'scripts/train.py'),'-c',str(path),'--name',f'ablation_{variant}','--no_plot','--quiet']
                for split, file in paths.items():
                    cmd.extend([f'--{split}_npz',str(file)])
                print(f'[{i}/{len(jobs)}] RUN {key}', flush=True)
                raw = suite/'logs'/f'{key}.log'; raw.parent.mkdir(parents=True,exist_ok=True)
                with raw.open('a') as log:
                    # Capture both streams, retain raw output, print compact epoch
                    # summaries/errors live rather than flooding 24 jobs' bars.
                    process = subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                    try:
                        for line in process.stdout:
                            log.write(line); log.flush()
                            if line.strip() and not line.lstrip().startswith(('train epoch ', 'val epoch ', 'test best epoch ')):
                                print(line,end='',flush=True)
                        code = process.wait()
                    except BaseException:
                        process.terminate(); process.wait()
                        raise
                result = completed_run(cfg,variant)
                if code or not result:
                    print(f'[FAILED] {key}: exit={code}; see {raw}',flush=True)
                    result = None
            rows.append({'dataset':dataset,'pattern':pattern,'rate':policy['rate'],'variant':variant,'status':'complete' if result else 'incomplete',**(result or {})})
            write_json(suite/'summary.json',rows)
        print(f'[summary] {suite/"summary.json"}',flush=True)
        if len(rows) != len(jobs) or any(r['status'] != 'complete' for r in rows):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
