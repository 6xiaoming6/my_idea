#!/usr/bin/env python3
"""V22.1 controlled comparison; reuse original runs, isolate all new artifacts."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile

from train import ROOT, build_config, load_suite, resolve
from run_experiments import completed


def extension_identity():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ('refinement_model.py', 'run_refinement.py')}


def load_policy(path):
    policy = json.loads(resolve(path).read_text())
    if policy['epochs'] < 1 or not policy['seeds'] or len(set(policy['seeds'])) != len(policy['seeds']):
        raise ValueError('Positive epoch budget and distinct nonempty seeds required')
    if set(policy['variants']) & set(policy['baselines']):
        raise ValueError('Baseline and candidate names must not overlap')
    if policy['baselines'] != ['uniform', 'moe']:
        raise ValueError('This comparison requires the original uniform and moe baselines')
    for options in policy['variants'].values():
        if not 0 < options['initial_strength'] < options['max_strength'] < 1:
            raise ValueError('Invalid routing strength limits')
        if not 0 <= options['fallback_probability'] < 1:
            raise ValueError('Invalid fallback probability')
    return policy, load_suite(policy['suite'], policy['profile'])


def refined_config(policy, suite, variant, dataset, pattern, rate, seed, world_size=2):
    if variant in policy['baselines']:
        return build_config(suite, variant, dataset, pattern, rate, seed,
                            policy['epochs'], world_size=world_size)
    suite = copy.deepcopy(suite)
    suite['common']['output_dir'] = policy['output_dir']
    suite['variants'][variant] = {'model': {'version': 'v22.1', 'v22': {
        'mode': 'moe', 'refinement': policy['variants'][variant],
        'refinement_source': extension_identity(),
    }}}
    return build_config(suite, variant, dataset, pattern, rate, seed,
                        policy['epochs'], world_size=world_size)


def jobs(policy, suite, devices, variants=None):
    for dataset, pattern, rate in policy['points']:
        for variant in policy['baselines'] + list(variants or policy['variants']):
            for seed in policy['seeds']:
                cfg, paths = refined_config(policy, suite, variant, dataset, pattern, str(rate), seed, len(devices))
                yield variant, dataset, pattern, str(rate), seed, cfg, paths


def worker():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', required=True)
    args, _ = parser.parse_known_args(sys.argv[2:])
    cfg = json.loads(Path(args.config).read_text())
    if cfg['model']['v22'].get('refinement_source') != extension_identity():
        raise RuntimeError('Refinement implementation changed after configuration creation')
    sys.path.insert(0, str(ROOT / 'src'))
    from refinement_model import install_refinement_builder
    install_refinement_builder()
    import train_ddp
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    train_ddp.main()


def launch(job, args, interpreter, policy):
    variant, dataset, pattern, rate, seed, cfg, paths = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(args.gpus), PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(args.cpu_threads)
    if variant in policy['baselines']:
        command = interpreter + ['scripts/v22/train.py', '--config', str(resolve(policy['suite'])),
                    '--profile', policy['profile'], '--variant', variant, '--dataset', dataset,
                    '--mask', pattern, '--rate', rate, '--seed', str(seed), '--epochs', str(policy['epochs']),
                    '--gpus', *args.gpus, '--cpu-threads', str(args.cpu_threads), '--conda-env', 'current']
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    for path in [*paths.values(), *[cfg['data']['mask'][f'{split}_csv'] for split in paths]]:
        if not resolve(path).is_file():
            raise FileNotFoundError(f'Required real data/mask missing: {resolve(path)}')
    with tempfile.TemporaryDirectory(prefix='v221_') as directory:
        path = Path(directory) / 'config.json'
        path.write_text(json.dumps(cfg, indent=2))
        command = interpreter + ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                  f'--nproc_per_node={len(args.gpus)}', '--max_restarts=0', str(Path(__file__).resolve()),
                  '--worker', '-c', str(path), '--name', f'ablation_v22_{variant}', '--no_plot', '--quiet']
        command += [a for split, p in paths.items() for a in (f'--{split}_npz', p)]
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def stat(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None} if values else None


def summary(rows):
    groups, pairs = [], []
    points = sorted({(r['dataset'], r['pattern'], r['rate']) for r in rows})
    variants = list(dict.fromkeys(r['variant'] for r in rows))
    for dataset, pattern, rate in points:
        lookup = {(r['variant'], r['seed']): r for r in rows if r['status'] == 'complete' and
                  (r['dataset'], r['pattern'], r['rate']) == (dataset, pattern, rate)}
        for variant in variants:
            subset = [r for (v, _), r in lookup.items() if v == variant]
            groups.append(dict(dataset=dataset, pattern=pattern, rate=rate, variant=variant,
                               seeds=[r['seed'] for r in subset],
                               **{k: stat([r[k] for r in subset]) for k in ('val_mae', 'test_mae', 'test_rmse')}))
            if variant in ('uniform', 'moe'):
                continue
            references = ['uniform', 'moe'] + (['bounded'] if variant == 'bounded_fallback' and 'bounded' in variants else [])
            for ref in references:
                seeds = sorted(s for v, s in lookup if v == variant and (ref, s) in lookup)
                pairs.append(dict(dataset=dataset, pattern=pattern, rate=rate, candidate=variant, reference=ref,
                                  seeds=seeds, negative_is_better=True,
                                  **{k: stat([lookup[variant, z][k] - lookup[ref, z][k] for z in seeds])
                                     for k in ('val_mae', 'test_mae', 'test_rmse')}))
    return dict(runs=rows, groups=groups, paired_differences=pairs,
                note='Same-seed comparisons only. Sample SD, not significance. Candidates are selected by validation, not test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v22/refinement.json')
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--variants', nargs='+', help='Subset of refinement candidates; baselines are always included')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--conda-env', default='difftdi')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary', action='store_true', help='Read-only JSON summary; does not train')
    parser.add_argument('--save-summary', action='store_true', help='Also save summary JSON/.log; implies --summary')
    args = parser.parse_args()
    if len(args.gpus) < 2 or len(set(args.gpus)) != len(args.gpus) or any(not x.isdigit() for x in args.gpus) or args.cpu_threads < 1:
        parser.error('Use at least two distinct numeric GPUs and positive CPU threads; preserve DDP protocol')
    policy, suite = load_policy(args.config)
    if args.variants and any(v not in policy['variants'] for v in args.variants):
        parser.error(f'Unknown candidate; choices: {list(policy["variants"])}')
    work = list(jobs(policy, suite, args.gpus, list(dict.fromkeys(args.variants)) if args.variants else None))
    records = [completed(job[-2], job[0]) for job in work]
    if args.summary or args.save_summary:
        rows = [dict(variant=j[0], dataset=j[1], pattern=j[2], rate=j[3], seed=j[4],
                     status='complete' if old else 'missing_or_incompatible', **(old or {})) for j, old in zip(work, records)]
        payload = json.dumps(summary(rows), indent=2, allow_nan=False)
        print(payload)
        if args.save_summary and not args.dry_run:
            output = resolve(policy['output_dir'])
            output.mkdir(parents=True, exist_ok=True)
            path = output / f'comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
            path.write_text(payload)
            path.with_suffix('.log').write_text(payload + '\n')
            print(f'Saved {path}', file=sys.stderr)
        return
    pending = sum(old is None for old in records)
    print(f'[plan V22.1] total={len(work)} skip={len(work)-pending} pending={pending}; '
          f'epochs={policy["epochs"]}; pending_epoch_budget={pending*policy["epochs"]}', flush=True)
    interpreter = [sys.executable] if args.conda_env == 'current' else ['conda', 'run', '--no-capture-output', '-n', args.conda_env, 'python']
    for job, old in zip(work, records):
        name, dataset, pattern, rate, seed, _, _ = job
        label = f'{name} {dataset} {pattern}@{rate} seed={seed}'
        if old:
            print(f'[SKIP] {label}: {old["run"]}', flush=True)
        else:
            print(f'[{"WOULD RUN" if args.dry_run else "RUN"}] {label}', flush=True)
            if not args.dry_run:
                launch(job, args, interpreter, policy)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker()
    else:
        main()
