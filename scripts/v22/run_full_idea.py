#!/usr/bin/env python3
"""Isolated full Idea-2 training. Single GPU by default; explicit optional DDP.

Reuse the audited V22 engine, including exact test sharding and best-only saves.
No automatic GPU power changes, no launching training during --summary/--dry-run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from train import ROOT, build_config, load_suite, merge, resolve
from run_experiments import completed


def source_identity():
    return {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('full_model.py', 'run_full_idea.py')}


def configuration(policy, variant, dataset, pattern, rate, seed, world=1, cpu=False, smoke=False):
    if variant not in policy['variants'] or world < 1:
        raise ValueError('Unknown variant or invalid world size')
    suite = copy.deepcopy(load_suite(policy['suite']))
    training = policy['train']
    if int(training['epochs']) < 1 or int(training['val_epoch']) < 1 or int(policy['batch_size']) < 1:
        raise ValueError('epochs, val_epoch and batch_size must be positive')
    model_options = merge(policy['model'], policy['variants'][variant])
    if smoke:
        model_options.update(dim=8, num_groups=2, dropout=0.)
    suite['common'] = merge(suite['common'], {
        'output_dir': policy['output_dir'],
        'data': {'batch_size': policy['batch_size']},
        'model': {'version': 'v22-full-idea', 'v22': {
            'full_idea': model_options, 'full_source': source_identity(),
        }},
    })
    for entry in suite['datasets'].values():
        entry['train'] = merge(entry['train'], training)
    suite['distributed']['per_rank_batch_size'] = policy['batch_size']
    label = 'full_idea_' + variant
    suite['variants'][label] = {}
    # The same audited distributed engine also runs a one-rank job.
    cfg, paths = build_config(suite, label, dataset, pattern, str(rate), seed,
                             cpu=cpu, smoke=smoke, world_size=max(2, world))
    cfg['distributed'].update(world_size=world, global_batch_size=world*cfg['data']['batch_size'])
    if smoke:
        cfg['output_dir'] = 'outputs/v22/full_idea/smoke'
        cfg['data']['synthetic'].update(num_train=3, num_val=3, t=2, h=8, w=8)
    # build_config hashes original engine+src+data; also include FINAL one-rank /
    # smoke overrides and the isolated full-model implementation hashes.
    cfg['experiment_policy']['fingerprint'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cfg['experiment_policy']['model_implementation'] = 'scripts/v22/full_model.py:FullCoarseningMoE'
    return cfg, paths


def worker():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', required=True)
    args, _ = parser.parse_known_args(sys.argv[2:])
    cfg = json.loads(Path(args.config).read_text())
    if cfg['model']['v22'].get('full_source') != source_identity():
        raise RuntimeError('Full Idea-2 source changed after configuration creation')
    sys.path.insert(0, str(ROOT/'src'))
    from full_model import install_full_builder
    install_full_builder()
    import train_ddp
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    train_ddp.main()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/v22/full_idea.json')
    devices = p.add_mutually_exclusive_group()
    devices.add_argument('--gpu', default='0')
    devices.add_argument('--gpus', nargs='+')
    p.add_argument('--datasets', nargs='+', choices=['BikeNYC', 'TaxiBJ', 'CHAP'])
    p.add_argument('--patterns', nargs='+', choices=['fixed', 'random'])
    p.add_argument('--rates', nargs='+', choices=['0.2', '0.4', '0.6', '0.8'])
    p.add_argument('--variants', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Tiny synthetic pipeline check, never real-data evidence')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--conda-env', default='difftdi')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()
    gpus = args.gpus or [args.gpu]
    if args.cpu_threads < 1 or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        p.error('Use distinct numeric GPUs and positive CPU threads')
    policy = json.loads(resolve(args.config).read_text())
    if args.epochs is not None:
        policy['train']['epochs'] = args.epochs
    points = policy['points']
    if args.rates:
        points = [(d, m, r) for d, m in dict.fromkeys((d, m) for d, m, _ in points) for r in dict.fromkeys(args.rates)]
    points = [x for x in points if (not args.datasets or x[0] in args.datasets)
              and (not args.patterns or x[1] in args.patterns)]
    if not points:
        p.error('No selected experiment points')
    jobs = [(v, d, m, r, seed, *configuration(policy, v, d, m, r, seed, len(gpus), args.cpu, args.smoke))
            for d, m, r in points for v in dict.fromkeys(args.variants or policy['default_variants'])
            for seed in dict.fromkeys(args.seeds or policy['seeds'])]
    interpreter = [sys.executable] if args.conda_env == 'current' else ['conda', 'run', '--no-capture-output', '-n', args.conda_env, 'python']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.cpu else ','.join(gpus),
               PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(args.cpu_threads)
    # Validate inputs for every selected job BEFORE starting a long queue.
    if not args.smoke and not (args.summary or args.dry_run):
        for *_, cfg, paths in jobs:
            for path in [*paths.values(), *[cfg['data']['mask'][f'{s}_csv'] for s in paths]]:
                if not resolve(path).is_file():
                    raise FileNotFoundError(resolve(path))
    for number, (v, d, m, r, seed, cfg, paths) in enumerate(jobs, 1):
        label = 'full_idea_' + v
        old = completed(cfg, label)
        if args.summary:
            print(json.dumps(dict(variant=v, dataset=d, pattern=m, rate=r, seed=seed,
                                  status='complete' if old else 'missing_or_incompatible', **(old or {}))))
            continue
        print(f'[{number}/{len(jobs)}] {"SKIP" if old else "PLAN" if args.dry_run else "RUN"} {v} {d} {m}@{r} seed={seed}; '
              f'epochs={cfg["train"]["epochs"]} val_epoch={cfg["train"]["val_epoch"]} ranks={len(gpus)}', flush=True)
        if old:
            print(old['run'], flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix='v22_full_') as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(cfg, indent=2))
            command = interpreter + ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                       f'--nproc_per_node={len(gpus)}', '--max_restarts=0', str(Path(__file__).resolve()),
                       '--worker', '-c', str(path), '--name', f'ablation_v22_{label}', '--no_plot', '--quiet']
            command += ['--synthetic'] if args.smoke else [a for s, value in paths.items() for a in (f'--{s}_npz', value)]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    worker() if len(sys.argv) > 1 and sys.argv[1] == '--worker' else main()
