#!/usr/bin/env python3
"""Run the six matched v25 RAS-CoE ablations sequentially."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from stmoe_imputer.config import deep_update  # noqa: E402

VARIANTS = (
    'A0_current_c2', 'A1_naive_feedback', 'A2_legacy_acceptance',
    'A3_latent_commit', 'A4_supervised_commit', 'A5_full_ras_coe',
)
SOURCES = {
    'taxibj': {split: ROOT / f'data/TaxiBJ/v24_clean_abc_20260917/taxibj_{split}.npz'
               for split in ('train', 'val', 'test')},
    'bikenyc': {split: ROOT / f'data/BikeNYC/bikenyc_{split}.npz'
                for split in ('train', 'val', 'test')},
}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _code_inputs() -> list[Path]:
    return sorted(set((ROOT / 'src/stmoe_imputer').rglob('*.py')) |
                  set((ROOT / 'src/v25_ras_coe').rglob('*.py')) |
                  set((ROOT / 'scripts/v25').glob('*.py')) |
                  {ROOT / 'scripts/train.py'})


def _plan(dataset: str, epochs: int, batch_size: int, seeds: list[int]) -> dict:
    base_path = ROOT / f'configs/v25/ras_coe_{dataset}_base.json'
    base = json.loads(base_path.read_text())
    jobs = []
    for seed in seeds:
        for variant in VARIANTS:
            patch = json.loads((ROOT / 'configs/v25' / f'{variant}.json').read_text())
            cfg = deep_update(base, patch)
            cfg['seed'] = seed
            cfg['data']['batch_size'] = batch_size
            cfg['train']['epochs'] = epochs
            cfg['train']['scheduler']['total_epochs'] = epochs
            cfg['experiment_plan'] = {'stage': 'ras_coe_ablation', 'variant': variant,
                                      'dataset': dataset, 'seed': seed}
            if cfg['model']['main'].get('use_multiscale') or cfg['data'].get('multiscale'):
                raise ValueError('v25 first version must remain single-scale')
            if cfg['model']['architecture'] != 'v25_ras_coe':
                raise ValueError('Expected v25_ras_coe architecture')
            jobs.append({'name': f'ras_{variant.lower()}_seed{seed}',
                         'variant': variant, 'seed': seed, 'config': cfg})
    paths = _code_inputs() + [base_path] + [ROOT / 'configs/v25' / f'{v}.json' for v in VARIANTS]
    paths += list(SOURCES[dataset].values())
    if dataset == 'taxibj':
        paths.append(ROOT / 'data/TaxiBJ/v24_clean_abc_20260917/manifest.json')
    hashes = {str(path.relative_to(ROOT)): _file_digest(path) for path in paths}
    identity = {'dataset': dataset, 'epochs': epochs, 'batch_size': batch_size,
                'seeds': seeds, 'jobs': jobs, 'hashes': hashes}
    fingerprint = _digest(identity)[:16]
    suite = ROOT / 'outputs/v25-RAS-CoE/experiments/ras_coe_ablation' / dataset / fingerprint
    for job in jobs:
        job['config']['output_dir'] = str(suite / 'runs')
        job['config']['experiment_suite_fingerprint'] = fingerprint
    return {'suite': str(suite), 'fingerprint': fingerprint, 'dataset': dataset,
            'epochs': epochs, 'batch_size': batch_size, 'seeds': seeds,
            'sources': {key: str(path) for key, path in SOURCES[dataset].items()},
            'input_hashes': hashes, 'jobs': jobs}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def _receipt_valid(path: Path, cfg: dict) -> bool:
    try:
        receipt = json.loads(path.read_text())
        run_dir = Path(receipt['run_dir'])
        if (receipt['status'] != 'finished' or
                receipt['completed_epochs'] != cfg['train']['epochs'] or
                receipt['config_sha256'] != _digest(cfg) or
                json.loads((run_dir / 'config.json').read_text()) != cfg or
                not (run_dir / 'checkpoints/best.pth').is_file()):
            return False
        rows = [json.loads(line) for line in (run_dir / 'logs/metrics.jsonl').read_text().splitlines()]
        epochs = [row for row in rows if 'epoch' in row]
        vals = [row['epoch'] for row in epochs if row.get('val') is not None]
        expected = [i for i in range(1, cfg['train']['epochs'] + 1)
                    if i % cfg['train']['val_epoch'] == 0 or i == cfg['train']['epochs']]
        if [row['epoch'] for row in epochs] != list(range(1, cfg['train']['epochs'] + 1)) or vals != expected:
            return False
        tests = [row for row in rows if row.get('stage') == 'test']
        return len(tests) == 1 and all(math.isfinite(float(receipt['test'][key]))
                                       for key in ('loss', 'mae', 'rmse'))
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _summarize(plan: dict) -> list[dict]:
    suite = Path(plan['suite'])
    rows = []
    for job in plan['jobs']:
        receipts = sorted((suite / 'results').glob(job['name'] + '.attempt*.json'))
        receipt = next((json.loads(path.read_text()) for path in reversed(receipts)
                        if _receipt_valid(path, job['config'])), None)
        rows.append({'name': job['name'], 'variant': job['variant'], 'seed': job['seed'],
                     'status': 'complete' if receipt else 'pending',
                     'best_epoch': receipt['best_epoch'] if receipt else '',
                     'best_val_mae': receipt['best_val_mae'] if receipt else '',
                     'test_mae': receipt['test']['mae'] if receipt else '',
                     'test_rmse': receipt['test']['rmse'] if receipt else '',
                     'seconds': receipt['total_time_sec'] if receipt else '',
                     'run_dir': receipt['run_dir'] if receipt else ''})
    with (suite / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def _launch(plan: dict, job: dict, gpu_ids: list[int]) -> None:
    suite = Path(plan['suite'])
    attempts = list((suite / 'results').glob(job['name'] + '.attempt*.json'))
    attempt = len(attempts) + 1
    config_path = suite / 'configs' / (job['name'] + '.json')
    receipt = suite / 'results' / f'{job["name"]}.attempt{attempt}.json'
    source_args = []
    for split, path in plan['sources'].items():
        source_args += [f'--{split}_npz', path]
    train_args = ['scripts/v25/train.py', '-c', str(config_path), '--name', job['name'],
                  '--no_plot', '--result-file', str(receipt), *source_args]
    if len(gpu_ids) == 2:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=2', *train_args]
    else:
        command = [sys.executable, '-u', *train_args]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids)),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    log_path = suite / 'launcher_logs' / f'{job["name"]}.attempt{attempt}.log'
    with log_path.open('wb') as raw:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while chunk := process.stdout.read1(4096):
                raw.write(chunk); raw.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if status or not _receipt_valid(receipt, job['config']):
        raise RuntimeError(f'{job["name"]} failed or produced incomplete results; see {log_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(SOURCES), default='taxibj')
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument('--gpu', type=int)
    devices.add_argument('--gpus', default='0,1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or any(seed < 0 for seed in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        parser.error('epochs must be positive and seeds must be unique nonnegative integers')
    gpu_ids = [args.gpu] if args.gpu is not None else [int(item) for item in args.gpus.split(',')]
    if len(gpu_ids) not in (1, 2) or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        parser.error('use one --gpu or two distinct --gpus indices')
    batch_size = 32 if args.batch_size is None else args.batch_size
    if batch_size < 1 or batch_size % len(gpu_ids):
        parser.error('global batch size must be positive and divisible by GPU count')
    plan = _plan(args.dataset, args.epochs, batch_size, args.seeds)
    if args.dry_run:
        print(json.dumps({'suite': plan['suite'], 'jobs': [
            {'variant': job['variant'], 'name': job['name'],
             'epochs': job['config']['train']['epochs'],
             'checkpoint': job['config']['train']['best_checkpoint_name']}
            for job in plan['jobs']], 'batch_size': batch_size}, indent=2))
        return
    suite = Path(plan['suite'])
    if args.summary_only and not suite.is_dir():
        raise FileNotFoundError(suite)
    suite.mkdir(parents=True, exist_ok=True)
    _write_json(suite / 'plan.json', plan)
    for job in plan['jobs']:
        _write_json(suite / 'configs' / (job['name'] + '.json'), job['config'])
    if args.summary_only:
        _summarize(plan)
        print(suite / 'summary.csv')
        return
    queue_lock = ROOT / 'outputs/v25-RAS-CoE/experiments/.gpu_queue.lock'
    queue_lock.parent.mkdir(parents=True, exist_ok=True)
    with queue_lock.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another v25 GPU queue is already running') from None
        if subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.strip():
            raise RuntimeError('A GPU compute process is already running; wait before launching v25')
        (suite / 'results').mkdir(exist_ok=True)
        (suite / 'launcher_logs').mkdir(exist_ok=True)
        for job in plan['jobs']:
            for relative, expected in plan['input_hashes'].items():
                path = ROOT / relative
                if not path.is_file() or _file_digest(path) != expected:
                    raise RuntimeError(f'Input changed during the v25 suite: {relative}')
            receipts = sorted((suite / 'results').glob(job['name'] + '.attempt*.json'))
            if any(_receipt_valid(path, job['config']) for path in receipts):
                continue
            _launch(plan, job, gpu_ids)
            _summarize(plan)
    print(suite / 'summary.csv')


if __name__ == '__main__':
    main()
