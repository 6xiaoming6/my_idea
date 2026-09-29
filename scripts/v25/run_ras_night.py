#!/usr/bin/env python3
"""Frozen, sequential ten-arm RAS-CoE night queue with resumable receipts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stmoe_imputer.config import deep_update  # noqa: E402
from run_ablations import SOURCES, _code_inputs, _digest, _file_digest  # noqa: E402

CONFIG_DIR = ROOT / 'configs/v25/ras_night'
SUITE = ROOT / 'outputs/v25-RAS/night_20260928_bs32'
FAMILIES = ('random_point', 'node_outage', 'temporal_gap', 'spatial_region',
            'spatiotemporal_block', 'stripe', 'moving_region', 'multi_block', 'composite')
REPAIR_KEYS = ('initial_mae', *[f'step{s}_{field}' for s in range(1, 5) for field in (
    'mae', 'candidate_harm_rate', 'accepted_harm_rate', 'overrepair_prevention_rate',
    'repair_accept_mean', 'pred_accept_rate', 'oracle_accept_rate', 'nonworse_sample_rate',
    'oracle_selective_mae', 'acceptance_oracle_gap')])
SUMMARY_FIELDS = ('experiment_id', 'name', 'status', 'best_epoch', 'best_val_mae',
                  'test_mae', 'test_rmse', 'test_wape', 'test_mape',
                  'initial_mae', *[f'step{s}_{field}' for s in range(1, 5) for field in (
                      'mae', 'candidate_harm_rate', 'accepted_harm_rate',
                      'overrepair_prevention_rate', 'repair_accept_mean', 'pred_accept_rate',
                      'oracle_accept_rate', 'nonworse_sample_rate')],
                  'accept_accuracy', 'accept_precision', 'accept_recall', 'accept_f1',
                  'all_steps_monotonic_sample_rate', *[f'family_{f}_mae' for f in FAMILIES],
                  'params', 'completed_epochs', 'train_time_min', 'avg_epoch_time_sec',
                  'forward_latency_ms', 'peak_memory_gb', 'run_dir')


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def git_metadata() -> dict:
    def command(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return {'git_commit': command('git', 'rev-parse', 'HEAD'),
            'git_dirty_status': command('git', 'status', '--short')}


def plan() -> list[dict]:
    base = json.loads((CONFIG_DIR / 'base_taxibj.json').read_text())
    manifest = json.loads((CONFIG_DIR / 'experiments.json').read_text())
    expected = [f'E{i}' for i in range(10)]
    if [item['id'] for item in manifest] != expected:
        raise ValueError('Night experiments must be ordered E0 through E9')
    jobs = []
    for item in manifest:
        cfg = deep_update(base, json.loads((CONFIG_DIR / item['patch']).read_text()))
        cfg['seed'] = 7
        cfg['output_dir'] = str(Path('outputs/v25-RAS/night_20260928_bs32') / item['directory'] / 'attempts' / 'ACTIVE')
        cfg['experiment_plan'] = {'suite': 'ras_night_20260928_bs32', 'id': item['id'], 'name': item['name']}
        coe, train, data, loss = cfg['model']['coe'], cfg['train'], cfg['data'], cfg['loss']
        assert (cfg['model']['architecture'], cfg['model']['main']['use_multiscale']) == ('v25_ras_coe', False)
        assert (coe['num_steps'], len(coe['expert_pool']), coe['top_k'], coe['expert_sharing']) == (4, 8, 2, 'shared')
        assert (data['batch_size'], train['epochs'], train['val_epoch']) == (32, 100, 2 if item['id'] == 'E0' else 5)
        assert (train['scheduler']['type'], train['scheduler']['total_epochs']) == ('cosine', 100)
        assert train['best_checkpoint_name'] == 'best.pt' and train['save_best_checkpoint'] and train['full_checkpoint']
        assert loss['lambda_coe_monotonic'] == 0 and not train['early_stopping']['enabled']
        assert data['train_mask_diversity']['families'] == list(FAMILIES)
        assert data['eval_mask_diversity']['families'] == list(FAMILIES)
        assert data['train_mask_diversity']['rates'] == data['eval_mask_diversity']['rates'] == [0.4]
        jobs.append({**item, 'config': cfg})
    return jobs


def inputs() -> dict[str, str]:
    paths = _code_inputs() + sorted(CONFIG_DIR.glob('*.json')) + list(SOURCES['taxibj'].values())
    paths.append(ROOT / 'data/TaxiBJ/v24_clean_abc_20260917/manifest.json')
    paths.append(ROOT / 'model_designs/v25_RAS-CoE_10_experiments_night_spec.md')
    return {str(p.relative_to(ROOT)): _file_digest(p) for p in sorted(set(paths))}


def concrete_config(job: dict, attempt: int) -> dict:
    cfg = json.loads(json.dumps(job['config']))
    cfg['output_dir'] = str(Path('outputs/v25-RAS/night_20260928_bs32') / job['directory'] /
                            'attempts' / f'attempt{attempt}')
    return cfg


def valid_receipt(path: Path, cfg: dict) -> bool:
    try:
        receipt = json.loads(path.read_text())
        run_dir = Path(receipt['run_dir'])
        if (receipt.get('status') != 'finished' or receipt.get('completed_epochs') != 100 or
                receipt.get('config_sha256') != _digest(cfg) or
                json.loads((run_dir / 'config.json').read_text()) != cfg or
                not (run_dir / 'checkpoints/best.pt').is_file()):
            return False
        rows = [json.loads(line) for line in (run_dir / 'logs/metrics.jsonl').read_text().splitlines()]
        epochs = [row for row in rows if 'epoch' in row]
        vals = [row['epoch'] for row in epochs if row.get('val') is not None]
        tests = [row for row in rows if row.get('stage') == 'test']
        return (len(epochs) == 100 and [row['epoch'] for row in epochs] == list(range(1, 101)) and
                vals == [e for e in range(1, 101) if e % cfg['train']['val_epoch'] == 0 or e == 100] and len(tests) == 1 and
                all(math.isfinite(float(receipt['test'][key])) for key in ('mae', 'rmse', 'wape')))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def complete(job: dict) -> dict | None:
    group = SUITE / job['directory']
    path = group / 'receipt.json'
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text())
        cfg = concrete_config(job, int(receipt['attempt']))
        required = [group / 'summary.json', group / 'analysis/repair_metrics.json',
                    group / 'analysis/routing_metrics.json', group / 'analysis/family_metrics.json',
                    group / 'logs/train.log', group / 'logs/val.log', group / 'logs/test.log',
                    group / 'logs/metrics.jsonl', group / 'checkpoints/best.pt']
        return receipt if valid_receipt(path, cfg) and all(p.is_file() for p in required) else None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def next_attempt(job: dict) -> int:
    attempts = SUITE / job['directory'] / 'attempts'
    return 1 + max((int(p.name[7:]) for p in attempts.glob('attempt[0-9]*')
                    if p.name[7:].isdigit()), default=0)


def link_outputs(group: Path, run_dir: Path) -> None:
    for folder, names in {'logs': ('train.log', 'val.log', 'test.log', 'metrics.jsonl'),
                          'checkpoints': ('best.pt',)}.items():
        destination = group / folder
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = run_dir / folder / name
            link = destination / name
            if link.is_symlink() or link.is_file():
                link.unlink()
            link.symlink_to(os.path.relpath(target, destination))


def analysis_files(group: Path, receipt: dict) -> None:
    metrics = receipt['test']
    repair_prefixes = ('coe_initial_', 'coe_step', 'coe_all_steps_', 'coe_accept_', 'coe_repair_')
    repair = {k: v for k, v in metrics.items() if k.startswith(repair_prefixes)}
    for s in range(1, 5):
        repair.setdefault(f'coe_step{s}_overrepair_prevention_rate', None)
    routing = {k: v for k, v in metrics.items() if any(token in k for token in
               ('route_', 'expert_', 'pair_', 'partner_', 'rejection'))}
    families = {family: metrics.get(f'coe_family_{family}_mae') for family in FAMILIES}
    save_json(group / 'analysis/repair_metrics.json', repair)
    save_json(group / 'analysis/routing_metrics.json', routing)
    save_json(group / 'analysis/family_metrics.json', families)


def row_for(job: dict, receipt_override: dict | None = None) -> dict:
    receipt = receipt_override if receipt_override is not None else complete(job)
    row = {field: None for field in SUMMARY_FIELDS}
    row.update(experiment_id=job['id'], name=job['name'],
               status='complete' if receipt else 'failed' if (SUITE/job['directory']/'failure_receipt.json').is_file() else 'pending')
    if receipt is None:
        return row
    test = receipt['test']
    row.update(best_epoch=receipt['best_epoch'], best_val_mae=receipt['best_val_mae'],
               test_mae=test.get('mae'), test_rmse=test.get('rmse'),
               test_wape=test.get('wape'), test_mape=test.get('mape'),
               params=receipt.get('total_params'), completed_epochs=receipt['completed_epochs'],
               train_time_min=receipt.get('total_time_sec', 0) / 60,
               forward_latency_ms=test.get('forward_ms_per_batch_per_rank'),
               peak_memory_gb=receipt.get('peak_memory_gb'), run_dir=receipt['run_dir'])
    run_dir = Path(receipt['run_dir'])
    epochs = [json.loads(line) for line in (run_dir/'logs/metrics.jsonl').read_text().splitlines()]
    perfs = [r.get('perf', {}) for r in epochs if 'epoch' in r]
    train_seconds = sum(p.get('train_time_sec', 0) for p in perfs)
    row['avg_epoch_time_sec'] = sum(p.get('epoch_time_sec', 0) for p in perfs) / len(perfs)
    row['train_time_min'] = train_seconds / 60
    for key in REPAIR_KEYS:
        field = key
        metric = 'coe_' + key
        if key.endswith('repair_accept_mean'):
            field = key.replace('repair_accept_mean', 'repair_accept_mean')
        if field in row:
            row[field] = test.get(metric)
    for name in ('accuracy', 'precision', 'recall', 'f1'):
        row['accept_' + name] = test.get('coe_accept_' + name)
    row['all_steps_monotonic_sample_rate'] = test.get('coe_all_steps_monotonic_sample_rate')
    for family in FAMILIES:
        row[f'family_{family}_mae'] = test.get(f'coe_family_{family}_mae')
    return row


def fmt(value: object) -> str:
    if value is None or value == '':
        return '—'
    if isinstance(value, (int, float)):
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    return str(value)


def markdown_table(columns: tuple[str, ...], rows: list[dict]) -> list[str]:
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row.get(column)) for column in columns) + ' |')
    return lines


def summarize(jobs: list[dict]) -> list[dict]:
    SUITE.mkdir(parents=True, exist_ok=True)
    rows = [row_for(job) for job in jobs]
    temp = SUITE / 'summary.csv.tmp'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temp, SUITE / 'summary.csv')
    by_id = {row['experiment_id']: row for row in rows}
    lines = ['# RAS-CoE ten-arm exploratory comparison', '',
             'Single seed (7); differences are descriptive and do not establish statistical significance.', '']
    sections = (
        ('Main results', ('experiment_id', 'status', 'best_val_mae', 'test_mae', 'test_rmse', 'best_epoch')),
        ('Round MAE', ('experiment_id', 'initial_mae', 'step1_mae', 'step2_mae', 'step3_mae', 'step4_mae')),
        ('Over-repair', ('experiment_id', *[f'step{s}_{kind}' for s in range(1, 5)
                         for kind in ('candidate_harm_rate', 'accepted_harm_rate', 'overrepair_prevention_rate')])),
        ('Acceptance quality', ('experiment_id', 'step1_pred_accept_rate', 'step1_oracle_accept_rate',
                                'accept_precision', 'accept_recall', 'accept_f1')),
        ('Mask families', ('experiment_id', *[f'family_{f}_mae' for f in FAMILIES])),
    )
    for title, columns in sections:
        lines.extend([f'## {title}', '', *markdown_table(columns, rows), ''])
    lines.extend(['## Pairwise differences (second minus first)', ''])
    pair_rows = []
    for left, right in (('E0','E1'),('E1','E2'),('E2','E3'),('E3','E4'),('E4','E5'),
                        ('E5','E6'),('E5','E7'),('E8','E9'),('E0','E5')):
        a,b = by_id[left], by_id[right]
        result={'comparison':f'{right} - {left}'}
        for key in ('best_val_mae','test_mae','test_rmse','all_steps_monotonic_sample_rate',
                    *[f'step{s}_accepted_harm_rate' for s in range(1,5)]):
            result[key] = b[key] - a[key] if a[key] is not None and b[key] is not None else None
        pair_rows.append(result)
    lines.extend(markdown_table(tuple(pair_rows[0]), pair_rows))
    lines.extend(['', 'All tests use the best validation checkpoint once; no test metric selects a checkpoint.', ''])
    (SUITE/'comparison.md').write_text('\n'.join(lines))
    return rows


def status(jobs: list[dict], current: str | None, started_at: str) -> None:
    rows = summarize(jobs)
    save_json(SUITE/'night_status.json', {
        'started_at': started_at, 'updated_at': now(),
        'finished_at': now() if current is None and all(r['status'] != 'pending' for r in rows) else None,
        'experiments_total': len(jobs),
        'experiments_completed': sum(r['status'] == 'complete' for r in rows),
        'experiments_failed': sum(r['status'] == 'failed' for r in rows),
        'current_experiment': current,
        'results': {r['experiment_id']: r['status'] for r in rows},
    })


def gpu_busy() -> bool:
    process = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True)
    return bool(process.stdout.strip())


def run_one(job: dict, gpus: list[int]) -> None:
    group = SUITE / job['directory']
    group.mkdir(parents=True, exist_ok=True)
    attempt = next_attempt(job)
    cfg = concrete_config(job, attempt)
    save_json(group/'config.json', cfg)
    attempt_dir = group/'attempts'/f'attempt{attempt}'
    attempt_dir.mkdir(parents=True, exist_ok=True)
    config_path = attempt_dir/'input_config.json'
    raw_receipt = attempt_dir/'receipt.json'
    save_json(config_path, cfg)
    source_args = [argument for split,path in SOURCES['taxibj'].items()
                   for argument in (f'--{split}_npz',str(path))]
    train_args = ['scripts/v25/train.py', '-c',str(config_path),'--name',job['name'],
                  '--no_plot','--result-file',str(raw_receipt),*source_args]
    command = ([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node=2',*train_args] if len(gpus)==2 else
               [sys.executable,'-u',*train_args])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
             PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    log_path=attempt_dir/'launcher.log'
    try:
        with log_path.open('wb') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
            try:
                while chunk := process.stdout.read1(4096):
                    stream.write(chunk);stream.flush()
                    sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
                code=process.wait()
            except BaseException:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
                raise
        if code:
            raise RuntimeError(f'{job["id"]} exited {code}; see {log_path}')
        if not valid_receipt(raw_receipt,cfg):
            raise RuntimeError(f'{job["id"]} returned incomplete receipt; see {log_path}')
        receipt=json.loads(raw_receipt.read_text())
        receipt.update(experiment_id=job['id'],name=job['name'],attempt=attempt,
                       completed_at=now(),config_path=str(config_path))
        save_json(group/'receipt.json',receipt)
        run_dir=Path(receipt['run_dir'])
        link_outputs(group,run_dir)
        analysis_files(group,receipt)
        row=row_for(job,receipt)
        save_json(group/'summary.json',row)
        failure=group/'failure_receipt.json'
        if failure.exists(): failure.unlink()
        print(f'COMPLETE {job["id"]}: test MAE={receipt["test"]["mae"]:.4f}',flush=True)
    except BaseException as exc:
        if isinstance(exc,KeyboardInterrupt): raise
        save_json(group/'failure_receipt.json', {
            'experiment_id':job['id'],'timestamp':now(),'exception_type':type(exc).__name__,
            'exception':str(exc),'traceback':traceback.format_exc(),'config_path':str(config_path),
            'launcher_log':str(log_path),'attempt':attempt,**git_metadata()})
        print(f'FAILED {job["id"]}: {exc}',file=sys.stderr,flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',choices=[f'E{i}' for i in range(10)],default='E0')
    parser.add_argument('--only',nargs='+',choices=[f'E{i}' for i in range(10)])
    parser.add_argument('--force',action='store_true',help='rerun selected completed arms in new attempts')
    parser.add_argument('--gpus',default='0',help='one or two physical GPU indices')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--no-wait',action='store_true',help='fail if another GPU job is active')
    args=parser.parse_args()
    gpus=[int(item) for item in args.gpus.split(',')]
    if len(gpus) not in (1,2) or len(gpus)!=len(set(gpus)) or min(gpus)<0:
        parser.error('pass one or two distinct nonnegative GPU indices')
    jobs=plan()
    selected=[j for j in jobs if int(j['id'][1:])>=int(args.start[1:]) and
              (args.only is None or j['id'] in args.only)]
    if args.dry_run:
        print(json.dumps({'suite':str(SUITE),'selected':[{'id':j['id'],'name':j['name'],
                          'directory':j['directory']} for j in selected],
                          'epochs':100,'scheduler_total':100,'global_batch':32,
                          'gpu_count':len(gpus)},indent=2))
        return
    if args.summary_only:
        if not SUITE.is_dir(): raise FileNotFoundError(SUITE)
        summarize(jobs)
        print(SUITE/'summary.csv')
        return
    fingerprint=inputs()
    SUITE.mkdir(parents=True,exist_ok=True)
    frozen=SUITE/'input_hashes.json'
    if frozen.exists() and json.loads(frozen.read_text())!=fingerprint:
        raise RuntimeError('Night-run source/config/data changed since queue creation')
    save_json(frozen,fingerprint)
    lock=SUITE/'night_queue.lock'
    with lock.open('a') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another RAS night queue is active') from None
        prior=SUITE/'night_status.json'
        started_at=json.loads(prior.read_text()).get('started_at',now()) if prior.exists() else now()
        status(jobs,None,started_at)
        for job in selected:
            if not args.force and complete(job):
                print(f'SKIP_COMPLETED {job["id"]}',flush=True)
                continue
            if (SUITE/job['directory']).is_dir():
                print(f'INCOMPLETE {job["id"]}: launching a new attempt',flush=True)
            while gpu_busy():
                if args.no_wait: raise RuntimeError('GPU compute process already active')
                print('WAIT_GPU: current GPU job is still running; checking again in 30 s',flush=True)
                time.sleep(30)
            for relative,expected in fingerprint.items():
                if _file_digest(ROOT/relative)!=expected:
                    raise RuntimeError(f'Night-run input changed: {relative}')
            status(jobs,job['id'],started_at)
            run_one(job,gpus)
            status(jobs,None,started_at)
        status(jobs,None,started_at)
    print(SUITE/'summary.csv',flush=True)


if __name__=='__main__':
    main()
