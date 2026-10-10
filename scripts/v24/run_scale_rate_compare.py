#!/usr/bin/env python3
"""16 independent trainings: fixed CMFF vs free three-scale Top1, two datasets/four rates."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
from run_b3_c3 import load, write, digest
from run_id_priority_exploration import static_jobs

CONFIG = ROOT / 'configs/v24/scale_rate_compare/protocol.json'


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jobs(datasets=('taxibj', 'bikenyc'), rates=(.2, .4, .6, .8), epochs=100, batch_size=32, communication='delta'):
    result = {}
    for dataset in datasets:
        origin = static_jobs(dataset, epochs, batch_size)['U02']
        for rate in rates:
            prefix = f'{dataset}_r{round(rate * 100):02}'
            for method, policy in (('cmff', 'fixed'), ('top1', 'st')):
                j = copy.deepcopy(origin)
                name = prefix + '_' + method
                c = j['config']
                c['data']['mask'].update(pattern='random', missing_rate=rate)
                for key in ('train_mask_diversity', 'eval_mask_diversity'):
                    c['data'][key]['rates'] = [rate]
                c['model']['coe']['id_priority'] = {
                    'enabled': True, 'communication': communication, 'bound': .1,
                    'scale_policy': policy, 'constraint': 'none', 'fixed_epochs': 0,
                    'deterministic_spatial_ops': True,
                    'scale_soft_warmup_epochs': 5 if method == 'top1' else 0,
                    'scale_soft_transition_epochs': 5 if method == 'top1' else 0,
                }
                c['train'].update(strict_replay=True, logging={'diagnostic_every': 5, 'path_top_k': 10})
                protocol = copy.deepcopy(load(CONFIG))
                protocol['rate'] = rate
                c['experiment_plan'] = {
                    'suite': 'scale_rate_compare', 'variant': name, 'dataset': dataset,
                    'method': method, 'rate': rate, 'reference': prefix + '_cmff',
                    'protocol': protocol, 'selection': 'ID validation MAE only',
                    'comparison': 'Train independently at each rate; paired masks and initialization',
                }
                j.update(variant=name, name='scale_' + ('soft_top1' if method == 'top1' else method), reference=prefix + '_cmff',
                         dataset=dataset, rate=rate, method=method)
                result[name] = j
    return result


def evaluation_group(job):
    return f'evaluations_rate{job["rate"]:g}'


def run_directory(job, stamp=None):
    c = job['config']
    stamp = stamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    return (Path(c['output_dir']) / c['data']['dataset_name'] / 'custom'
            / f'{stamp}_{job["name"]}_seed{c["seed"]}' / 'random'
            / f'rate{job["rate"]:g}' / f'{stamp}_seed{c["seed"]}_bs{c["data"]["batch_size"]}')


def launch(suite, job, gpu):
    if base.receipt(suite, job):
        print(f'Skip completed {job["variant"]}', flush=True)
        return
    n = job['variant']
    pointer = suite / 'runs' / f'{n}.json'
    run = Path(load(pointer)['run_dir']) if pointer.exists() else run_directory(job)
    base.frozen(pointer, {'run_dir': str(run)})
    base.frozen(suite / 'jobs' / f'{n}.json', job)
    command = [sys.executable, '-u', str(suite / 'source_snapshot/scripts/v24/train_four_direction.py'),
               '--job', str(suite / 'jobs' / f'{n}.json'), '--run-dir', str(run),
               '--result', str(suite / 'results' / f'{n}.json')]
    print(f'Starting {n} on GPU {gpu}', flush=True)
    base.subprocess_run(command, suite / 'launcher_logs' / f'{n}.train.log', gpu, visible=True)
    if not base.receipt(suite, job):
        raise RuntimeError('Training did not produce a valid receipt: ' + n)


def prepare_initialization(suite, all_jobs):
    """Same state dict for fixed/free policies; only requires_grad differs."""
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    records = {}
    for n, j in all_jobs.items():
        path = suite / 'initialization' / (j['dataset'] + '.pth')
        # One constructor per policy/dataset suffices: mask rate is not a model input config.
        key = (j['dataset'], j['method'])
        if key not in records:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(j['config']['seed'])
                model = DualBranchSTImputer.from_config(j['config'])
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sha = state_hash(state)
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=False)
                if saved['sha256'] != sha or state_hash(saved['model']) != sha:
                    raise RuntimeError('Common initialization mismatch: ' + n)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.tmp')
                torch.save({'model': state, 'sha256': sha, 'seed': j['config']['seed']}, temp)
                temp.replace(path)
            records[key] = {'sha256': sha, 'total_params': sum(p.numel() for p in model.parameters()),
                            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            del model, state
        j['common_initialization'] = str(path)
    base.frozen(suite / 'initialization_audit.json', {'/'.join(k): v for k, v in records.items()})


def export_report(suite, all_jobs):
    path = suite / 'source_snapshot/scripts/v24/report_scale_rate_compare.py'
    spec = importlib.util.spec_from_file_location('frozen_scale_rate_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_report(ROOT, suite, all_jobs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', nargs='+', choices=('taxibj', 'bikenyc'), default=['taxibj', 'bikenyc'])
    p.add_argument('--rates', nargs='+', type=float, choices=(.2, .4, .6, .8), default=[.2, .4, .6, .8])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--communication', choices=('delta', 'none'), default='delta')
    p.add_argument('--suite', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1 or a.gpu < 0 or len(set(a.dataset)) != len(a.dataset) or len(set(a.rates)) != len(a.rates):
        p.error('Invalid or duplicated arguments')
    if a.suite:
        suite = a.suite.resolve()
        payload = load(suite / 'plan.json')
        all_jobs = copy.deepcopy(payload['jobs'])
    else:
        all_jobs = jobs(a.dataset, a.rates, a.epochs, a.batch_size, a.communication)
        source = base.manifest(a.dataset[0])
        for path in (ROOT / 'tests/test_v24_scale_rate_compare.py', ROOT / 'scripts/v24/README_SCALE_RATE_COMPARE.md'):
            source[str(path.relative_to(ROOT))] = filehash(path)
        data = {}
        for j in all_jobs.values():
            for path in j['sources'].values():
                if path not in data:
                    stat = Path(path).stat()
                    data[path] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': filehash(path)}
        payload = {'jobs': all_jobs, 'order': list(all_jobs), 'source': source, 'data': data}
        suite = ROOT / 'outputs/v24-COE/experiments/scale_rate_compare' / digest(payload)[:16]
    if a.dry_run:
        print(json.dumps({'suite': str(suite), 'count': len(all_jobs), 'order': list(all_jobs), 'jobs': all_jobs}, ensure_ascii=False, indent=2))
        return
    if a.summary_only:
        print(export_report(suite, all_jobs))
        return
    suite.parent.mkdir(parents=True, exist_ok=True)
    with (suite.parent / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite.mkdir(exist_ok=True)
        for name, sha in payload['source'].items():
            target = suite / 'source_snapshot' / name
            if not target.exists():
                if a.suite:
                    raise RuntimeError('Missing frozen source: ' + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            if filehash(target) != sha:
                raise RuntimeError('Frozen source changed: ' + name)
        base.frozen(suite / 'plan.json', payload)
        # Resume orchestration from the frozen file as well as frozen training/evaluation.
        if Path(__file__).resolve() != suite / 'source_snapshot/scripts/v24/run_scale_rate_compare.py':
            import os
            command = [sys.executable, '-u', str(suite / 'source_snapshot/scripts/v24/run_scale_rate_compare.py'),
                       '--suite', str(suite), '--gpu', str(a.gpu)]
            # Parent owns queue lock until subprocess exits.
            import subprocess
            env = dict(os.environ, SCALE_RATE_FROZEN_PARENT='1')
            # The child bypasses this entrypoint's locking in frozen_main below.
            code = subprocess.call(command, env=env)
            if code:
                raise SystemExit(code)
            return
        run_queue(suite, payload, all_jobs, a.gpu)


def run_queue(suite, payload, all_jobs, gpu):
    prepare_initialization(suite, all_jobs)
    write(suite / 'resolved_jobs.json', all_jobs)
    if not (suite / 'started.json').exists():
        write(suite / 'started.json', {'time': time.time(), 'gpu': gpu})
    for n in payload['order']:
        j = all_jobs[n]
        for path, stamp in payload['data'].items():
            st = Path(path).stat()
            if (st.st_size, st.st_mtime_ns) != (stamp['size'], stamp['mtime_ns']):
                raise RuntimeError('Dataset changed: ' + path)
        base.frozen(suite / 'configs' / f'{n}.json', j['config'])
        try:
            launch(suite, j, gpu)
            r = base.receipt(suite, j)
            base.evaluate_record(suite, n, Path(r['run_dir']) / 'checkpoints/best.pth', r['config_sha256'],
                                 j['config']['experiment_plan']['protocol'], j['sources']['test'], evaluation_group(j), gpu)
        except BaseException as error:
            write(suite / 'failures' / f'{n}.json', {'error': repr(error), 'time': time.time()})
            export_report(suite, all_jobs)
            raise
        export_report(suite, all_jobs)
    write(suite / 'completed.json', {'time': time.time(), 'count': len(all_jobs)})
    print(f'Results: {suite}\nReport: {export_report(suite, all_jobs)}', flush=True)


if __name__ == '__main__':
    import os
    if os.environ.get('SCALE_RATE_FROZEN_PARENT') == '1':
        # Only the snapshot child of the lock-owning root launcher enters here.
        parser = argparse.ArgumentParser()
        parser.add_argument('--suite', type=Path, required=True)
        parser.add_argument('--gpu', type=int, required=True)
        args = parser.parse_args()
        suite = args.suite.resolve()
        payload = load(suite / 'plan.json')
        # Outputs/reports belong to the live repository, imports to the snapshot.
        ROOT = Path(payload['jobs'][payload['order'][0]]['config']['output_dir']).parents[1]
        run_queue(suite, payload, copy.deepcopy(payload['jobs']), args.gpu)
    else:
        main()
