#!/usr/bin/env python3
"""Wait for a frozen suite's training, six evaluations and report, then run ours.

This process uses only the standard library while waiting.  It never queries a
GPU, imports torch, or starts/resumes the dependency.  Its receipts are separate
from the training runner's state.json and a suite-local flock prevents duplicate
waiters.  A failed or user-paused dependency requires explicit intervention.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


RUNNER = 'scripts/v25/run_taxibj_input_communication.py'
PROTOCOLS = {
    'in_distribution', 'unseen_combinations', 'unseen_geometry', 'unseen_triple',
    'unseen_combinations_repeat', 'unseen_geometry_repeat',
}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _completion_status(suite, plan):
    """Return waiting/ready/blocked; completed markers alone are insufficient."""
    state_path = suite / 'state.json'
    state = load(state_path) if state_path.exists() else {}
    phase = str(state.get('phase', '')).lower()
    if phase in {'failed', 'blocked'} or phase.startswith('paused'):
        return 'blocked', f'dependency phase={phase}: {state.get("error", "")}'
    for filename in ('paused.json', 'paused_evaluation.json'):
        marker = suite / filename
        if marker.exists():
            paused = load(marker)
            if str(paused.get('status', paused.get('phase', 'paused'))).startswith('paused'):
                return 'blocked', f'dependency has user pause marker {filename}'
    if not (suite / 'completed.json').exists() or phase != 'completed':
        return 'waiting', f'dependency phase={phase or "not_started"}; completion pending'

    completed = load(suite / 'completed.json')
    report_path = suite / 'report_path.json'
    if not report_path.exists():
        return 'blocked', 'completed dependency is missing report_path.json'
    report = load(report_path)
    if report.get('complete') is not True or report.get('issues'):
        return 'blocked', 'dependency report is not complete or has audit issues'
    target = Path(report.get('path', ''))
    if not target.is_absolute() or not target.is_file() or target.stat().st_size == 0:
        return 'blocked', 'dependency formal report is missing or empty'
    if completed.get('report') != str(target) or state.get('report') != str(target):
        return 'blocked', 'dependency report paths disagree'
    jobs = plan['jobs']
    if not jobs or set(plan['order']) != set(jobs) or completed.get('count') != len(jobs):
        return 'blocked', 'dependency completed job count/order does not match frozen plan'
    for name, job in jobs.items():
        paths = [suite / folder / f'{name}.json' for folder in ('jobs', 'results', 'evaluations')]
        if not all(path.is_file() for path in paths):
            return 'blocked', f'dependency {name} is missing job/training/evaluation receipt'
        resolved, result, evaluation = map(load, paths)
        config = job['config']
        config_hash = digest(config)
        if resolved['config'] != config or resolved['sources'] != job['sources']:
            return 'blocked', f'dependency {name} resolved job differs from frozen plan'
        if result.get('status') != 'finished' or result.get('config_sha256') != config_hash:
            return 'blocked', f'dependency {name} training status/config mismatch'
        if result.get('completed_epochs') != config['train']['epochs']:
            return 'blocked', f'dependency {name} has incomplete training epochs'
        epoch = result.get('best_epoch')
        if not isinstance(epoch, int) or not 1 <= epoch <= config['train']['epochs']:
            return 'blocked', f'dependency {name} has invalid best epoch'
        run = Path(result['run_dir'])
        if not run.is_absolute() or not all((run / 'checkpoints' / p).is_file() for p in ('best.pth', 'last.pth')):
            return 'blocked', f'dependency {name} is missing best/last checkpoints'
        protocol = config['experiment_plan']['protocol']
        if set(protocol['evaluations']) != PROTOCOLS or set(evaluation.get('sets', {})) != PROTOCOLS:
            return 'blocked', f'dependency {name} must finish all six evaluation protocols'
        expected = {
            'status': 'finished', 'config_sha256': config_hash,
            'protocol_sha256': digest(protocol), 'best_epoch': epoch,
            'checkpoint': str((run / 'checkpoints/best.pth').resolve()),
            'npz': str(Path(job['sources']['test']).resolve()), 'split': 'test',
        }
        if any(evaluation.get(key) != value for key, value in expected.items()):
            return 'blocked', f'dependency {name} evaluation config/protocol/checkpoint/epoch mismatch'
        for label, spec in protocol['evaluations'].items():
            row = evaluation['sets'][label]
            expected_mask = {'families': spec['families'], 'rates': [spec.get('rate', protocol['rate'])],
                             'seed': spec['seed'], 'resample_each_epoch': False}
            if row.get('mask_spec') != expected_mask or row.get('samples', 0) <= 0:
                return 'blocked', f'dependency {name}/{label} mask or sample receipt mismatch'
            metrics = row.get('metrics', {})
            if any(not isinstance(metrics.get(k), (int, float)) or not math.isfinite(metrics[k])
                   or metrics[k] < 0 for k in ('mae', 'rmse')):
                return 'blocked', f'dependency {name}/{label} missing/nonfinite metrics'
    return 'ready', f'{len(jobs)} jobs, six protocols each, and formal report verified'


def dependency_status(dependency):
    """Public gate used by both the waiter and the frozen training runner."""
    try:
        suite = Path(dependency['path'])
        if not suite.is_absolute():
            return 'blocked', 'dependency path must be absolute'
        if file_sha(suite / 'plan.json') != dependency['plan_sha256']:
            return 'blocked', 'dependency plan hash changed'
        return _completion_status(suite, load(suite / 'plan.json'))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return 'blocked', f'dependency receipt audit failed: {error}'


@contextmanager
def exclusive_waiter(suite):
    with (suite / 'waiter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def queue_locks_available(suite, plan):
    """Check scheduler locks, not GPU occupancy; never hold locks while waiting."""
    paths = [suite.parent / 'queue.lock',
             Path(plan['project_root']) / 'outputs/v24-COE/experiments/gpu1.queue.lock']
    with ExitStack() as stack:
        try:
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def emit(suite, phase, detail, **extra):
    record = {'time': time.time(), 'phase': phase, 'detail': detail, 'pid': os.getpid(), **extra}
    write(suite / 'waiting.json', record)
    append_event(suite, record)


def append_event(suite, record):
    line = json.dumps(record, ensure_ascii=False) + '\n'
    with (suite / 'waiting_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(line)
        handle.flush()
    print(line, end='', flush=True)


def run(suite, *, poll_seconds=15, log_seconds=60):
    suite = Path(suite).resolve()
    if not 0 < poll_seconds <= 60 or log_seconds <= 0:
        raise ValueError('poll_seconds must be in (0, 60] and log_seconds positive')
    try:
        with exclusive_waiter(suite):
            return _run_locked(suite, poll_seconds, log_seconds)
    except BlockingIOError:
        append_event(suite, {'time': time.time(), 'phase': 'duplicate_waiter', 'pid': os.getpid(),
                             'detail': 'another waiter owns this suite; no runner launched'})
        return 75


def _run_locked(suite, poll_seconds, log_seconds):
    plan_hash = file_sha(suite / 'plan.json')
    plan = load(suite / 'plan.json')
    last_event, last_log = None, -float('inf')
    try:
        dependency = plan['run_after_suite']
        if Path(dependency['path']).resolve() == suite:
            raise RuntimeError('suite cannot depend on itself')
        while True:
            if file_sha(suite / 'plan.json') != plan_hash:
                raise RuntimeError('waiting suite plan changed')
            if (suite / 'completed.json').exists():
                own_status, own_detail = _completion_status(suite, plan)
                if own_status != 'ready':
                    raise RuntimeError('own completion receipt is invalid: ' + own_detail)
                emit(suite, 'already_completed', own_detail)
                return 0
            status, detail = dependency_status(dependency)
            if status == 'blocked':
                emit(suite, 'blocked', detail, dependency=dependency['path'])
                return 2
            if status == 'ready' and queue_locks_available(suite, plan):
                break
            if status == 'ready':
                detail = 'dependency complete; waiting for scheduler locks to be released'
            event = (status, detail)
            if event != last_event or time.monotonic() - last_log >= log_seconds:
                emit(suite, 'waiting', detail, dependency=dependency['path'])
                last_event, last_log = event, time.monotonic()
            time.sleep(poll_seconds)

        # Audit our executable before launch.  The runner also checks every
        # frozen source, data, reference and dependency again before training.
        runner = suite / 'source_snapshot' / RUNNER
        if file_sha(runner) != plan['source'][RUNNER]:
            raise RuntimeError('frozen training runner hash changed')
        emit(suite, 'launching', detail, dependency=dependency['path'])
        command = [sys.executable, '-u', str(runner), '--suite', str(suite)]
        code = subprocess.call(command, cwd=plan['project_root'], env=dict(os.environ, PYTHONFAULTHANDLER='1'))
        if code:
            emit(suite, 'failed', 'training runner exited; no automatic retry', returncode=code)
            return code if code > 0 else 128 - code
        status, detail = _completion_status(suite, plan)
        if status != 'ready':
            raise RuntimeError('runner exited without valid final receipts: ' + detail)
        emit(suite, 'completed', detail)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        emit(suite, 'blocked', str(error))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    return run(args.suite)


if __name__ == '__main__':
    raise SystemExit(main())
