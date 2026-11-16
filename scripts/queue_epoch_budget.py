#!/usr/bin/env python3
"""Serial GPU1 queue; complete-checkpoint recovery and fail-stop semantics."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from release_utils import load, write, sha256
from epoch_budget_common import completed
from report_epoch_budget import report

ACTIVE = None


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    global ACTIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args(); suite = args.suite.resolve()
    plan = load(suite / 'plan.json'); source = Path(plan['source'])
    with (suite / '.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, expected in load(suite / 'manifest.json')['files'].items():
            if sha256(name) != expected:
                raise RuntimeError('Frozen artifact changed: ' + name)
        protocol = load(source / 'configs/test_protocol.json')
        state = dict(status='running', started=stamp(), completed=[], active=None)

        def stop(sig, frame):
            if ACTIVE is not None and ACTIVE.poll() is None:
                ACTIVE.terminate()
            raise SystemExit(128 + sig)
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try:
            for job in plan['jobs']:
                name = job['name']; run = suite / 'runs' / name
                cfg = load(source / job['config'])
                if completed(run, job, cfg, protocol):
                    print(f'{stamp()} {name}: skip completed', flush=True)
                else:
                    state.update(active=name, updated=stamp()); write(suite / 'queue_state.json', state)
                    (run / 'logs').mkdir(parents=True, exist_ok=True)
                    command = [plan['python'], '-u', str(source / 'scripts/train.py'),
                               '--data-dir', plan['data_dir'], '--run-dir', str(run),
                               '--config', str(source / job['config']), '--gpu', '1',
                               '--threads', '2', '--evaluate']
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OPENBLAS_NUM_THREADS='2',
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                    print(f'{stamp()} {name}: {job["epochs"]} epochs, eta_min={job["eta_min"]}, GPU1', flush=True)
                    with (run / 'logs/console.log').open('ab', buffering=0) as log:
                        ACTIVE = subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT)
                        while chunk := ACTIVE.stdout.read1(65536):
                            log.write(chunk); sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
                        code = ACTIVE.wait(); ACTIVE = None
                    if code:
                        raise RuntimeError(f'{name} exited {code}; queue stopped, logs preserved')
                    if not completed(run, job, cfg, protocol):
                        raise RuntimeError(f'{name} returned without complete training/evaluation')
                state['completed'].append(name); state.update(updated=stamp())
                write(suite / 'queue_state.json', state)
            state.update(status='reporting', active=None); write(suite / 'queue_state.json', state)
            path = report(suite)
            state.update(status='finished', report=str(path), finished=stamp())
            write(suite / 'queue_state.json', state)
            print(f'Complete: {path}', flush=True)
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, SystemExit) else 'failed',
                         error=str(error), updated=stamp())
            write(suite / 'queue_state.json', state)
            if ACTIVE is not None and ACTIVE.poll() is None:
                ACTIVE.terminate(); ACTIVE.wait(timeout=30)
            try:
                report(suite)
            except Exception:
                traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
