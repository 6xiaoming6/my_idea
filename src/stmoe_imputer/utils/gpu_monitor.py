"""Low-frequency, read-only GPU telemetry, isolated from training and its RNG.

Run this file directly to attach to an existing PID, or to follow training
children of a frozen queue without modifying its source snapshot.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime

GPU_FIELDS = ('index', 'uuid', 'pci.bus_id', 'driver_version', 'temperature.gpu',
              'power.draw', 'power.limit', 'utilization.gpu', 'memory.used', 'memory.total')
APP_FIELDS = ('gpu_uuid', 'pid', 'used_memory')


def process_identity(pid):
    """Linux start ticks prevent a recycled PID from extending monitoring."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def query(kind, fields, timeout=2):
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-{kind}=' + ','.join(fields),
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=timeout, check=False)
        if result.returncode:
            return {'error': (result.stderr or result.stdout)[:1000],
                    'returncode': result.returncode}
        rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        return {'rows': [dict(zip(fields, row)) for row in rows if len(row) == len(fields)],
                **({'unparsed': [','.join(r)[:500] for r in rows if len(r) != len(fields)]}
                   if any(len(r) != len(fields) for r in rows) else {})}
    except subprocess.TimeoutExpired:
        return {'error': 'nvidia-smi query timed out', 'timeout_sec': timeout}
    except OSError as error:
        return {'error': str(error)[:1000]}


def host_memory():
    wanted = {'MemAvailable', 'MemTotal', 'SwapFree', 'SwapTotal'}
    try:
        return {k: int(v.split()[0]) for line in Path('/proc/meminfo').read_text().splitlines()
                for k, v in [line.split(':', 1)] if k in wanted}
    except (OSError, ValueError):
        return {}


def process_status(pid):
    try:
        return {k: v.strip() for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                for k, v in [line.split(':', 1)] if k in {'State', 'Threads', 'VmRSS'}}
    except OSError:
        return {}


class Log:
    def __init__(self, run_dir, pid, identity, mode):
        self.pid, self.identity = pid, identity
        self.path = Path(run_dir) / 'logs/gpu_status.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a', encoding='utf-8', buffering=1)
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise
        try:
            visible = next((v.split('=', 1)[1] for v in
                            Path(f'/proc/{pid}/environ').read_bytes().decode(errors='replace').split('\0')
                            if v.startswith('CUDA_VISIBLE_DEVICES=')), None)
        except OSError:
            visible = None
        self.emit('start', pid=pid, process_start_ticks=identity, mode=mode,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  cuda_visible_devices=visible, scope='all host GPUs',
                  units={'temperature': 'C', 'power': 'W', 'gpu_memory': 'MiB',
                         'utilization': '%', 'host_memory': 'KiB'})

    def emit(self, event, **values):
        self.file.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(timespec='seconds'),
                                        monotonic_sec=round(time.monotonic(), 3), event=event,
                                        **values), ensure_ascii=False, separators=(',', ':')) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self, reason):
        try:
            self.emit('stop', reason=reason)
        finally:
            self.file.close()


def training_children(queue_pid):
    """Only descend from the specified queue; never monitor unrelated jobs."""
    pending, seen, found = [queue_pid], set(), []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            pending.extend(int(v) for v in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
            args = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(Path(a).name == 'train_four_direction.py' for a in args) and '--run-dir' in args:
                found.append((pid, args[args.index('--run-dir') + 1]))
        except (OSError, ValueError, IndexError):
            continue
    return found


def monitor(pid, run_dir=None, queue=False, interval=15, stop=None):
    stop = stop or threading.Event()
    identity = process_identity(pid)
    if identity is None:
        return
    logs = {}
    try:
        while not stop.is_set() and process_identity(pid) == identity:
            targets = training_children(pid) if queue else [(pid, run_dir)]
            for child, directory in targets:
                child_identity = process_identity(child)
                if child_identity is not None and child not in logs:
                    try:
                        logs[child] = Log(directory, child, child_identity,
                                          'attached_queue' if queue else 'training')
                    except BlockingIOError:
                        continue  # Another monitor already owns this run's log.
            for child, log in list(logs.items()):
                if process_identity(child) != log.identity:
                    log.close('training_process_exited')
                    del logs[child]
            if logs:
                sample = dict(gpus=query('gpu', GPU_FIELDS),
                              compute_processes=query('compute-apps', APP_FIELDS),
                              host_memory_kib=host_memory())
                for child, log in logs.items():
                    log.emit('sample', training_pid=child, training_process=process_status(child), **sample)
            stop.wait(interval)
    except Exception as error:
        for log in logs.values():
            try:
                log.emit('monitor_error', error=repr(error)[:1000])
            except OSError:
                pass
        raise
    finally:
        for log in logs.values():
            log.close('monitor_stopped' if stop.is_set() else 'watched_process_exited')


class GPUMonitor:
    """A bounded-lifetime subprocess; telemetry failure never fails training."""
    def __init__(self, run_dir, enabled=True):
        self.run_dir, self.enabled, self.process = Path(run_dir), enabled, None

    def __enter__(self):
        if self.enabled:
            try:
                self.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), '--pid', str(os.getpid()),
                     '--run-dir', str(self.run_dir.resolve())], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            except OSError as error:
                warnings.warn(f'GPU monitor could not start: {error}')
        return self

    def __exit__(self, *exc):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.process.kill()
                # Never delay training exit indefinitely on driver failure.
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pid', type=int)
    group.add_argument('--queue-pid', type=int)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    if args.interval <= 0 or (args.pid is not None and args.run_dir is None):
        parser.error('positive interval and --run-dir for --pid required')
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    monitor(args.queue_pid if args.queue_pid is not None else args.pid,
            args.run_dir, queue=args.queue_pid is not None, interval=args.interval, stop=stop)


if __name__ == '__main__':
    main()
