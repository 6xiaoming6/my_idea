#!/usr/bin/env python3
"""Run A -> B -> C -> D -> E sequentially after checking GPUs are idle.

Training is manual only. This launcher neither stops other jobs nor starts a
second GPU. Dry-run and summary-only never query/use a GPU or launch training.
"""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_experiments as runner


def check_gpu_idle(gpu):
    result=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],
                          check=True,capture_output=True,text=True)
    if str(gpu) not in result.stdout.split():
        raise RuntimeError(f'GPU {gpu} does not exist')
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name',
                              '--format=csv,noheader'],check=True,capture_output=True,text=True)
    if processes.stdout.strip():
        raise RuntimeError('A GPU already has a compute process. The five-job queue was NOT started. '
                           'Wait for existing GPU work to stop; do not run on two GPUs simultaneously.\n'+processes.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if args.gpu<0:p.error('--gpu must be nonnegative')
    policy=ROOT/'configs/v24/abcde_experiments.json'
    command=[sys.executable,'-u',str(ROOT/'scripts/v24/run_experiments.py'),
             '--config',str(policy),'--study','abcde','--gpu',str(args.gpu)]
    if args.dry_run or args.summary_only:
        command.append('--dry-run' if args.dry_run else '--summary-only')
        return subprocess.call(command,cwd=ROOT)
    output=runner.resolve(runner.load(policy)['output_dir']);output.mkdir(parents=True,exist_ok=True)
    with (output/'.single_gpu_queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('The A/B/C/D/E queue is already running') from None
        check_gpu_idle(args.gpu)
        return subprocess.call(command,cwd=ROOT)

if __name__=='__main__':sys.exit(main())
