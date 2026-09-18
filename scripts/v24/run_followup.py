#!/usr/bin/env python3
"""Run fixed-path, D-static and F sequentially on one explicitly idle GPU."""
from __future__ import annotations
import argparse, fcntl, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/v24'))
import run_experiments as runner

def check_gpu_idle(gpu):
    visible=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    if str(gpu) not in visible.stdout.split(): raise RuntimeError(f'GPU {gpu} does not exist')
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],check=True,capture_output=True,text=True)
    if processes.stdout.strip(): raise RuntimeError('A GPU already has a compute process; follow-up queue was not started:\n'+processes.stdout)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true');a=p.parse_args()
    policy=ROOT/'configs/v24/followup_experiments.json';command=[sys.executable,'-u',str(ROOT/'scripts/v24/run_experiments.py'),'--config',str(policy),'--study','followup','--gpu',str(a.gpu)]
    if a.dry_run or a.summary_only:return subprocess.call(command+(['--dry-run'] if a.dry_run else ['--summary-only']),cwd=ROOT)
    output=runner.resolve(runner.load(policy)['output_dir']);output.mkdir(parents=True,exist_ok=True)
    with (output/'.single_gpu_queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Follow-up queue already running') from None
        check_gpu_idle(a.gpu);return subprocess.call(command,cwd=ROOT)
if __name__=='__main__':sys.exit(main())
