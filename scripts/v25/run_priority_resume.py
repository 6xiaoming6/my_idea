#!/usr/bin/env python3
"""Run the authorized priority suite, then restore the explicitly paused queue."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from run_after_suite import load, write, file_sha, _completion_status, append_event

RUNNER='scripts/v25/run_taxibj_input_communication.py'


def verify_pause(spec):
    old=Path(spec['path']);pause_path=Path(spec['pause_path'])
    if file_sha(old/'plan.json')!=spec['plan_sha256'] or file_sha(pause_path)!=spec['pause_sha256']:
        raise RuntimeError('Paused queue plan or user authorization changed')
    pause=load(pause_path)
    if pause.get('status')!='paused_by_user' or load(old/'state.json').get('phase')!='paused_by_user':
        raise RuntimeError('Original queue is no longer paused by user')
    if file_sha(pause['checkpoint'])!=pause['checkpoint_sha256']:
        raise RuntimeError('Paused complete checkpoint changed')
    plan=load(old/'plan.json');runner=old/'source_snapshot'/RUNNER
    if file_sha(runner)!=plan['source'][RUNNER]:
        raise RuntimeError('Original frozen runner changed')
    return old,pause,runner


def run(suite):
    suite=Path(suite).resolve();plan=load(suite/'plan.json');spec=plan['resume_after_priority']
    with (suite/'priority_controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify_pause(spec)
            runner=suite/'source_snapshot'/RUNNER
            if file_sha(runner)!=plan['source'][RUNNER]:raise RuntimeError('Priority runner changed')
            write(suite/'priority_status.json',{'phase':'running_priority','time':time.time()})
            subprocess.run([sys.executable,'-u',str(runner),'--suite',str(suite)],cwd=plan['project_root'],check=True)
            status,detail=_completion_status(suite,plan)
            if status!='ready':raise RuntimeError('Priority suite incomplete: '+detail)
            old,pause,runner=verify_pause(spec)
            resumed={**pause,'status':'resuming_after_I26','resumed_at':time.time(),'priority_suite':str(suite)}
            write(old/'resume_after_i26.json',resumed)
            append_event(suite,{'phase':'resuming_original','time':time.time(),'checkpoint':pause['checkpoint']})
            write(suite/'priority_status.json',{'phase':'resuming_original','time':time.time()})
            # The frozen original runner skips completed jobs and restores last.pth.
            subprocess.run([sys.executable,'-u',str(runner),'--suite',str(old)],cwd=plan['project_root'],check=True)
            status,detail=_completion_status(old,load(old/'plan.json'))
            if status!='ready':raise RuntimeError('Resumed queue incomplete: '+detail)
            write(suite/'priority_status.json',{'phase':'completed','time':time.time()})
        except BaseException as error:
            write(suite/'priority_status.json',{'phase':'blocked','time':time.time(),'error':repr(error)})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    run(p.parse_args().suite)
