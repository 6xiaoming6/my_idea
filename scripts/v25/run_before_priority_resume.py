#!/usr/bin/env python3
"""After I26 completes, run I27 before releasing the held resume controller."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_after_suite import load,write,file_sha,dependency_status,_completion_status,append_event
from run_interleaved import process,same_process,children
from run_priority_resume import verify_pause

RUNNER='scripts/v25/run_taxibj_input_communication.py'


def held_controller(spec):
    path=Path(spec['controller_hold'])
    if file_sha(path)!=spec['controller_hold_sha256']:raise RuntimeError('Controller hold evidence changed')
    identity=load(path)
    if not same_process(identity):raise RuntimeError('Held resume controller identity changed')
    if process(identity['pid'])['state'] not in ('T','t'):raise RuntimeError('Resume controller must remain suspended')
    dependency=load(Path(spec['path'])/'plan.json')
    verify_pause(dependency['resume_after_priority'])
    return identity


def check_ready(plan):
    spec=plan['insert_before_resume'];identity=held_controller(spec)
    status,detail=dependency_status(spec)
    if status=='blocked':raise RuntimeError(detail)
    active=any((p:=process(pid)) and p['state'] not in ('Z','X') for pid in children(identity['pid']))
    return status=='ready' and not active,detail,identity


def validate_launch(plan):
    ready,detail,_=check_ready(plan)
    if not ready:raise RuntimeError('I26 must finish and release its training queue: '+detail)


def run(suite):
    suite=Path(suite).resolve();plan=load(suite/'plan.json')
    def emit(phase,detail):
        row={'phase':phase,'detail':detail,'time':time.time(),'pid':os.getpid()}
        write(suite/'waiting.json',row);append_event(suite,row)
    with (suite/'insert_controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            last_log=0
            while True:
                ready,detail,identity=check_ready(plan)
                if ready:break
                if time.monotonic()-last_log>=60:
                    emit('waiting_for_I26',detail);last_log=time.monotonic()
                time.sleep(5)
            runner=suite/'source_snapshot'/RUNNER
            if file_sha(runner)!=plan['source'][RUNNER]:raise RuntimeError('I27 frozen runner changed')
            emit('running_I27','I26 training, six tests and report complete; starting I27 on GPU1')
            subprocess.run([sys.executable,'-u',str(runner),'--suite',str(suite)],cwd=plan['project_root'],check=True)
            status,detail=_completion_status(suite,plan)
            if status!='ready':raise RuntimeError('I27 final audit failed: '+detail)
            identity=held_controller(plan['insert_before_resume'])
            os.kill(identity['pid'],signal.SIGCONT)
            emit('completed','I27 complete; original I26 controller released to resume I19-I22')
        except BaseException as error:
            emit('blocked',repr(error)+'; original resume controller remains held; no automatic retry')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    run(p.parse_args().suite)
