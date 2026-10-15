#!/usr/bin/env python3
"""Insert a frozen suite at an existing queue's evaluation boundary.

The original parent retains both scheduler locks. Only its worker is SIGSTOP'ed,
AFTER the requested evaluation child exists; that child can finish because its
stdout goes directly to a file. No training process is signalled. The inserted
worker borrows this verified exclusive interval and resumes the old worker only
on successful training/evaluation/report completion. All decisions are logged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_after_suite import load, write, digest, file_sha, _completion_status


def process(pid):
    root = Path('/proc')/str(pid)
    try:
        fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
        args = (root/'cmdline').read_bytes().decode().strip('\0').split('\0')
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'start': fields[19], 'args': args}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(identity):
    current = process(identity['pid'])
    return current is not None and current['start'] == identity['start'] and current['args'] == identity['args']


def children(pid):
    try:
        return [int(x) for x in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
    except FileNotFoundError:
        return []


def boundary_status(spec):
    """Audit one training plus six-evaluation receipt, without requiring later jobs."""
    suite = Path(spec['path'])
    if file_sha(suite/'plan.json') != spec['plan_sha256']:
        raise RuntimeError('Dependency plan changed')
    plan = load(suite/'plan.json'); name = spec['job']; job = plan['jobs'][name]
    state = load(suite/'state.json') if (suite/'state.json').exists() else {}
    if state.get('phase') in ('failed','blocked') or str(state.get('phase','')).startswith('paused'):
        raise RuntimeError('Dependency failed or paused: '+str(state))
    for marker in ('paused.json','paused_evaluation.json'):
        if (suite/marker).exists():
            raise RuntimeError('Dependency has a user pause marker: '+marker)
    result, evaluation = suite/'results'/f'{name}.json', suite/'evaluations'/f'{name}.json'
    if not result.exists() or not evaluation.exists():
        return False
    r,e = load(result),load(evaluation); cfg=job['config']; protocol=cfg['experiment_plan']['protocol']
    if e.get('status') != 'finished': return False
    run=Path(r['run_dir']); cp=run/'checkpoints/best.pth'
    if not (r.get('status')=='finished' and r['completed_epochs']==cfg['train']['epochs']
            and r['config_sha256']==e['config_sha256']==digest(cfg)
            and e['protocol_sha256']==digest(protocol) and e['best_epoch']==r['best_epoch']
            and e['checkpoint']==str(cp.resolve()) and e['split']=='test'
            and e['npz']==str(Path(job['sources']['test']).resolve())
            and cp.is_file() and (run/'checkpoints/last.pth').is_file()
            and set(e['sets'])==set(protocol['evaluations'])):
        raise RuntimeError('Dependency boundary receipts mismatch')
    import math
    for label, spec in protocol['evaluations'].items():
        row=e['sets'][label]
        expected={'families':spec['families'],'rates':[spec.get('rate',protocol['rate'])],
                  'seed':spec['seed'],'resample_each_epoch':False}
        if row['mask_spec']!=expected or row['samples']<=0 or any(
                not math.isfinite(row['metrics'][k]) or row['metrics'][k]<0 for k in ('mae','rmse')):
            raise RuntimeError('Dependency boundary protocol mismatch: '+label)
    return True


def locks_owned(plan, parent_pid):
    paths=[Path(plan['interleave_after']['path']).parent/'queue.lock',
           Path(plan['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock']
    records=Path('/proc/locks').read_text().splitlines()
    for path in paths:
        st=path.stat(); identity=(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)
        found=False
        for row in records:
            fields=row.split()
            if len(fields)<6 or fields[1]!='FLOCK' or fields[4]!=str(parent_pid): continue
            parts=fields[5].split(':')
            if len(parts)==3 and (int(parts[0],16),int(parts[1],16),int(parts[2]))==identity:
                found=True;break
        if not found: raise RuntimeError('Original parent no longer owns queue lock '+str(path))


def validate_launch(suite, plan):
    lease=load(Path(suite)/'interleave_lease.json')
    if lease['status']!='leased' or lease['controller']['pid']!=os.getppid():
        raise RuntimeError('Inserted worker requires its live boundary controller')
    for name in ('controller','parent','worker'):
        if not same_process(lease[name]): raise RuntimeError('Interleave process identity changed: '+name)
    if process(lease['worker']['pid'])['state'] not in ('T','t'):
        raise RuntimeError('Original scheduler is not suspended')
    locks_owned(plan,lease['parent']['pid'])
    if not boundary_status(plan['interleave_after']): raise RuntimeError('I18 evaluation not complete')
    if any((p:=process(pid)) and p['state'] not in ('Z','X') for pid in children(lease['worker']['pid'])):
        raise RuntimeError('Original evaluation child has not exited')


def emit(suite,phase,detail):
    row={'time':time.time(),'phase':phase,'detail':detail,'pid':os.getpid()}
    write(suite/'waiting.json',row)
    with (suite/'waiting_events.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def adopt_suspended(plan, parent, worker):
    """Explicit handoff after user pauses an inserted run; never resume it."""
    source=plan.get('adopt_suspended_from')
    if not source:raise RuntimeError('No frozen authorization for suspended handoff')
    old=Path(source['path'])
    if file_sha(old/'paused.json')!=source['pause_sha256'] or file_sha(old/'interleave_lease.json')!=source['lease_sha256']:
        raise RuntimeError('Paused insertion handoff evidence changed')
    pause,lease=load(old/'paused.json'),load(old/'interleave_lease.json')
    if pause.get('status')!='paused_by_user' or load(old/'state.json').get('phase')!='paused_by_user':
        raise RuntimeError('Previous insertion was not paused by user')
    if same_process(lease['controller']):raise RuntimeError('Previous insertion controller is still alive')
    for name,identity in [('parent',parent),('worker',worker)]:
        if not same_process(lease[name]) or lease[name]['pid']!=identity['pid']:
            raise RuntimeError('Original scheduler identity changed during handoff')
    if process(worker['pid'])['state'] not in ('T','t'):
        raise RuntimeError('Original scheduler must remain suspended during handoff')
    locks_owned(plan,parent['pid'])
    if not boundary_status(plan['interleave_after']):raise RuntimeError('Original boundary evaluation is incomplete')
    if any((p:=process(pid)) and p['state'] not in ('Z','X') for pid in children(worker['pid'])):
        raise RuntimeError('Original boundary evaluation is still active')


def run(suite,parent_pid,worker_pid,adopt=False):
    suite=Path(suite).resolve();plan=load(suite/'plan.json');spec=plan['interleave_after']
    dep=Path(spec['path']); parent=process(parent_pid);worker=process(worker_pid)
    if not parent or not worker or worker['ppid']!=parent_pid:
        raise RuntimeError('Original scheduler parent/worker not found')
    if '--worker' not in worker['args'] or str(dep) not in worker['args'] or str(dep) not in parent['args']:
        raise RuntimeError('Refusing to signal unrelated process')
    with (suite/'interleave.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suspended=False
        try:
            if adopt:
                adopt_suspended(plan,parent,worker)
                suspended=True
                emit(suite,'adopted_user_pause','Replacing explicitly paused I24; original GPU1 locks retained')
            last_log=0
            while not suspended:
                if not same_process(parent) or not same_process(worker):
                    raise RuntimeError('Original scheduler exited; no automatic restart')
                locks_owned(plan,parent_pid)
                boundary_status(spec)  # Detect failed or explicitly paused dependency.
                state=load(dep/'state.json')
                if state.get('variant')==spec['job'] and state.get('phase')=='evaluating':
                    live=[process(pid) for pid in children(worker_pid)]
                    evaluation=str(dep/'evaluations'/f"{spec['job']}.json")
                    if any(p and evaluation in p['args'] for p in live):
                        os.kill(worker_pid,signal.SIGSTOP);suspended=True
                        for _ in range(50):
                            if process(worker_pid)['state'] in ('T','t'):break
                            time.sleep(.02)
                        if process(worker_pid)['state'] not in ('T','t'):raise RuntimeError('Scheduler did not suspend')
                        state=load(dep/'state.json')
                        if state.get('variant')!=spec['job'] or state.get('phase')!='evaluating':
                            os.kill(worker_pid,signal.SIGCONT);suspended=False
                            raise RuntimeError('Missed requested boundary; original queue unchanged')
                        break
                order=load(dep/'plan.json')['order']
                if state.get('variant') in order and order.index(state['variant'])>order.index(spec['job']):
                    raise RuntimeError('Requested boundary already passed; no training interrupted')
                if time.monotonic()-last_log>=60:
                    emit(suite,'waiting',f"Insert after {spec['job']}; current {state}");last_log=time.monotonic()
                time.sleep(.5)
            emit(suite,'waiting_for_evaluation','Original scheduler suspended; I18 evaluation continues independently')
            while True:
                if not same_process(parent) or not same_process(worker): raise RuntimeError('Original scheduler lost during insertion')
                locks_owned(plan,parent_pid)
                ready=boundary_status(spec)
                active=any((p:=process(pid)) and p['state'] not in ('Z','X') for pid in children(worker_pid))
                if not active:
                    if not ready:raise RuntimeError('I18 evaluation child exited without complete receipts')
                    break
                time.sleep(1)
            lease={'status':'leased','parent':parent,'worker':worker,'controller':process(os.getpid()),
                   'dependency':spec,'time':time.time()}
            write(suite/'interleave_lease.json',lease)
            # The original scheduler still owns the global locks. The inserted
            # --worker is guarded by validate_launch and never contends for them.
            emit(suite,'running_insert','I18 training and six evaluations complete; starting selected inserted jobs')
            runner=suite/'source_snapshot/scripts/v25/run_taxibj_input_communication.py'
            if file_sha(runner)!=plan['source']['scripts/v25/run_taxibj_input_communication.py']:
                raise RuntimeError('Inserted runner source changed')
            code=subprocess.call([sys.executable,'-u',str(runner),'--suite',str(suite),'--worker'],cwd=plan['project_root'])
            if code:raise RuntimeError('Inserted queue failed; original scheduler remains suspended; no automatic retry')
            status,detail=_completion_status(suite,plan)
            if status!='ready':raise RuntimeError('Inserted queue final audit failed: '+detail)
            if not same_process(worker) or not same_process(parent):raise RuntimeError('Cannot resume changed original process')
            locks_owned(plan,parent_pid)
            os.kill(worker_pid,signal.SIGCONT);suspended=False
            lease['status']='released';write(suite/'interleave_lease.json',lease)
            emit(suite,'completed','Inserted jobs complete; original scheduler resumed for I19-I22')
        except BaseException as error:
            emit(suite,'blocked',str(error)+f'; original_scheduler_suspended={suspended}')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--worker-pid',type=int,required=True)
    p.add_argument('--adopt-suspended',action='store_true')
    a=p.parse_args();run(a.suite,a.parent_pid,a.worker_pid,a.adopt_suspended)
