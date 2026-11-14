#!/usr/bin/env python3
"""Frozen ten-run queue. Serial GPU1, finite failures, complete-checkpoint resume."""
import argparse,fcntl,json,os,signal,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime
from release_utils import load,write,sha256,digest
from report_tokens import report
ACTIVE=None

def stamp():return datetime.now().astimezone().isoformat()
def main():
 global ACTIVE
 p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();suite=a.suite.resolve()
 plan=load(suite/'plan.json');source=Path(plan['source'])
 with (suite/'.queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for path,expected in load(suite/'manifest.json')['files'].items():
   if sha256(path)!=expected:raise RuntimeError('Frozen artifact changed: '+path)
  if load(suite/'preflight_gpu/results.json')['status']!='passed':raise RuntimeError('GPU preflight missing')
  protocol=load(source/'configs/test_protocol.json')
  def stop(sig,frame):
   if ACTIVE is not None and ACTIVE.poll() is None:ACTIVE.terminate()
   raise SystemExit(128+sig)
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  state={'status':'running','started':stamp(),'completed':[]}
  try:
   for job in plan['jobs']:
    name=job['name'];run=suite/'runs'/name;cfg=source/job['config']
    result=load(run/'result.json') if (run/'result.json').exists() else {}
    ev=load(run/'evaluation.json') if (run/'evaluation.json').exists() else {}
    done=result.get('status')=='finished' and result.get('completed_epochs')==100 and ev.get('status')=='finished' and set(ev.get('sets',{}))==set(protocol['evaluations'])
    if done:
     if result['config_sha256']!=digest(load(cfg)) or ev['config_sha256']!=digest(load(cfg)):raise RuntimeError('Completed config mismatch')
     if not all((run/'checkpoints'/f'{k}.pth').is_file() for k in ('best','last')):raise RuntimeError('Missing completed checkpoint')
     print(f'{stamp()} {name} skip completed',flush=True)
    else:
     state.update(active=name,updated=stamp());write(suite/'queue_state.json',state)
     (run/'logs').mkdir(parents=True,exist_ok=True)
     command=[plan['python'],'-u',str(source/'scripts/train.py'),'--data-dir',plan['data_dir'],'--run-dir',str(run),'--config',str(cfg),'--name',name,'--gpu','1','--threads','2','--evaluate']
     print(f'{stamp()} {name} start/resume GPU1',flush=True)
     env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
     with (run/'logs/console.log').open('ab',buffering=0) as log:
      ACTIVE=subprocess.Popen(command,cwd=source,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
      while chunk:=ACTIVE.stdout.read1(65536):
       log.write(chunk);sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
      code=ACTIVE.wait();ACTIVE=None
     if code:raise RuntimeError(f'{name} exited {code}; queue stopped, see {run}/logs/console.log')
     result=load(run/'result.json');ev=load(run/'evaluation.json')
     if result['status']!='finished' or result['completed_epochs']!=100 or ev['status']!='finished' or set(ev['sets'])!=set(protocol['evaluations']):raise RuntimeError('Incomplete run after successful process')
     expected=plan['initializations'][name]['state_sha256']
     if load(run/'initialization.json')['full_state_sha256']!=expected:raise RuntimeError('Training initialization differs from frozen preflight')
    state['completed'].append(name);state.update(updated=stamp());write(suite/'queue_state.json',state)
   state.update(status='reporting',active=None);write(suite/'queue_state.json',state)
   path=report(suite)
   state.update(status='finished',report=str(path),finished=stamp());write(suite/'queue_state.json',state)
   print(f'Complete: {path}',flush=True)
  except BaseException as exc:
   state.update(status='failed' if not isinstance(exc,SystemExit) else 'interrupted',error=str(exc),updated=stamp())
   write(suite/'queue_state.json',state)
   if ACTIVE is not None and ACTIVE.poll() is None:
    ACTIVE.terminate();ACTIVE.wait(timeout=30)
   try:report(suite)
   except Exception:traceback.print_exc()
   raise
if __name__=='__main__':main()
