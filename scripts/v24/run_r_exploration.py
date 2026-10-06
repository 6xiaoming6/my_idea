#!/usr/bin/env python3
"""R1--R10: frozen, sequential single-GPU multiscale/generalization/memory suite."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_b3_c3 as baseline
import run_n_exploration as previous
from stmoe_imputer.config import deep_update
from run_team_accept_v4 import check_gpu_idle
CONFIG_DIR=ROOT/'configs/v24/r_exploration'
VARIANTS=tuple(f'R{i}' for i in range(1,11))


def jobs(dataset='taxibj',epochs=100,batch_size=32,variants=VARIANTS):
    protocol=baseline.load(CONFIG_DIR/'protocol.json');result=[]
    for variant in variants:
        spec=baseline.load(CONFIG_DIR/f'{variant}.json')
        job=previous.jobs(dataset,epochs,batch_size,(spec['base_variant'],))[0]
        cfg=deep_update(copy.deepcopy(job['config']),spec['override'])
        cfg['experiment_plan']={'suite':'r_exploration','variant':variant,'base_variant':spec['base_variant'],
                                'protocol':copy.deepcopy(protocol)}
        result.append({**job,'variant':variant,'name':spec['name'],'config':cfg})
    return result


def manifest(dataset):
    source=previous.manifest(dataset)
    for p in [*CONFIG_DIR.glob('*.json'), ROOT/'scripts/v24/README_R_EXPLORATION.md',
              ROOT/'tests/test_v24_r_exploration.py']:
        source[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return source


def references(dataset):
    path=CONFIG_DIR/f'references_{dataset}.json'
    if not path.exists():return []
    result=[]
    for ref in baseline.load(path):
        path=ROOT/ref['checkpoint']
        if not path.is_file():raise FileNotFoundError(f'Missing pinned reference checkpoint: {path}')
        stat=path.stat()
        result.append({**ref,'checkpoint':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    return result


def evaluation_valid(path,checkpoint,sha,protocol):
    if not path.exists():return False
    r=baseline.load(path)
    return (r.get('status')=='finished' and r.get('checkpoint')==str(Path(checkpoint).resolve())
            and r.get('config_sha256')==sha and r.get('protocol_sha256')==baseline.digest(protocol)
            and set(r.get('sets',{}))==set(protocol['evaluations']))


def run_evaluation(suite,name,checkpoint,sha,protocol,test_npz,gpu,reference=False,device="cuda"):
    destination=suite/('reference_evaluations' if reference else 'evaluations')/f'{name}.json'
    if evaluation_valid(destination,checkpoint,sha,protocol):return
    if device != "cpu":check_gpu_idle([gpu])
    command=[sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/evaluate_r_exploration.py'),
             '--checkpoint',str(checkpoint),'--protocol',str(suite/'protocol.json'),'--test-npz',test_npz,
             '--output',str(destination),'--expected-sha',sha,'--device',device]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    log_path=suite/'launcher_logs'/f'{name}.evaluation.log';log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('ab') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:code=process.wait()
        except BaseException:process.terminate();process.wait();raise
    if code or not evaluation_valid(destination,checkpoint,sha,protocol):
        raise RuntimeError(f'{name} evaluation failed; see {log_path}')


def summarize(suite,plan,protocol):
    rows=[]
    for job in plan:
        train=baseline.completed(suite,job);evaluation=None
        path=suite/'evaluations'/f'{job["variant"]}.json'
        if train:
            ckpt=Path(train['run_dir'])/'checkpoints/best.pth'
            if evaluation_valid(path,ckpt,baseline.digest(job['config']),protocol):evaluation=baseline.load(path)
        rows.append({'variant':job['variant'],'status':'finished' if evaluation else 'trained' if train else 'pending',
                     'run_dir':train['run_dir'] if train else None,'best_epoch':train['best_epoch'] if train else None,
                     'val_mae':train['best_val_mae'] if train else None,'training_seconds':train['total_time_sec'] if train else None,
                     'evaluations':{k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in evaluation['sets'].items()} if evaluation else {}})
    baseline.write(suite/'summary.json',rows)


def main(*, job_builder=jobs, source_manifest=manifest, reference_builder=references,
         variants=VARIANTS, suite_name='r_exploration', description=__doc__):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--dataset',choices=('taxibj','bikenyc'),default='taxibj')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--variants',nargs='+',choices=variants,default=list(variants))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.batch_size)<1 or a.gpu<0 or len(set(a.variants))!=len(a.variants):p.error('Invalid budget, GPU or repeated variants')
    plan=job_builder(a.dataset,a.epochs,a.batch_size,a.variants);protocol=plan[0]['config']['experiment_plan']['protocol']
    source=source_manifest(a.dataset);refs=reference_builder(a.dataset)
    data={v:{'size':Path(v).stat().st_size,'mtime_ns':Path(v).stat().st_mtime_ns} for v in plan[0]['sources'].values()}
    payload={'jobs':plan,'source':source,'data':data,'references':refs}
    fingerprint=baseline.digest(payload)[:16]
    suite=ROOT/'outputs/v24-COE/experiments'/suite_name/a.dataset/fingerprint
    if a.dry_run:
        import json
        print(json.dumps({'suite':str(suite),**payload},ensure_ascii=False,indent=2));return
    if a.summary_only:
        if not suite.exists():raise FileNotFoundError(suite)
        summarize(suite,plan,protocol);print(suite/'summary.json');return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suite.mkdir(exist_ok=True)
        for name,sha in source.items():
            target=suite/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen source changed')
            else:shutil.copyfile(ROOT/name,target)
        baseline.write(suite/'plan.json',{**payload,'fingerprint':fingerprint})
        baseline.write(suite/'protocol.json',protocol)
        for job in plan:baseline.write(suite/'configs'/f'{job["variant"]}.json',job['config'])
        # Read-only reference evaluation, before starting the ordered train queue.
        for ref in refs:
            print(f'Evaluating reference {ref["variant"]}',flush=True)
            stat=Path(ref['checkpoint']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(ref['size'],ref['mtime_ns']):raise RuntimeError('Reference changed')
            run_evaluation(suite,'reference_'+ref['variant'],ref['checkpoint'],ref['config_sha256'],protocol,
                           plan[0]['sources']['test'],a.gpu,reference=True)
        for job in plan:
            for path,stamp in data.items():
                stat=Path(path).stat()
                if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=stamp:raise RuntimeError('Dataset changed')
            try:
                if not baseline.completed(suite,job):baseline.launch(suite,job,a.gpu)
                receipt=baseline.completed(suite,job)
                run_evaluation(suite,job['variant'],Path(receipt['run_dir'])/'checkpoints/best.pth',
                               baseline.digest(job['config']),protocol,job['sources']['test'],a.gpu)
            finally:summarize(suite,plan,protocol)
        print(f'Results: {suite / "summary.json"}',flush=True)

if __name__=='__main__':main()
