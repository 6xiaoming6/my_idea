#!/usr/bin/env python3
"""Sequential single-GPU B3 foundation -> C3 executed local-support experiment."""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stmoe_imputer.config import deep_update
from run_team_accept_v4 import check_gpu_idle

LABELS={'B1':'moe_single_top8','B2':'moe_independent_top2','B3':'coe_feedback_base','C3':'coe_local_support'}

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)

def jobs(dataset='taxibj',epochs=100,batch_size=32,variants=('B3','C3')):
    base=load(ROOT/f'configs/v24/coe_main_s4_e8_{dataset}_base.json')
    policy=load(ROOT/f'configs/v24/coe_direct_baselines_{dataset}_experiments.json')
    sources=policy['studies'][policy['default_study']]['sources']
    result=[]
    for variant in variants:
        cfg=deep_update(copy.deepcopy(base),load(ROOT/f'configs/v24/b3_c3/{variant}.json'))
        cfg['data']['batch_size']=batch_size
        cfg['output_dir']=str(ROOT/'outputs/v24-COE')
        cfg['train']['epochs']=epochs;cfg['train']['scheduler']['total_epochs']=epochs
        for key in ('lr_router','lr_aux','lr_v14','partner_probe'):cfg['train'].pop(key,None)
        cfg['experiment_plan']={'suite':'b3_c3','variant':variant,'baseline':'B3_feedback_direct_shared_top2'}
        assert cfg['data']['train_mask_diversity']['families']==cfg['data']['eval_mask_diversity']['families']
        assert cfg['data']['train_mask_diversity']['rates']==cfg['data']['eval_mask_diversity']['rates']==[.4]
        result.append({'variant':variant,'name':LABELS[variant],'config':cfg,
                       'sources':{k:str(ROOT/v) for k,v in sources.items()}})
    if variants==('B3','C3') or list(variants)==['B3','C3']:
        a,b=(copy.deepcopy(j['config']) for j in result)
        for cfg in (a,b):
            cfg.pop('experiment_plan');cfg['model']['coe'].pop('local_routing')
        assert a==b,'B3 and C3 must differ only in local routing/support'
    return result

def source_manifest(dataset):
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/train.py',Path(__file__).resolve()]
    paths+=list((ROOT/'configs/v24/b3_c3').glob('*.json'))
    paths+=[ROOT/f'configs/v24/coe_main_s4_e8_{dataset}_base.json',ROOT/f'configs/v24/coe_direct_baselines_{dataset}_experiments.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def completed(suite,job):
    for receipt_path in sorted((suite/'results').glob(job['variant']+'.attempt*.json'),reverse=True):
        receipt=load(receipt_path)
        if receipt.get('status')!='finished' or receipt.get('config_sha256')!=digest(job['config']):continue
        if receipt.get('completed_epochs')!=job['config']['train']['epochs'] or not receipt.get('test'):continue
        run=Path(receipt['run_dir'])
        if all((run/'checkpoints'/name).is_file() for name in ('best.pth','last.pth')):
            return receipt
    return None

def summarize(suite,plan):
    rows=[]
    for job in plan:
        receipt=completed(suite,job)
        rows.append({'variant':job['variant'],'status':'finished' if receipt else 'pending',
                     'best_epoch':receipt['best_epoch'] if receipt else None,
                     'val_mae':receipt['best_val_mae'] if receipt else None,
                     'test_mae':receipt['test']['mae'] if receipt else None,
                     'test_rmse':receipt['test']['rmse'] if receipt else None,
                     'total_time_sec':receipt['total_time_sec'] if receipt else None,
                     'run_dir':receipt['run_dir'] if receipt else None})
    write(suite/'summary.json',rows)
    with (suite/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

def launch(suite,job,gpu):
    attempts=list((suite/'launcher_logs').glob(job['variant']+'.attempt*.log'))
    attempt=max([int(p.stem.split('attempt')[-1]) for p in attempts],default=0)+1
    config_path=suite/'configs'/f'{job["variant"]}.json';write(config_path,job['config'])
    receipt_path=suite/'results'/f'{job["variant"]}.attempt{attempt}.json'
    command=[sys.executable,'-u',str(suite/'source_snapshot/scripts/train.py'),'-c',str(config_path),
             '--name',job['name'],'--no_plot','--result-file',str(receipt_path)]
    for split,path in job['sources'].items():command.extend([f'--{split}_npz',path])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    check_gpu_idle([gpu])
    print(f'Starting {job["variant"]} {job["name"]} on GPU {gpu}',flush=True)
    log_path=suite/'launcher_logs'/f'{job["variant"]}.attempt{attempt}.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('wb') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            while chunk:=os.read(process.stdout.fileno(),4096):
                log.write(chunk);log.flush();sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if code or not completed(suite,job):raise RuntimeError(f'{job["variant"]} failed; see {log_path}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('taxibj','bikenyc'),default='taxibj')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--variants',nargs='+',choices=LABELS,default=['B3','C3'])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.gpu<0:parser.error('epochs/batch-size must be positive; gpu nonnegative')
    plan=jobs(args.dataset,args.epochs,args.batch_size,args.variants)
    code=source_manifest(args.dataset)
    data={p:{'size':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in plan[0]['sources'].values()}
    fingerprint=digest({'jobs':plan,'source':code,'data':data})[:16]
    suite=ROOT/'outputs/v24-COE/experiments/b3_c3'/args.dataset/fingerprint
    if args.dry_run:
        print(json.dumps({'suite':str(suite),'jobs':plan},ensure_ascii=False,indent=2));return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suite.mkdir(exist_ok=True)
        if not args.summary_only:
            for name,sha in code.items():
                target=suite/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen source changed')
                else:shutil.copyfile(ROOT/name,target)
            write(suite/'plan.json',{'jobs':plan,'source':code,'data':data,'fingerprint':fingerprint})
            for job in plan:
                if completed(suite,job):continue
                for path, stamp in data.items():
                    actual={'size':Path(path).stat().st_size,'mtime_ns':Path(path).stat().st_mtime_ns}
                    if actual!=stamp:raise RuntimeError(f'Dataset changed during queue: {path}')
                try:launch(suite,job,args.gpu)
                finally:summarize(suite,plan)
        summarize(suite,plan)
        print(f'Results: {suite / "summary.json"}',flush=True)

if __name__=='__main__':main()
