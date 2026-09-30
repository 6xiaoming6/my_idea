#!/usr/bin/env python3
"""N1--N7, based on B-series models; sequential single-GPU training and OOD testing."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b3_c3 as baseline
from stmoe_imputer.config import deep_update
from run_team_accept_v4 import check_gpu_idle

VARIANTS = tuple(f'N{i}' for i in range(1,8))
CONFIG_DIR = ROOT/'configs/v24/n_exploration'


def jobs(dataset='taxibj', epochs=100, batch_size=32, variants=VARIANTS):
    protocol = baseline.load(CONFIG_DIR/'protocol.json')
    result = []
    for variant in variants:
        spec = baseline.load(CONFIG_DIR/f'{variant}.json')
        job = baseline.jobs(dataset, epochs, batch_size, (spec['base_variant'],))[0]
        cfg = deep_update(job['config'], spec['override'])
        for key, resample in [('train_mask_diversity',True), ('eval_mask_diversity',False)]:
            cfg['data'][key] = {'families':protocol['train_families'], 'rates':[protocol['rate']],
                                'seed':protocol['mask_seed'], 'resample_each_epoch':resample}
        cfg['experiment_plan'] = {'suite':'n_exploration','variant':variant,'base_variant':spec['base_variant'],
                                  'protocol':copy.deepcopy(protocol)}
        result.append({**job,'variant':variant,'name':spec['name'],'config':cfg})
    return result


def manifest(dataset):
    paths = list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts/v24').glob('*.py'))
    paths += [ROOT/'scripts/train.py'] + list(CONFIG_DIR.glob('*.json'))
    paths += list((ROOT/'configs/v24/b3_c3').glob('*.json'))
    paths += [ROOT/f'configs/v24/coe_main_s4_e8_{dataset}_base.json',
              ROOT/f'configs/v24/coe_direct_baselines_{dataset}_experiments.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def evaluate_job(suite, job, gpu):
    receipt = baseline.completed(suite,job)
    if receipt is None:
        raise RuntimeError(f'{job["variant"]} has no completed training')
    destination = suite/'evaluations'/f'{job["variant"]}.json'
    if destination.exists():
        previous = baseline.load(destination)
        if (previous.get('status') == 'finished' and previous.get('run_dir') == receipt['run_dir']
                and previous.get('config_sha256') == baseline.digest(job['config'])):
            return
    check_gpu_idle([gpu])
    command = [sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/evaluate_n_exploration.py'),
               '--config',str(suite/'configs'/f'{job["variant"]}.json'),
               '--checkpoint',str(Path(receipt['run_dir'])/'checkpoints/best.pth'),
               '--test-npz',job['sources']['test'],'--output',str(destination)]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    log_path=suite/'launcher_logs'/f'{job["variant"]}.evaluation.log'
    with log_path.open('ab') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if code or not destination.exists() or baseline.load(destination).get('status') != 'finished':
        raise RuntimeError(f'OOD evaluation failed; see {log_path}')


def diagnose_routes(suite, job, gpu, samples):
    if not samples or job['variant'] != 'N3':
        return
    receipt=baseline.completed(suite,job)
    output=Path(receipt['run_dir'])/'logs/route_candidate_diagnostic.json'
    if output.exists() and baseline.load(output).get('status')=='finished':
        return
    check_gpu_idle([gpu])
    command=[sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/diagnose_n_routes.py'),
             '--config',str(suite/'configs'/f'{job["variant"]}.json'),
             '--checkpoint',str(Path(receipt['run_dir'])/'checkpoints/best.pth'),
             '--val-npz',job['sources']['val'],'--samples',str(samples),'--output',str(output)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    log_path=suite/'launcher_logs'/'N3.route_diagnostic.log'
    with log_path.open('ab') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if code:raise RuntimeError(f'Route diagnostic failed; see {log_path}')


def summarize(suite, plan):
    rows = []
    for job in plan:
        train = baseline.completed(suite,job)
        evaluation = suite/'evaluations'/f'{job["variant"]}.json'
        valid = None
        if train and evaluation.exists():
            candidate = baseline.load(evaluation)
            if candidate.get('run_dir') == train['run_dir'] and candidate.get('status') == 'finished':
                valid = candidate
        rows.append({'variant':job['variant'],'status':'finished' if valid else ('trained' if train else 'pending'),
                     'run_dir':train['run_dir'] if train else None,'best_epoch':train['best_epoch'] if train else None,
                     'val_mae':train['best_val_mae'] if train else None,
                     'training_seconds':train['total_time_sec'] if train else None,
                     'evaluations':{name:{k:r['metrics'][k] for k in ('mae','rmse')} for name,r in valid['sets'].items()} if valid else {}})
    baseline.write(suite/'summary.json',rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('taxibj','bikenyc'),default='taxibj')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--route-diagnostic-samples',type=int,default=96,help='N3 validation-only diagnostic; 0 disables')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.gpu<0 or args.route_diagnostic_samples<0 or len(set(args.variants))!=len(args.variants):
        parser.error('Positive epochs/batch-size, nonnegative GPU and unique variants required')
    plan=jobs(args.dataset,args.epochs,args.batch_size,args.variants)
    source=manifest(args.dataset)
    data={p:{'size':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in plan[0]['sources'].values()}
    fingerprint=baseline.digest({'jobs':plan,'source':source,'data':data,'route_diagnostic_samples':args.route_diagnostic_samples})[:16]
    suite=ROOT/'outputs/v24-COE/experiments/n_exploration'/args.dataset/fingerprint
    if args.dry_run:
        print(json.dumps({'suite':str(suite),'jobs':plan},ensure_ascii=False,indent=2));return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suite.mkdir(exist_ok=True)
        if not args.summary_only:
            for name,sha in source.items():
                target=suite/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen source changed')
                else:shutil.copyfile(ROOT/name,target)
            baseline.write(suite/'plan.json',{'jobs':plan,'source':source,'data':data,'fingerprint':fingerprint,'route_diagnostic_samples':args.route_diagnostic_samples})
            for job in plan:
                for path,stamp in data.items():
                    actual={'size':Path(path).stat().st_size,'mtime_ns':Path(path).stat().st_mtime_ns}
                    if actual!=stamp:raise RuntimeError(f'Dataset changed: {path}')
                try:
                    if not baseline.completed(suite,job):baseline.launch(suite,job,args.gpu)
                    evaluate_job(suite,job,args.gpu)
                    diagnose_routes(suite,job,args.gpu,args.route_diagnostic_samples)
                finally:summarize(suite,plan)
        summarize(suite,plan)
        print(f'Results: {suite / "summary.json"}',flush=True)


if __name__=='__main__':main()
