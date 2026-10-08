#!/usr/bin/env python3
"""V1-V4: serial three-scale output coordination, frozen jobs and resumable training."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
from run_id_priority_exploration import static_jobs as u_jobs
from run_b3_c3 import load,write,digest
CONFIG=ROOT/'configs/v24/coordination_exploration'
ORDER=('V1','V2','V3','V4')


def jobs(dataset='taxibj',epochs=100,batch_size=32):
    origin=u_jobs(dataset,epochs,batch_size)['U02'];result={}
    for n,t in load(CONFIG/'variants.json').items():
        j=copy.deepcopy(origin);c=j['config']
        if t['mode']!='none':c['model']['coe']['coordination']={'enabled':True,'mode':t['mode'],'strength':.1,'aux_weight':.05}
        c['train']['logging']={'diagnostic_every':5,'path_top_k':10}
        c['experiment_plan']={'suite':'coordination_exploration','variant':n,'method':t['method'],'reference':t['reference'],
                              'protocol':load(CONFIG/'test_protocol.json'),'initialization':'common U02 backbone; decoder copies; isolated allocation RNG'}
        j.update(variant=n,name=t['name'],config=c,reference=t['reference']);result[n]=j
    return result


def rows(suite,all_jobs):
    result=[]
    for n,j in all_jobs.items():
        r=base.receipt(suite,j);p=suite/'evaluations'/f'{n}.json';e=load(p) if p.exists() else {}
        result.append({'variant':n,'method':j['config']['experiment_plan']['method'],'reference':j['reference'],
          'status':'finished' if r and e.get('status')=='finished' else 'trained' if r else 'pending',
          'run_dir':r['run_dir'] if r else None,'best_epoch':r['best_epoch'] if r else None,
          'val_mae':r['best_val_mae'] if r else None,'training_seconds':r['total_time_sec'] if r else None,
          'evaluations':{k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in e.get('sets',{}).items()}})
    write(suite/'summary.json',result);return result


def report(suite,all_jobs):
    p=suite/'source_snapshot/scripts/v24/report_coordination_exploration.py'
    spec=importlib.util.spec_from_file_location('frozen_v_report',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.export_report(ROOT,suite,rows(suite,all_jobs),all_jobs)


def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['taxibj','bikenyc'],default='taxibj');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--variants',nargs='+',choices=ORDER,default=list(ORDER));p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true')
    p.add_argument('--suite',type=Path,help='Resume or summarize an existing frozen queue even after workspace changes')
    a=p.parse_args()
    if min(a.epochs,a.batch_size)<1 or a.gpu<0 or len(set(a.variants))!=len(a.variants):p.error('Invalid arguments')
    if a.suite:
        suite=a.suite.resolve();payload=load(suite/'plan.json');all_jobs=payload['jobs']
        cfg=all_jobs['V1']['config']
        if (cfg['train']['epochs'],cfg['data']['batch_size'])!=(a.epochs,a.batch_size):p.error('Frozen budget differs; specify matching --epochs and --batch-size')
        if payload['dataset']!=a.dataset:p.error('Frozen dataset differs')
    else:
        all_jobs=jobs(a.dataset,a.epochs,a.batch_size);sources=base.manifest(a.dataset)
        for f in [ROOT/'tests/test_v24_coordination.py',ROOT/'scripts/v24/README_COORDINATION.md']:
            sources[str(f.relative_to(ROOT))]=filehash(f)
        data={path:{'size':Path(path).stat().st_size,'mtime_ns':Path(path).stat().st_mtime_ns,'sha256':filehash(path)} for path in all_jobs['V1']['sources'].values()}
        payload={'jobs':all_jobs,'source':sources,'data':data,'dataset':a.dataset,'protocol':load(CONFIG/'test_protocol.json')}
        fingerprint=digest(payload)[:16];suite=ROOT/'outputs/v24-COE/experiments/coordination_exploration'/a.dataset/fingerprint
    requested=[n for n in ORDER if n in a.variants]
    if a.dry_run:
        print(json.dumps({'suite':str(suite),'requested':requested,'jobs':all_jobs},ensure_ascii=False,indent=2));return
    if a.summary_only:print(report(suite,all_jobs));return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);suite.mkdir(exist_ok=True)
        for path,sha in payload['source'].items():
            target=suite/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                if a.suite:raise RuntimeError('Frozen source missing: '+str(target))
                shutil.copyfile(ROOT/path,target)
            if filehash(target)!=sha:raise RuntimeError('Snapshot hash mismatch: '+path)
        base.frozen(suite/'plan.json',payload)
        if not (suite/'started.json').exists():write(suite/'started.json',{'time':time.time()})
        for n,j in all_jobs.items():base.frozen(suite/'configs'/f'{n}.json',j['config'])
        for n in requested:
            for path,stamp in payload['data'].items():
                s=Path(path).stat()
                if (s.st_size,s.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Input data changed: '+path)
            try:
                base.launch(suite,all_jobs[n],a.gpu)
                r=base.receipt(suite,all_jobs[n]);print(f'Evaluating {n} (six fixed protocols)',flush=True)
                base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],payload['protocol'],all_jobs[n]['sources']['test'],'evaluations',a.gpu)
            except BaseException as error:
                write(suite/'failures'/f'{n}.json',{'time':time.time(),'error':repr(error)})
                report(suite,all_jobs);raise
            report(suite,all_jobs)
        print(f'Results: {suite}\nReport: {report(suite,all_jobs)}',flush=True)

if __name__=='__main__':main()
