#!/usr/bin/env python3
"""Eight controlled fixed-scale / cross-round-delta experiments; GPU0 serial."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
from run_b3_c3 import load,write,digest
CONFIG=ROOT/'configs/v24/scale_memory_followup'
OLD=ROOT/'outputs/v24-COE/experiments/four_direction_exploration/taxibj/625dc23938a50059'
ORDER=tuple(f'D{i}' for i in range(1,9))


def jobs(dataset='taxibj',epochs=100,batch_size=32):
    origin=base.static_jobs(dataset,epochs,batch_size)['S01'];result={}
    for name in ORDER:
        t=load(CONFIG/f'{name}.json');job=copy.deepcopy(origin);c=job['config']
        c['seed']=c['data']['loader_seed']=t['seed']
        c['model']['coe']['four_direction']={'enabled':True,'scale':'fixed','memory':t['memory'],'fixed_path':[{'F':0,'M':1,'C':2}[v] for v in t['path']]}
        c['experiment_plan'].update(suite='scale_memory_followup',variant=name,method=t['method'],reference=t['reference'])
        job.update(variant=name,name=t['path'].lower()+('_delta' if t['memory']=='delta' else '_native'),reference=t['reference'])
        result[name]=job
    return result


def source_manifest(dataset):
    result=base.manifest(dataset)
    for f in [ROOT/'scripts/v24/README_SCALE_MEMORY_FOLLOWUP.md',ROOT/'tests/test_v24_scale_memory_followup.py']:
        result[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    return result


def old_references():
    out={}
    for name in ('S01','S11','S12','G12'):
        r=load(OLD/'results'/f'{name}.json');cp=Path(r['run_dir'])/'checkpoints/best.pth'
        if r['completed_epochs']!=100 or not cp.exists():raise RuntimeError('Missing completed reference '+name)
        out[name]={'receipt':r,'checkpoint':str(cp),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
                   'config':load(Path(r['run_dir'])/'config.json'),'evaluation':load(OLD/'evaluations'/f'{name}.json')}
    return out


def collect(suite,all_jobs,names):
    result=[]
    for n in names:
        j=all_jobs[n];r=base.receipt(suite,j);p=suite/'evaluations'/f'{n}.json';e=load(p) if p.exists() else {}
        complete=r and all((suite/f/f'{n}.json').exists() and load(suite/f/f'{n}.json').get('status')=='finished' for f in ('evaluations','confirmation','rate_transfer'))
        result.append({'variant':n,'status':'finished' if complete else 'trained' if r else 'pending','method':j['config']['experiment_plan']['method'],'reference':j['reference'],'seed':j['config']['seed'],
          'run_dir':r['run_dir'] if r else None,'best_epoch':r['best_epoch'] if r else None,'val_mae':r['best_val_mae'] if r else None,'training_seconds':r['total_time_sec'] if r else None,
          'evaluations':{k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in e.get('sets',{}).items()}})
    write(suite/'summary.json',result);return result


def report(suite,all_jobs,names):
    from report_scale_memory_followup import export_report
    return export_report(ROOT,suite,collect(suite,all_jobs,names),all_jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['taxibj'],default='taxibj');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--variants',nargs='+',choices=ORDER,default=list(ORDER));p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if a.gpu!=0:p.error('This authorized queue uses GPU0 only')
    if a.epochs!=100 or a.batch_size!=32:p.error('Matched completed references require 100 epochs and batch32')
    if len(set(a.variants))!=len(a.variants):p.error('Duplicate variants')
    all_jobs=jobs(a.dataset,a.epochs,a.batch_size);names=[n for n in ORDER if n in a.variants]
    for n in names:
        ref=all_jobs[n]['reference']
        if ref in ORDER and ref not in names:names.append(ref)
    names=[n for n in ORDER if n in names]
    sources=source_manifest(a.dataset);refs=old_references();paths=all_jobs['D1']['sources']
    data={v:{'size':Path(v).stat().st_size,'mtime_ns':Path(v).stat().st_mtime_ns} for v in paths.values()}
    payload={'jobs':all_jobs,'source':sources,'data':data,'references':refs,'confirmation':load(CONFIG/'confirmation_protocol.json')}
    fingerprint=digest(payload)[:16];suite=ROOT/'outputs/v24-COE/experiments/scale_memory_followup'/a.dataset/fingerprint
    if a.dry_run:
        import json
        print(json.dumps({'suite':str(suite),'requested':names,'jobs':all_jobs},ensure_ascii=False,indent=2));return
    if a.summary_only:
        print(report(suite,all_jobs,names));return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);suite.mkdir(exist_ok=True)
        for path,sha in sources.items():
            target=suite/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen source changed')
            else:shutil.copyfile(ROOT/path,target)
        base.frozen(suite/'plan.json',{**payload,'fingerprint':fingerprint})
        if not (suite/'started.json').exists():base.frozen(suite/'started.json',{'time':time.time()})
        write(suite/'resolved_jobs.json',all_jobs)
        for n,ref in refs.items():
            base.frozen(suite/'references'/f'{n}.json',ref)
            base.frozen(suite/'reference_evaluations'/f'{n}.json',ref['evaluation'])
        test=load(ROOT/'configs/v24/four_direction_exploration/test_protocol.json');confirm=payload['confirmation']
        transfer={'rate':.4,'split':'test','evaluations':{}}
        for rate in (.2,.6,.8):
            for key,spec in {**confirm['evaluations'],'in_distribution':test['evaluations']['in_distribution']}.items():
                transfer['evaluations'][f'{key}_rate{rate}']={**spec,'rate':rate}
        for name in names:
            for path,stamp in data.items():
                s=Path(path).stat()
                if {'size':s.st_size,'mtime_ns':s.st_mtime_ns}!=stamp:raise RuntimeError('Data changed')
            job=all_jobs[name];base.frozen(suite/'configs'/f'{name}.json',job['config'])
            try:
                base.launch(suite,job,a.gpu);r=base.receipt(suite,job);cp=Path(r['run_dir'])/'checkpoints/best.pth'
                for group,protocol in [('evaluations',test),('confirmation',confirm),('rate_transfer',transfer)]:
                    base.evaluate_record(suite,name,cp,r['config_sha256'],protocol,paths['test'],group,a.gpu)
            finally:report(suite,all_jobs,names)
        # Previously examined masks are retained as development evidence; all references
        # receive the same new confirmation and missing-rate masks as the eight jobs.
        for n,ref in refs.items():
            if hashlib.sha256(Path(ref['checkpoint']).read_bytes()).hexdigest()!=ref['checkpoint_sha256']:raise RuntimeError('Reference checkpoint changed')
            for group,protocol in [('confirmation',confirm),('rate_transfer',transfer)]:
                base.evaluate_record(suite,n,ref['checkpoint'],ref['receipt']['config_sha256'],protocol,paths['test'],group,a.gpu)
        dest=report(suite,all_jobs,names);print(f'Results: {suite / "summary.json"}\nReport: {dest}',flush=True)

if __name__=='__main__':main()
