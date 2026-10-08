#!/usr/bin/env python3
"""W01-W30, fixed CMFF U02, exact replay gate and ID-only candidate selection."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
from run_id_priority_exploration import static_jobs as u_jobs
from run_coordination_exploration import filehash
from run_b3_c3 import load,write,digest
CONFIG=ROOT/'configs/v24/backbone_exploration';ORDER=tuple(f'W{i:02}' for i in range(1,31));STATIC=ORDER[:26]
GROUPS={'A':ORDER[2:10],'B':ORDER[10:18],'C':ORDER[18:26]}


def jobs(dataset='taxibj',epochs=100,batch_size=32):
    origin=u_jobs(dataset,epochs,batch_size)['U02'];result={};templates=load(CONFIG/'variants.json')
    for n in STATIC:
        t=templates[n];j=copy.deepcopy(origin);c=j['config'];c['model']['coe']['backbone_exploration']={'enabled':True,**t['module']}
        c['train'].update(strict_replay=True,logging={'diagnostic_every':5,'path_top_k':10})
        c['experiment_plan']={'suite':'backbone_exploration','variant':n,'method':t['method'],'reference':t['reference'],
          'protocol':load(CONFIG/'test_protocol.json'),'selection':'ID validation only','seed_note':'W01/W02 exact replay, not independent seeds'}
        j.update(variant=n,name=t['name'],config=c,reference=t['reference']);result[n]=j
    return result


def choose(evidence):
    if set(evidence)!=set(STATIC) or any(e['split']!='val' for e in evidence.values()):raise ValueError('All 26 validation records required')
    ref=evidence['W01']['mae'];scores={};selected={}
    for group,names in GROUPS.items():
        for n in names:scores[n]={**evidence[n],'eligible':evidence[n]['mae']<=.99*ref,'id_ratio':evidence[n]['mae']/ref}
        selected[group]=min(names,key=lambda n:(not scores[n]['eligible'],scores[n]['mae'],scores[n]['trainable_params'],n))
    return {'selected':selected,'scores':scores,'evidence_sha256':digest(evidence)}


def freeze_selection(suite,all_jobs):
    evidence={}
    for n in STATIC:
        r=base.receipt(suite,all_jobs[n])
        if r is None:raise RuntimeError('Missing training receipt '+n)
        run=Path(r['run_dir']);h=[json.loads(x) for x in (run/'logs/metrics.jsonl').read_text().splitlines()]
        v=next(x['val'] for x in h if x['epoch']==r['best_epoch'])
        if v is None or v['mae']!=r['best_val_mae']:raise RuntimeError('Validation mismatch')
        evidence[n]={'split':'val','mae':v['mae'],'trainable_params':load(run/'training_metadata.json')['trainable_params'],'config_sha256':r['config_sha256'],'best_epoch':r['best_epoch'],'val_sha256':digest(v)}
    p=suite/'selection.json'
    if p.exists():
        s=load(p)
        if s['evidence_sha256']!=digest(evidence):raise RuntimeError('Frozen selection changed')
        return s
    s=choose(evidence);base.frozen(suite/'selection_evidence.json',evidence);base.frozen(p,s);return s


def resolve(n,all_jobs,selection,templates):
    t=templates[n];j=copy.deepcopy(all_jobs['W01']);module={'enabled':True};sources=[selection['selected'][g] for g in t['sources']]
    for source in sources:
        extra=all_jobs[source]['config']['model']['coe']['backbone_exploration']
        for k,v in extra.items():
            if k in module and module[k]!=v:raise ValueError('Overlapping mechanisms disagree')
            module[k]=v
    j['config']['model']['coe']['backbone_exploration']=module
    j['config']['experiment_plan'].update(variant=n,method=t['method']+'; '+','.join(sources),sources=sources,selection_sha256=digest(selection))
    j.update(variant=n,name=t['name'],reference='W01');return j


def replay_gate(suite,all_jobs):
    runs=[Path(base.receipt(suite,all_jobs[n])['run_dir']) for n in ('W01','W02')]
    audits=[load(r/'replay_audit.json') for r in runs];initial=[load(r/'initialization.json') for r in runs]
    expected=all_jobs['W01']['config']['train']['epochs']
    passed=len(audits[0])==len(audits[1])==expected and audits[0]==audits[1] and initial[0]==initial[1]
    record={'passed':passed,'epochs':expected,'audit_hashes':[digest(a) for a in audits],'initializations':initial}
    if not passed:
        record['mismatching_epochs']=[a['epoch'] for a,b in zip(*audits) if a!=b];write(suite/'replay_gate.json',record)
        raise RuntimeError('W01/W02 did not replay exactly; remaining experiments stopped')
    base.frozen(suite/'replay_gate.json',record);return record


def rows(suite,all_jobs,templates):
    result=[]
    for n in ORDER:
        j=all_jobs.get(n);r=base.receipt(suite,j) if j else None;p=suite/'evaluations'/f'{n}.json';e=load(p) if p.exists() else {}
        pointer=suite/'runs'/f'{n}.json';partial=load(pointer)['run_dir'] if pointer.exists() else None
        status='finished' if r and e.get('status')=='finished' else 'trained' if r else 'failed' if (suite/'failures'/f'{n}.json').exists() else 'partial' if partial else 'pending' if j else 'awaiting_selection'
        result.append({'variant':n,'method':j['config']['experiment_plan']['method'] if j else templates[n]['method'],
          'reference':j['reference'] if j else 'W01','status':status,
          'run_dir':r['run_dir'] if r else partial,'best_epoch':r['best_epoch'] if r else None,'val_mae':r['best_val_mae'] if r else None,
          'training_seconds':r['total_time_sec'] if r else None,'evaluations':{k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in e.get('sets',{}).items()}})
    write(suite/'summary.json',result);return result


def report(suite,all_jobs,templates):
    path=suite/'source_snapshot/scripts/v24/report_backbone_exploration.py'
    s=importlib.util.spec_from_file_location('frozen_w_report',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.export_report(ROOT,suite,rows(suite,all_jobs,templates),all_jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['taxibj','bikenyc'],default='taxibj');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--variants',nargs='+',choices=ORDER,default=list(ORDER));p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true');p.add_argument('--suite',type=Path)
    a=p.parse_args()
    if a.gpu<0 or min(a.epochs,a.batch_size)<1 or len(set(a.variants))!=len(a.variants):p.error('Invalid arguments')
    requested=set(a.variants)|{'W01','W02'}
    if requested.intersection(ORDER[26:]):requested.update(STATIC)
    requested=[n for n in ORDER if n in requested]
    if a.suite:
        suite=a.suite.resolve();payload=load(suite/'plan.json');all_jobs=copy.deepcopy(payload['jobs']);templates=payload['templates']
        cfg=all_jobs['W01']['config']
        if (a.dataset,a.epochs,a.batch_size)!=(payload['dataset'],cfg['train']['epochs'],cfg['data']['batch_size']):p.error('Frozen dataset/budget differs')
    else:
        all_jobs=jobs(a.dataset,a.epochs,a.batch_size);sources=base.manifest(a.dataset);templates=load(CONFIG/'variants.json')
        for f in [ROOT/'tests/test_v24_backbone.py',ROOT/'scripts/v24/README_BACKBONE_EXPLORATION.md']:sources[str(f.relative_to(ROOT))]=filehash(f)
        data={v:{'size':Path(v).stat().st_size,'mtime_ns':Path(v).stat().st_mtime_ns,'sha256':filehash(v)} for v in all_jobs['W01']['sources'].values()}
        payload={'jobs':all_jobs,'source':sources,'templates':templates,'data':data,'dataset':a.dataset,'protocol':load(CONFIG/'test_protocol.json'),'confirmation':load(CONFIG/'confirmation_protocol.json')}
        suite=ROOT/'outputs/v24-COE/experiments/backbone_exploration'/a.dataset/digest(payload)[:16]
    if a.dry_run:print(json.dumps({'suite':str(suite),'requested':requested,'jobs':all_jobs,'templates':templates},ensure_ascii=False,indent=2));return
    if (suite/'resolved_jobs.json').exists():all_jobs=load(suite/'resolved_jobs.json')
    if a.summary_only:print(report(suite,all_jobs,templates));return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);suite.mkdir(exist_ok=True)
        for path,sha in payload['source'].items():
            target=suite/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                if a.suite:raise RuntimeError('Frozen source missing')
                shutil.copyfile(ROOT/path,target)
            if filehash(target)!=sha:raise RuntimeError('Snapshot changed: '+path)
        base.frozen(suite/'plan.json',payload)
        for job in all_jobs.values():job['common_initialization']=str(suite/'common_initialization.pth')
        write(suite/'resolved_jobs.json',all_jobs)
        if not (suite/'started.json').exists():write(suite/'started.json',{'time':time.time()})
        for n in requested:
            for path,stamp in payload['data'].items():
                st=Path(path).stat()
                if (st.st_size,st.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Dataset changed')
            try:
                if n not in ('W01','W02'):replay_gate(suite,all_jobs)
                if n in ORDER[26:]:
                    selection=freeze_selection(suite,all_jobs);j=resolve(n,all_jobs,selection,templates)
                    if n in all_jobs and all_jobs[n]!=j:raise RuntimeError('Frozen dynamic configuration changed')
                    all_jobs[n]=j;write(suite/'resolved_jobs.json',all_jobs)
                j=all_jobs[n];base.frozen(suite/'configs'/f'{n}.json',j['config']);base.launch(suite,j,a.gpu)
                if n=='W02':replay_gate(suite,all_jobs)
                r=base.receipt(suite,j)
                base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],payload['protocol'],j['sources']['test'],'evaluations',a.gpu)
            except BaseException as error:
                write(suite/'failures'/f'{n}.json',{'time':time.time(),'error':repr(error)});report(suite,all_jobs,templates);raise
            rows(suite,all_jobs,templates)
            if n in ('W02','W10','W18','W26','W30'):report(suite,all_jobs,templates)
        if set(requested)==set(ORDER):
            selection=freeze_selection(suite,all_jobs);names=sorted({'W01',*selection['selected'].values(),*ORDER[26:]})
            for n in names:
                r=base.receipt(suite,all_jobs[n]);base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],payload['confirmation'],all_jobs[n]['sources']['test'],'confirmation',a.gpu)
            base.frozen(suite/'confirmation_complete.json',{'names':names,'protocol_sha256':digest(payload['confirmation'])})
        print(f'Results: {suite}\nReport: {report(suite,all_jobs,templates)}',flush=True)

if __name__=='__main__':main()
