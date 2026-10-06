#!/usr/bin/env python3
"""U01-U30: ID-only selection, matched new references, single-GPU serial queue."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
from run_b3_c3 import load,write,digest
CONFIG=ROOT/'configs/v24/id_priority_exploration'
SCREEN=tuple([f'U{i:02}' for i in range(1,17)]+['U18'])
REPEATS=tuple(f'U{i:02}' for i in range(19,31))
ORDER=SCREEN+('U17','U29','U30')+tuple(f'U{i:02}' for i in range(19,29))
MEMORY=tuple(f'U{i:02}' for i in range(2,8))
SCALE=('U08','U09','U10','U11','U12','U15','U16')
FAMILIES={**dict.fromkeys(('U02','U03','U04'),'state_delta'),**dict.fromkeys(('U05','U06'),'operation_input'),'U07':'operation_router'}


def templates():return {n:load(CONFIG/f'{n}.json') for n in ORDER}


def static_jobs(dataset='taxibj',epochs=100,batch_size=32):
    origin=base.nbase.jobs(dataset,epochs,batch_size,('N3',))[0]
    result={}
    for n in SCREEN:
        t=templates()[n];j=copy.deepcopy(origin);c=j['config'];c['seed']=c['data']['loader_seed']=7
        coe=c['model']['coe'];coe['expert_sharing']='shared'
        coe['four_direction']={'enabled':True,'scale':'fixed'}
        coe['id_priority']={'enabled':True,**t['module']}
        c['experiment_plan']={'suite':'id_priority_exploration','variant':n,'method':t['method'],'reference':t['reference'],'selection_metric':'ID validation MAE only','protocol':load(CONFIG/'test_protocol.json')}
        j.update(variant=n,name=t['name'],config=c,reference=t['reference']);result[n]=j
    return result


def choose(evidence):
    required=('U01',)+MEMORY+SCALE
    if any(n not in evidence or evidence[n].get('split')!='val' for n in required):
        raise ValueError('Selection requires complete validation-only evidence')
    scores={};baseline=evidence['U01']['val_mae']
    for n in MEMORY+SCALE:
        v=evidence[n];ratio=v['val_mae']/baseline
        scores[n]={'val_mae':v['val_mae'],'id_ratio':ratio,'eligible':ratio<=.99,'area':v['area'],'trainable_params':v['trainable_params']}
    key=lambda n:(not scores[n]['eligible'],scores[n]['val_mae'],scores[n]['area'],scores[n]['trainable_params'],n)
    memories=[]
    for n in sorted(MEMORY,key=key):
        if all(FAMILIES[n]!=FAMILIES[v] for v in memories):memories.append(n)
        if len(memories)==2:break
    return {'memory':memories,'scale':sorted(SCALE,key=key)[:2],'scores':scores,'evidence_sha256':digest(evidence),'criterion':'ID validation; >=1% preferred; distinct memory families'}


def resolve(name,jobs,selection,specs=None):
    t=(specs or templates())[name]
    def lookup(s):return selection[s[1:-1]][int(s[-1])] if s.startswith('$') else s
    origin=lookup(t['source']);j=copy.deepcopy(jobs[origin]);c=j['config'];seed=t['seed'];c['seed']=c['data']['loader_seed']=seed
    if t.get('memory_source'):
        mem=jobs[lookup(t['memory_source'])]['config']['model']['coe']['id_priority']
        c['model']['coe']['id_priority'].update(communication=mem['communication'],bound=mem.get('bound',.1))
    description=t['method']+f'; source={origin}'
    if t.get('memory_source'):
        memory_source=lookup(t['memory_source']);description+=f'; memory_source={memory_source}'
        c['experiment_plan']['memory_source']=memory_source
    c['experiment_plan'].update(variant=name,method=description,source=origin,reference=t['reference'])
    if selection is not None:c['experiment_plan']['selection_sha256']=digest(selection)
    j.update(variant=name,name=j['name']+('_combined' if name=='U17' else '_repeat'),reference=t['reference'])
    return j


def collect_evidence(suite,jobs):
    evidence={}
    for n in ('U01',)+MEMORY+SCALE:
        r=base.receipt(suite,jobs[n])
        if r is None:raise RuntimeError('Missing training reference '+n)
        run=Path(r['run_dir']);history=[json.loads(s) for s in (run/'logs/metrics.jsonl').read_text().splitlines()]
        row=next(h for h in history if h['epoch']==r['best_epoch']);val=row['val']
        if val is None or val['mae']!=r['best_val_mae']:raise RuntimeError('Best validation mismatch')
        meta=load(run/'training_metadata.json')
        evidence[n]={'split':'val','val_mae':val['mae'],'area':val['coe_expert_grid_equivalents'],'trainable_params':meta['trainable_params'],
                     'config_sha256':r['config_sha256'],'best_epoch':r['best_epoch'],'validation_sha256':digest(val)}
    return evidence


def selection_record(suite,jobs):
    evidence=collect_evidence(suite,jobs)
    if (suite/'selection.json').exists():
        saved=load(suite/'selection.json')
        if saved['evidence_sha256']!=digest(evidence):raise RuntimeError('Frozen selection evidence changed')
        return saved
    selected=choose(evidence);base.frozen(suite/'selection_evidence.json',evidence);base.frozen(suite/'selection.json',selected);return selected


def manifest(dataset):
    result=base.manifest(dataset)
    for p in [ROOT/'scripts/v24/README_ID_PRIORITY_EXPLORATION.md',ROOT/'tests/test_v24_id_priority.py']:
        result[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def rows(suite,jobs,requested,specs=None):
    result=[];specs=specs or templates()
    for n in requested:
        j=jobs.get(n);r=base.receipt(suite,j) if j else None;p=suite/'evaluations'/f'{n}.json';e=load(p) if p.exists() else {}
        result.append({'variant':n,'status':'finished' if r and e.get('status')=='finished' else 'trained' if r else 'pending' if j else 'awaiting_selection',
          'method':j['config']['experiment_plan']['method'] if j else specs[n]['method'],'reference':j['reference'] if j else specs[n]['reference'],
          'seed':j['config']['seed'] if j else specs[n]['seed'],'run_dir':r['run_dir'] if r else None,
          'best_epoch':r['best_epoch'] if r else None,'val_mae':r['best_val_mae'] if r else None,'training_seconds':r['total_time_sec'] if r else None,
          'evaluations':{k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in e.get('sets',{}).items()}})
    write(suite/'summary.json',result);return result


def report(suite,jobs,requested,label,specs):
    file=suite/'source_snapshot/scripts/v24/report_id_priority_exploration.py'
    s=importlib.util.spec_from_file_location('frozen_u_report',file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.export_report(ROOT,suite,rows(suite,jobs,requested,specs),jobs,label)


def history(dataset):
    if dataset!='taxibj':return {}
    old=ROOT/'outputs/v24-COE/experiments/four_direction_exploration/taxibj/625dc23938a50059'
    later=ROOT/'outputs/v24-COE/experiments/scale_memory_followup/taxibj/018c3ec72556bd07'
    result={}
    for n in ['S01','S11','S12','M07','M15','M16','G12']:
        f=old/'evaluations'/f'{n}.json'
        if f.exists():result[n]=load(f)
    for i in range(1,9):
        f=later/'evaluations'/f'D{i}.json'
        if f.exists():result[f'D{i}']=load(f)
    return result


def requested_names(variants):
    requested=set(variants)
    if requested.intersection({'U17',*REPEATS[:10]}):requested.update(SCREEN);requested.update(('U17','U29','U30'))
    if 'U13' in requested:requested.add('U14')
    requested.add('U01')
    return [n for n in ORDER if n in requested]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['taxibj','bikenyc'],default='taxibj');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--variants',nargs='+',choices=ORDER,default=list(ORDER));p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.batch_size)<1 or a.gpu<0 or len(set(a.variants))!=len(a.variants):p.error('Invalid arguments')
    requested=requested_names(a.variants)
    static=static_jobs(a.dataset,a.epochs,a.batch_size);specs=templates();sources=manifest(a.dataset);historical=history(a.dataset)
    if a.epochs<=20 and not a.dry_run and any(static.get(n,{}).get('config',{}).get('model',{}).get('coe',{}).get('id_priority',{}).get('fixed_epochs',0)>=a.epochs for n in requested):
        p.error('Use >20 epochs to actually exercise staged scale groups')
    paths=static['U01']['sources'];data={v:{'size':Path(v).stat().st_size,'mtime_ns':Path(v).stat().st_mtime_ns} for v in paths.values()}
    payload={'static_jobs':static,'templates':specs,'source':sources,'data':data,'historical':historical,
             'test_protocol':load(CONFIG/'test_protocol.json'),'confirmation_protocol':load(CONFIG/'confirmation_protocol.json')}
    fingerprint=digest(payload)[:16];suite=ROOT/'outputs/v24-COE/experiments/id_priority_exploration'/a.dataset/fingerprint
    if a.dry_run:
        print(json.dumps({'suite':str(suite),'count':len(requested),'requested':requested,'static_jobs':static,'dynamic_templates':{n:t for n,t in specs.items() if n not in SCREEN}},ensure_ascii=False,indent=2));return
    if a.summary_only:
        jobs=load(suite/'resolved_jobs.json');print(report(suite,jobs,requested,'summary',specs));return
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
        jobs=load(suite/'resolved_jobs.json') if (suite/'resolved_jobs.json').exists() else {n:static[n] for n in requested if n in static}
        jobs.update({n:static[n] for n in requested if n in static and n not in jobs})
        write(suite/'resolved_jobs.json',jobs)
        for n,e in historical.items():base.frozen(suite/'historical_evaluations'/f'{n}.json',e)
        selected=None
        for n in requested:
            for path,stamp in data.items():
                s=Path(path).stat()
                if {'size':s.st_size,'mtime_ns':s.st_mtime_ns}!=stamp:raise RuntimeError('Data changed')
            if n not in static:
                if n not in ('U29','U30'):
                    selected=selection_record(suite,jobs)
                job=resolve(n,jobs,None if n in ('U29','U30') else selected,specs)
                if n in jobs and jobs[n]!=job:raise RuntimeError('Resolved configuration changed')
                jobs[n]=job;write(suite/'resolved_jobs.json',jobs)
            job=jobs[n];base.frozen(suite/'configs'/f'{n}.json',job['config'])
            try:
                base.launch(suite,job,a.gpu);r=base.receipt(suite,job)
                base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],payload['test_protocol'],paths['test'],'evaluations',a.gpu)
            except BaseException as error:
                write(suite/'failures'/f'{n}.json',{'error':repr(error),'time':time.time()});raise
            finally:rows(suite,jobs,requested,specs)
            if n in ('U07','U16','U18','U17','U30','U28'):report(suite,jobs,requested,'stage_'+n,specs)
        if set(requested)==set(ORDER):
            selected=selection_record(suite,jobs)
            names={'U01','U17',*REPEATS,*selected['memory'],*selected['scale']}
            confirm=payload['confirmation_protocol'];transfer={'rate':.4,'split':'test','evaluations':{}}
            for rate in (.2,.6,.8):
                for key,spec in {**confirm['evaluations'],'in_distribution':payload['test_protocol']['evaluations']['in_distribution']}.items():
                    transfer['evaluations'][f'{key}_rate{rate}']={**spec,'rate':rate}
            for n in sorted(names):
                r=base.receipt(suite,jobs[n]);cp=Path(r['run_dir'])/'checkpoints/best.pth'
                for group,protocol in [('confirmation',confirm),('rate_transfer',transfer)]:
                    base.evaluate_record(suite,n,cp,r['config_sha256'],protocol,paths['test'],group,a.gpu)
            base.frozen(suite/'final_evaluations_complete.json',{'names':sorted(names),'confirmation_sha256':digest(confirm),'transfer_sha256':digest(transfer)})
        dest=report(suite,jobs,requested,'final' if set(requested)==set(ORDER) else 'subset',specs)
        print(f'Results: {suite / "summary.json"}\nReport: {dest}',flush=True)

if __name__=='__main__':main()
