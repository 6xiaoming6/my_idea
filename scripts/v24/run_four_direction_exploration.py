#!/usr/bin/env python3
"""60 frozen M/S/G/P jobs, single GPU, full-epoch resume and validation-only selection."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,os,shutil,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from run_b3_c3 import load,write,digest
import run_n_exploration as nbase
import run_r_exploration as rbase
from run_team_accept_v4 import check_gpu_idle
CONFIG=ROOT/'configs/v24/four_direction_exploration'
STATIC=tuple([f'M{i:02}' for i in range(1,13)]+[f'S{i:02}' for i in range(1,11)]+[f'G{i:02}' for i in range(1,10)]+[f'P{i:02}' for i in range(1,9)])
DYNAMIC=tuple([f'M{i:02}' for i in range(13,19)]+[f'S{i:02}' for i in range(11,19)]+['G10','G11']+[f'P{i:02}' for i in range(9,13)]+['G12'])
ORDER=STATIC+DYNAMIC
DEV=('unseen_combinations','unseen_geometry','unseen_triple')


def templates():return {n:load(CONFIG/f'{n}.json') for n in ORDER}


def static_jobs(dataset='taxibj',epochs=100,batch_size=32):
    result={};test=load(CONFIG/'test_protocol.json')
    for name in STATIC:
        t=templates()[name]
        job=(rbase.jobs(dataset,epochs,batch_size,(t['legacy'],))[0] if t.get('legacy') else nbase.jobs(dataset,epochs,batch_size,('N3',))[0])
        c=job['config'];seed=t.get('seed',7);c['seed']=seed;c['data']['loader_seed']=seed
        coe=c['model']['coe'];coe['expert_sharing']=t.get('sharing','shared')
        if 'module' in t:coe['four_direction']={'enabled':True,**t['module']}
        c['experiment_plan']={'suite':'four_direction_exploration','variant':name,'method':t['method'],'reference':t['reference'],'protocol':test}
        result[name]={**job,'variant':name,'name':t['name'],'config':c,'reference':t['reference']}
    return result


def choose(evidence):
    # Only post-training validation split evaluations are accepted as selection evidence.
    scores={}
    def rank(names,reference,count=1):
        for name in names:
            ref=reference[name] if isinstance(reference,dict) else reference
            a,b=evidence[name],evidence[ref]
            if a['split']!='val' or b['split']!='val':raise ValueError('Selection cannot read test evidence')
            ratio=a['val_mae']/b['val_mae'];ood=sum(a['ood'][k]/b['ood'][k] for k in DEV)/3
            scores[name]={'reference':ref,'id_ratio':ratio,'ood_ratio':ood,'J':.5*(ratio+ood),'eligible':ratio<=1.02}
        eligible=[n for n in names if scores[n]['eligible']]
        key=lambda n:(scores[n]['J'],evidence[n]['area'],evidence[n]['params'],n)
        order=sorted(eligible,key=key)+sorted([n for n in names if n not in eligible],key=key)
        return order[:count]
    memory=rank([f'M{i:02}' for i in range(3,13)],'M01',2)
    scale=rank([f'S{i:02}' for i in range(2,7)],'S01')[0]
    mixture=rank(['S07','S09'],{'S07':'S08','S09':'S10'})[0]
    general=rank(['G07','G08','G09'],'S01')[0]
    pair=rank(['P02','P03','P04','P05','P06','P08'],'M01',2)
    return {'memory':memory,'scale':scale,'mixture':mixture,'mixture_reference':'S08' if mixture=='S07' else 'S10',
            'generalization':general,'pair':pair,'scores':scores,'evidence_sha256':digest(evidence)}


def resolve(name,static,selected,specs=None):
    t=(specs or templates())[name];source=t['source']
    def lookup(s):
        if not s.startswith('$'):return s
        k=s[1:]
        if k in ('memory0','memory1','pair0','pair1'):return selected[k[:-1]][int(k[-1])]
        return selected[k]
    origin=lookup(source);job=copy.deepcopy(static[origin]);cfg=job['config'];seed=t['seed']
    cfg['seed']=seed;cfg['data']['loader_seed']=seed
    if t.get('memory_source'):
        other=lookup(t['memory_source']);cfg['model']['coe']['four_direction']['memory']=static[other]['config']['model']['coe']['four_direction']['memory']
    refs={'M15':'M13','M16':'M14','M17':'M13','M18':'M14','S13':'S11','S14':'S12','S15':'S17','S16':'S18','G10':'S11','G11':'S12','P09':'M13','P10':'M14','P11':'M13','P12':'M14'}
    ref=refs.get(name,origin)
    cfg['experiment_plan'].update(variant=name,method=f"{t['method']}; resolved source={origin}",source=origin,reference=ref,selection_sha256=digest(selected))
    job.update(variant=name,name=job['name']+('_combined' if name=='G12' else '_repeat'),reference=ref)
    return job


def frozen(path,value):
    if path.exists():
        if load(path)!=value:raise RuntimeError(f'Frozen record changed: {path}')
    else:write(path,value)


def manifest(dataset):
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts/v24').glob('*.py'))+[ROOT/'scripts/train.py']
    files+=list((ROOT/'configs/v24').rglob('*.json'))+list((ROOT/'tests').glob('test_v24_four*.py'))
    files += [ROOT/'scripts/v24/README_FOUR_DIRECTION_EXPLORATION.md']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def references(dataset):
    if dataset!='taxibj':return {}
    old=ROOT/'outputs/v24-COE/experiments/r_exploration/taxibj/b21d7928e67f4565'
    paths={'N5':old/'reference_evaluations/reference_N5.json',**{n:old/'evaluations'/f'{n}.json' for n in ('R2','R5','R6')}}
    result={}
    for n,p in paths.items():
        r=load(p);cp=Path(r['checkpoint']);st=cp.stat()
        result[n]={'checkpoint':str(cp),'config_sha256':r['config_sha256'],'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    return result


def receipt(suite,job):
    p=suite/'results'/f"{job['variant']}.json"
    if not p.exists():return None
    r=load(p)
    if r.get('status')!='finished' or r['config_sha256']!=digest(job['config']):raise RuntimeError('Invalid training receipt')
    if r['completed_epochs']!=job['config']['train']['epochs']:raise RuntimeError('Incomplete training receipt')
    if not all((Path(r['run_dir'])/'checkpoints'/n).exists() for n in ('best.pth','last.pth')):raise RuntimeError('Completed checkpoint missing')
    return r


def subprocess_run(command,logpath,gpu,visible=False):
    check_gpu_idle([gpu]);logpath.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with logpath.open('ab') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE if visible else log,stderr=subprocess.STDOUT)
        try:
            if visible:
                while chunk:=os.read(process.stdout.fileno(),4096):log.write(chunk);log.flush();sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
            code=process.wait()
        except BaseException:process.terminate();process.wait();raise
    if code:raise RuntimeError(f'Child failed (exit {code}); see {logpath}')


def launch(suite,job,gpu):
    if receipt(suite,job):return
    name=job['variant'];pointer=suite/'runs'/f'{name}.json'
    if pointer.exists():run=Path(load(pointer)['run_dir'])
    else:
        cfg=job['config'];stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        run=Path(cfg['output_dir'])/cfg['data']['dataset_name']/'custom'/f"{stamp}_{job['name']}_seed{cfg['seed']}"/'random'/'rate0.4'/f"{stamp}_seed{cfg['seed']}_bs{cfg['data']['batch_size']}"
        frozen(pointer,{'run_dir':str(run)})
    frozen(suite/'jobs'/f'{name}.json',job)
    command=[sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/train_four_direction.py'),'--job',str(suite/'jobs'/f'{name}.json'),'--run-dir',str(run),'--result',str(suite/'results'/f'{name}.json')]
    print(f'Starting {name} {job["name"]} on GPU {gpu}',flush=True)
    subprocess_run(command,suite/'launcher_logs'/f'{name}.train.log',gpu,visible=True)
    if not receipt(suite,job):raise RuntimeError('Training did not produce receipt')


def evaluate_record(suite,name,cp,sha,protocol,npz,group,gpu):
    output=suite/group/f'{name}.json';ph=digest(protocol)
    if output.exists():
        r=load(output)
        if r.get('status')=='finished' and r['protocol_sha256']==ph and r['config_sha256']==sha and r['checkpoint']==str(Path(cp).resolve()) and r.get('npz')==str(Path(npz).resolve()):return r
    protocolpath=suite/'protocols'/f'{group}.json';frozen(protocolpath,protocol)
    subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/evaluate_four_direction.py'),
        '--checkpoint',str(cp),'--protocol',str(protocolpath),'--npz',npz,'--output',str(output),'--expected-sha',sha],suite/'launcher_logs'/f'{name}.{group}.log',gpu)
    r=load(output)
    if r['status']!='finished':raise RuntimeError('Incomplete evaluation')
    return r


def collect_evidence(suite,jobs):
    evidence={}
    for n in STATIC:
        r=receipt(suite,jobs[n]);p=suite/'development'/f'{n}.json'
        if r is None or not p.exists():raise RuntimeError('Incomplete development evidence')
        e=load(p)
        if e['status']!='finished' or e['split']!='val':raise RuntimeError('Invalid development split')
        meta=load(Path(r['run_dir'])/'training_metadata.json')
        evidence[n]={'split':'val','val_mae':r['best_val_mae'],'ood':{k:e['sets'][k]['metrics']['mae'] for k in DEV},
            'area':e['sets'][DEV[0]]['metrics'].get('coe_expert_grid_equivalents',8.),'params':meta['total_params'],
            'config_sha256':r['config_sha256'],'development_sha256':digest(e)}
    return evidence


def rows(suite,jobs,requested):
    result=[]
    for n in requested:
        job=jobs.get(n);r=receipt(suite,job) if job else None
        evalpath=suite/'evaluations'/f'{n}.json';devpath=suite/'development'/f'{n}.json'
        e=load(evalpath) if evalpath.exists() else {};d=load(devpath) if devpath.exists() else {}
        status='finished' if r and e.get('status')==d.get('status')=='finished' else 'trained' if r else 'pending' if job else 'awaiting_selection'
        result.append({'variant':n,'status':status,'method':job['config']['experiment_plan']['method'] if job else templates()[n]['method'],
            'reference':job.get('reference') if job else None,'seed':job['config']['seed'] if job else None,
            'run_dir':r['run_dir'] if r else None,'val_mae':r['best_val_mae'] if r else None,'best_epoch':r['best_epoch'] if r else None,
            'training_seconds':r['total_time_sec'] if r else None,
            'evaluations':{k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in e.get('sets',{}).items()}})
    write(suite/'summary.json',result);return result


def reporting(suite,jobs,requested,label):
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_four_report',suite/'source_snapshot/scripts/v24/report_four_direction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    r=rows(suite,jobs,requested);module.export_report(ROOT,suite,r,jobs,label)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['taxibj','bikenyc'],default='taxibj');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--variants',nargs='+',choices=ORDER,default=list(ORDER));p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if a.gpu<0 or min(a.epochs,a.batch_size)<1 or len(set(a.variants))!=len(a.variants):p.error('Invalid arguments')
    requested=set(a.variants)
    if requested.intersection(DYNAMIC):requested.update(STATIC)
    requested=[n for n in ORDER if n in requested]
    static=static_jobs(a.dataset,a.epochs,a.batch_size);specs=templates();sources=manifest(a.dataset);refs=references(a.dataset)
    data={v:{'size':Path(v).stat().st_size,'mtime_ns':Path(v).stat().st_mtime_ns} for v in static['M01']['sources'].values()}
    payload={'static_jobs':static,'templates':specs,'source':sources,'data':data,'references':refs}
    fingerprint=digest(payload)[:16];suite=ROOT/'outputs/v24-COE/experiments/four_direction_exploration'/a.dataset/fingerprint
    if a.dry_run:
        import json
        print(json.dumps({'suite':str(suite),'requested':requested,'count':len(requested),'static_jobs':static,'dynamic_templates':{n:specs[n] for n in DYNAMIC}},ensure_ascii=False,indent=2));return
    if a.summary_only:
        if not suite.exists():raise FileNotFoundError(suite)
        jobs=load(suite/'resolved_jobs.json');reporting(suite,jobs,requested,'summary');print(suite/'summary.json');return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);suite.mkdir(exist_ok=True)
        for path,sha in sources.items():
            target=suite/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise RuntimeError('Snapshot changed')
            else:shutil.copyfile(ROOT/path,target)
        frozen(suite/'plan.json',{**payload,'fingerprint':fingerprint})
        started=load(suite/'started.json')['time'] if (suite/'started.json').exists() else time.time();frozen(suite/'started.json',{'time':started})
        jobs=load(suite/'resolved_jobs.json') if (suite/'resolved_jobs.json').exists() else {n:static[n] for n in requested if n in STATIC}
        write(suite/'resolved_jobs.json',jobs)
        test=load(CONFIG/'test_protocol.json');dev=load(CONFIG/'development_protocol.json');source_paths=static['M01']['sources']
        for n,r in refs.items():
            st=Path(r['checkpoint']).stat()
            if (st.st_size,st.st_mtime_ns)!=(r['size'],r['mtime_ns']):raise RuntimeError('Reference changed')
            for group,protocol,split in [('reference_evaluations',test,'test'),('reference_development',dev,'val')]:
                evaluate_record(suite,n,r['checkpoint'],r['config_sha256'],protocol,source_paths[split],group,a.gpu)
        for name in requested:
            for path,stamp in data.items():
                st=Path(path).stat()
                if {'size':st.st_size,'mtime_ns':st.st_mtime_ns}!=stamp:raise RuntimeError('Data changed')
            if name in DYNAMIC:
                evidence=collect_evidence(suite,static);selection=choose(evidence);frozen(suite/'selection.json',selection)
                job=resolve(name,static,selection,specs)
                if name in jobs and jobs[name]!=job:raise RuntimeError('Resolved config changed')
                jobs[name]=job;write(suite/'resolved_jobs.json',jobs)
            job=jobs[name];frozen(suite/'configs'/f'{name}.json',job['config'])
            try:
                launch(suite,job,a.gpu);r=receipt(suite,job);cp=Path(r['run_dir'])/'checkpoints/best.pth'
                evaluate_record(suite,name,cp,r['config_sha256'],dev,job['sources']['val'],'development',a.gpu)
                evaluate_record(suite,name,cp,r['config_sha256'],test,job['sources']['test'],'evaluations',a.gpu)
            finally:
                rows(suite,jobs,requested)
            if name in ('M12','S10','G09','P08','G12'):reporting(suite,jobs,requested,'stage_'+name)
            if time.time()-started>=48*3600 and not (suite/'reported_48h.json').exists():
                reporting(suite,jobs,requested,'48h');write(suite/'reported_48h.json',{'time':time.time()})
        if set(requested)==set(ORDER):
            selected=load(suite/'selection.json');names=set(DYNAMIC)-{'G12'}
            names.update(['M01','S01',*selected['memory'],selected['scale'],selected['mixture'],selected['mixture_reference'],selected['generalization'],*selected['pair'],'G01','G02','G03','G04','G05','G06'])
            confirm=load(CONFIG/'confirmation_protocol.json')
            transfer={'rate':.4,'split':'test','evaluations':{}}
            for rate in (.2,.6,.8):
                for key,spec in {**confirm['evaluations'],'in_distribution':test['evaluations']['in_distribution']}.items():
                    transfer['evaluations'][f'{key}_rate{rate}']={**spec,'rate':rate}
            for n in sorted(names):
                r=receipt(suite,jobs[n]);cp=Path(r['run_dir'])/'checkpoints/best.pth'
                for group,protocol in [('confirmation',confirm),('rate_transfer',transfer)]:evaluate_record(suite,n,cp,r['config_sha256'],protocol,source_paths['test'],group,a.gpu)
            for n,r in refs.items():
                for group,protocol in [('confirmation',confirm),('rate_transfer',transfer)]:evaluate_record(suite,n,r['checkpoint'],r['config_sha256'],protocol,source_paths['test'],group,a.gpu)
        reporting(suite,jobs,requested,'final' if set(requested)==set(ORDER) else 'subset')
        print(f'Results: {suite / "summary.json"}',flush=True)

if __name__=='__main__':main()
