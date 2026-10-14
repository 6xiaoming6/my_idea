#!/usr/bin/env python3
"""BikeNYC/CHAP four-cell replications, frozen GPU1 serial queue."""
import argparse,copy,fcntl,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
import run_four_direction_exploration as base
from run_core_validation import prepare_initialization
from run_first_validation import METHODS,protocol,verify
from prepare_cross_dataset import SPECS,prepare
from prepare_data import sha
BIKE100=ROOT/'outputs/v25/experiments/cross_dataset_validation/65564fa00949e268'
REFERENCE=ROOT/'outputs/v25/experiments/first_validation/659d3ba9678e1dca'

def jobs(roots,project_root=ROOT,epochs=100):
    template=load(ROOT/'configs/v25/base.json');result={}
    for dataset,data_root in roots.items():
        spec=SPECS[dataset];manifest=load(Path(data_root)/'manifest.json')
        _,channels,t,h,w=manifest['outputs']['train']['shapes']['F']
        for method,(scales,communication) in METHODS.items():
            c=copy.deepcopy(template);c['output_dir']=str(project_root/'outputs/v25')
            c['train']['epochs']=epochs;c['train']['scheduler']['total_epochs']=epochs
            c['data'].update(dataset_name=spec['name'],input_scales=scales)
            c['data']['scales']['pooling_mode']=spec['pooling']
            c['data']['synthetic'].update(t=t,h=h,w=w)
            c['model']['c_in']=channels;c['model']['main'].update(max_t=t,h=h,w=w)
            c['model']['coe']['core_validation']['communication']=communication
            n=f'{dataset}_{method}_seed7'
            c['experiment_plan']={'suite':'v25_cross_dataset_validation','variant':n,
                'method':f'{dataset}: {scales} input, CMFF, delta={communication}',
                'protocol':protocol(),'selection':'ID validation MAE only','gpu':1,
                'data_manifest':str(Path(data_root)/'manifest.json'),
                'scope':'four factorial cells per dataset; no free scales/history/interaction/FFN'}
            result[n]={'variant':n,'name':f'{method.lower()}_v25_cross_dataset_e{epochs}','method':method,'dataset':dataset,
                'reference':{'A1':'none','A2':'A1','B1':'A1','B2':'B1 / A2'}[method],'config':c,
                'sources':{s:manifest['outputs'][s]['path'] for s in ('train','val','test')}}
    return result

def initialize(suite,js):
    combined={}
    for dataset in dict.fromkeys(j['dataset'] for j in js.values()):
        group={n:j for n,j in js.items() if j['dataset']==dataset}
        prepare_initialization(suite/'initialization_groups'/dataset,group)
        combined.update(load(suite/'initialization_groups'/dataset/'initialization_audit.json'))
    base.frozen(suite/'initialization_audit.json',combined)

def freeze(project_root=ROOT,datasets=('bikenyc','chap'),epochs=100):
    roots={ds:prepare(ds,project_root) for ds in datasets};js=jobs(roots,project_root,epochs)
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts/v24').glob('*.py'))+list((ROOT/'scripts/v25').glob('*.py'))
    files+=list((ROOT/'configs/v24').rglob('*.json'))+list((ROOT/'configs/v25').glob('*.json'))
    files+=[ROOT/'AGENTS.md',ROOT/'scripts/v25/README_CROSS_DATASET_VALIDATION.md',ROOT/'tests/test_v25_cross_dataset.py',ROOT/'tests/test_v25_observed_multiscale.py',ROOT/'tests/test_v25_bikenyc_200.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))};data={}
    for data_root in roots.values():
        manifest=load(data_root/'manifest.json')
        for p in [data_root/'manifest.json',*(Path(r['path']) for r in manifest['outputs'].values())]:
            st=p.stat();data[str(p)]={'sha256':sha(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    refs={}
    for method in METHODS:
        n=f'taxibj_{method}_seed7';r=load(REFERENCE/'results'/f'{n}.json');run=Path(r['run_dir'])
        for label,path in [('receipt',REFERENCE/'results'/f'{n}.json'),('evaluation',REFERENCE/'evaluations'/f'{n}.json'),('config',run/'config.json')]:
            refs[f'{method}/{label}.json']={'path':str(path),'sha256':sha(path)}
    if datasets==('bikenyc',) and epochs==200:
        for method in METHODS:
            n=f'bikenyc_{method}_seed7';r=load(BIKE100/'results'/f'{n}.json');run=Path(r['run_dir'])
            for label,path in [('receipt',BIKE100/'results'/f'{n}.json'),('evaluation',BIKE100/'evaluations'/f'{n}.json'),('config',run/'config.json'),('history',run/'logs/metrics.jsonl')]:
                refs[f'bikenyc100/{method}/{label}.json']={'path':str(path),'sha256':sha(path)}
        refs['bikenyc100/initialization_audit.json']={'path':str(BIKE100/'initialization_audit.json'),'sha256':sha(BIKE100/'initialization_audit.json')}
    payload={'project_root':str(project_root),'jobs':js,'source':sources,'data':data,'references':refs,
        'reference_suite':str(REFERENCE),'order':list(js),'gpu':1,'datasets':list(datasets),'epochs':epochs,
        'origin':{'branch':subprocess.check_output(['git','branch','--show-current'],cwd=project_root,text=True).strip(),
                  'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip()}}
    suite=project_root/'outputs/v25/experiments/cross_dataset_validation'/digest(payload)[:16]
    suite.mkdir(parents=True,exist_ok=True);base.frozen(suite/'plan.json',payload)
    for n,stamp in sources.items():
        target=suite/'source_snapshot'/n
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
        if sha(target)!=stamp:raise RuntimeError('Source mismatch '+n)
    for n,row in refs.items():
        target=suite/'references'/n
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['path'],target)
        if sha(target)!=row['sha256']:raise RuntimeError('Reference mismatch '+n)
    return suite

def gpu1_only(gpus=(1,)):
    if list(gpus)!=[1]:raise ValueError('GPU1 only')

def worker(suite,payload):
    from report_cross_dataset_validation import export_report
    base.check_gpu_idle=gpu1_only
    original=base.subprocess_run
    def child(command,logpath,gpu,visible=False):
        command=[str(suite/'source_snapshot/scripts/v25/train_scale_delta.py') if str(v).endswith('/scripts/v24/train_four_direction.py') else v for v in command]
        return original(command,logpath,gpu,visible)
    base.subprocess_run=child
    js=copy.deepcopy(payload['jobs']);initialize(suite,js)
    audit_path=suite/'references/bikenyc100/initialization_audit.json'
    if audit_path.exists():
        old=load(audit_path);new=load(suite/'initialization_audit.json')
        for n in js:
            if new[n]['full_sha256']!=old[n]['full_sha256']:raise RuntimeError('100/200 epoch initialization mismatch')
    base.frozen(suite/'resolved_jobs.json',js)
    for n,j in js.items():
        base.frozen(suite/'jobs'/f'{n}.json',j);base.frozen(suite/'configs'/f'{n}.json',j['config'])
    preflight=suite/'acceptance/preflight.json'
    if not preflight.exists() or load(preflight).get('status')!='passed':
        write(suite/'state.json',{'phase':'preflight','gpu':1,'time':time.time()})
        base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/preflight_cross_dataset.py'),'--suite',str(suite)],suite/'launcher_logs/preflight.log',1,True)
    if not (suite/'started.json').exists():write(suite/'started.json',{'time':time.time(),'gpu':1})
    for n in payload['order']:
        j=js[n]
        try:
            for path,stamp in payload['data'].items():
                st=Path(path).stat()
                if (st.st_size,st.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Frozen data changed')
            write(suite/'state.json',{'phase':'training','variant':n,'gpu':1,'time':time.time()})
            base.launch(suite,j,1);r=base.receipt(suite,j)
            write(suite/'state.json',{'phase':'evaluating','variant':n,'gpu':1,'time':time.time()})
            base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],j['config']['experiment_plan']['protocol'],j['sources']['test'],'evaluations',1)
            export_report(Path(payload['project_root']),suite,js)
        except BaseException as error:
            write(suite/'failures'/f'{n}.json',{'error':repr(error),'time':time.time()})
            export_report(Path(payload['project_root']),suite,js);raise
    report=export_report(Path(payload['project_root']),suite,js)
    if not load(suite/'report_path.json')['complete']:raise RuntimeError('Final evidence audit failed')
    write(suite/'completed.json',{'time':time.time(),'count':len(js),'report':str(report)})
    write(suite/'state.json',{'phase':'completed','time':time.time(),'report':str(report)})
    print('Completed:',suite,'Report:',report,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--summary-only',action='store_true')
    p.add_argument('--datasets',nargs='+',choices=tuple(SPECS),default=list(SPECS));p.add_argument('--epochs',type=int,default=100)
    a=p.parse_args()
    if a.epochs<1:p.error('epochs must be positive')
    suite=a.suite.resolve() if a.suite else freeze(datasets=tuple(a.datasets),epochs=a.epochs);payload=load(suite/'plan.json');verify(suite,payload)
    for n,r in payload['references'].items():
        if sha(suite/'references'/n)!=r['sha256']:raise RuntimeError('Reference modified')
    if a.prepare_only:print(suite);return
    if a.summary_only:
        from report_cross_dataset_validation import export_report
        print(export_report(Path(payload['project_root']),suite,load(suite/'resolved_jobs.json')));return
    if a.worker:worker(suite,payload);return
    gpu_lock=Path(payload['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock';gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock,gpu_lock.open('a') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_cross_dataset_validation.py'),'--suite',str(suite),'--worker'],check=True)
        except BaseException as error:
            write(suite/'state.json',{'phase':'failed','time':time.time(),'error':repr(error)});raise

if __name__=='__main__':main()
