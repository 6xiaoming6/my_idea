#!/usr/bin/env python3
"""Four from-scratch v25 input x communication cells, GPU0 serial only."""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
import run_four_direction_exploration as base
from run_core_validation import prepare_initialization
from prepare_data import prepare,sha

METHODS={'A1':('F','none'),'A2':('F','conditional'),'B1':('FMC','none'),'B2':('FMC','conditional')}


def check_gpu0(gpus=(0,)):
    if list(gpus)!=[0]:raise ValueError('GPU0 only')
    uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(row.split(',')[0].strip()==uuid for row in rows.splitlines()):
        raise RuntimeError('GPU0 already has a compute process; refusing to overlap')


def protocol():
    # Reuse the exact v24 ID/OOD family lists and mask seeds. This file is frozen
    # alongside the runner, so subsequent edits cannot change an active suite.
    p = load(ROOT/'configs/v24/core_validation/test_protocol.json')
    p['scale_protocol'] = 'same family and missing rate at all scales; independent RNG; A masks M/C to zero'
    return p


def jobs(data_root, project_root=ROOT):
    template=load(ROOT/'configs/v25/base.json');result={}
    for method,(scales,communication) in METHODS.items():
        c=copy.deepcopy(template);c['output_dir']=str(project_root/'outputs/v25')
        c['data']['input_scales']=scales
        c['model']['coe']['core_validation']['communication']=communication
        n=f'taxibj_{method}_seed7'
        c['experiment_plan']={'suite':'v25_first_validation_mixed_mask','variant':n,'method':f'{scales} input, CMFF, delta={communication}',
                             'protocol':protocol(),'selection':'ID validation MAE only','scope':'input x delta, no history/interaction/consistency'}
        result[n]={'variant':n,'name':f'{method.lower()}_v25_observed_multiscale_mixed','method':method,'dataset':'taxibj',
                   'reference':{'A1':'none','A2':'A1','B1':'A1','B2':'B1 / A2'}[method],'config':c,
                   'sources':{s:str(Path(data_root)/f'taxibj_{s}.npz') for s in ('train','val','test')}}
    return result


def freeze(project_root=ROOT):
    data_root=prepare(project_root/'data/TaxiBJ/v24_clean_abc_20260917',project_root/'data/TaxiBJ/v25_observed_multiscale')
    js=jobs(data_root,project_root)
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts/v24').glob('*.py'))+list((ROOT/'scripts/v25').glob('*.py'))
    files+=list((ROOT/'configs/v24').rglob('*.json'))+list((ROOT/'configs/v25').glob('*.json'))
    files+=[ROOT/'AGENTS.md',ROOT/'scripts/v25/README_FIRST_VALIDATION.md',ROOT/'tests/test_v25_observed_multiscale.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    data={}
    for p in [*(data_root/f'taxibj_{s}.npz' for s in ('train','val','test')),data_root/'manifest.json']:
        st=p.stat();data[str(p)]={'sha256':sha(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    origin={'branch':subprocess.check_output(['git','branch','--show-current'],cwd=project_root,text=True).strip(),
            'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),
            'v24_base':subprocess.check_output(['git','rev-parse','v24-COE'],cwd=project_root,text=True).strip()}
    payload={'project_root':str(project_root),'jobs':js,'source':sources,'data':data,'origin':origin,'order':list(js)}
    suite=project_root/'outputs/v25/experiments/first_validation'/digest(payload)[:16]
    suite.mkdir(parents=True,exist_ok=True)
    base.frozen(suite/'plan.json',payload)
    for name,stamp in sources.items():
        target=suite/'source_snapshot'/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        if sha(target)!=stamp:raise RuntimeError('Frozen source mismatch '+name)
    return suite


def verify(suite,payload):
    for name,stamp in payload['source'].items():
        if sha(suite/'source_snapshot'/name)!=stamp:raise RuntimeError('Frozen source modified '+name)
    for name,stamp in payload['data'].items():
        if sha(name)!=stamp['sha256']:raise RuntimeError('Data modified '+name)


def worker(suite,payload):
    from report_first_validation import export_report
    base.check_gpu_idle=check_gpu0
    js=copy.deepcopy(payload['jobs']);prepare_initialization(suite,js)
    base.frozen(suite/'resolved_jobs.json',js)
    for n,j in js.items():
        base.frozen(suite/'jobs'/f'{n}.json',j)
        base.frozen(suite/'configs'/f'{n}.json',j['config'])
    if not (suite/'acceptance/preflight.json').exists() or load(suite/'acceptance/preflight.json').get('status')!='passed':
        write(suite/'state.json',{'phase':'preflight','time':time.time()})
        base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/preflight.py'),
                             '--suite',str(suite)],suite/'launcher_logs/preflight.log',0,visible=True)
    if not (suite/'started.json').exists():write(suite/'started.json',{'time':time.time(),'gpu':0})
    for n,j in js.items():
        try:
            for path,stamp in payload['data'].items():
                stat=Path(path).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):
                    raise RuntimeError('Frozen dataset changed: '+path)
            write(suite/'state.json',{'phase':'training','variant':n,'time':time.time()})
            base.launch(suite,j,0);r=base.receipt(suite,j)
            write(suite/'state.json',{'phase':'evaluating','variant':n,'time':time.time()})
            base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],j['config']['experiment_plan']['protocol'],j['sources']['test'],'evaluations',0)
            export_report(Path(payload['project_root']),suite,js)
        except BaseException as error:
            write(suite/'failures'/f'{n}.json',{'error':repr(error),'time':time.time()})
            export_report(Path(payload['project_root']),suite,js)
            raise
    path=export_report(Path(payload['project_root']),suite,js)
    if not load(suite/'report_path.json')['complete']:
        raise RuntimeError('Final evidence audit did not pass; see staged report')
    write(suite/'completed.json',{'time':time.time(),'count':4,'report':str(path)})
    write(suite/'state.json',{'phase':'completed','time':time.time(),'report':str(path)})
    print(f'Completed: {suite}\nReport: {path}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args();suite=a.suite.resolve() if a.suite else freeze()
    payload=load(suite/'plan.json');verify(suite,payload)
    if a.prepare_only:print(suite);return
    if a.worker:
        worker(suite,payload);return
    # Share the existing v24 GPU0 queue lock across versions as well as a suite lock.
    gpu_lock=Path(payload['project_root'])/'outputs/v24-COE/experiments/gpu0.queue.lock'
    with (suite.parent/'queue.lock').open('a') as lock,gpu_lock.open('a') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_first_validation.py'),
                            '--suite',str(suite),'--worker'],check=True)
        except BaseException as error:
            write(suite/'state.json',{'phase':'failed','time':time.time(),'error':repr(error)})
            raise


if __name__=='__main__':main()
