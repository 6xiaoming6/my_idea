#!/usr/bin/env python3
"""Frozen nine-job v25 queue, physical GPU1, epoch-atomic continuation."""
from __future__ import annotations
import argparse, copy, fcntl, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
import run_four_direction_exploration as base
from run_core_validation import prepare_initialization
from run_first_validation import protocol,verify
from prepare_data import prepare,sha

REFERENCE=ROOT/'outputs/v25/experiments/first_validation/659d3ba9678e1dca'
METHODS={
 'S01':('free',[2,2,2,2],[],False,'每轮自由硬选 C/M/F，共享 Top-2，无差分','B1'),
 'S02':('CMFF',[4,4,4,4],[],False,'CMFF，共享 Top-4，无差分','B1'),
 'S03':('CMFF',[1,2,4,8],[],False,'CMFF，共享 Top-1/2/4/8，无差分','B1 / S02'),
 'S04':('CMFF',[2,2,2,2],[2,3,4],False,'原 B2 完整差分，从头独立复验','B2 / B1'),
 'S05':('CMFF',[2,2,2,2],[2],False,'仅第二轮差分 H1−H0（融合→C）','B1 / S04'),
 'S06':('CMFF',[2,2,2,2],[3],False,'仅第三轮差分 H2−H1（C→M）','B1 / S04'),
 'S07':('CMFF',[2,2,2,2],[4],False,'仅第四轮差分 H3−H2（M→F）','B1 / S04'),
 'S08':('CFFF',[2,2,2,2],[],False,'CFFF，共享 Top-2，无差分','B1 / S09'),
 'S09':('CFFF',[2,2,2,2],[4],True,'CFFF，仅第四轮 F 输出之间的差分','S08 / S07-B1'),
}
ORDER=['S04','S08','S09','S05','S06','S07','S01','S02','S03']


def jobs(data_root,project_root=ROOT):
    template=load(ROOT/'configs/v25/base.json');result={}
    for method,(path,ks,steps,same,description,reference) in METHODS.items():
        c=copy.deepcopy(template);c['output_dir']=str(project_root/'outputs/v25')
        c['model']['coe']['core_validation']['communication']='conditional' if steps else 'none'
        c['model']['coe']['scale_delta_exploration']={
            'enabled':True,'path':path,'top_k_per_step':ks,'delta_steps':steps,'same_scale_only':same,
            'scale_gradient':'selected_branch_st','top1_gradient':'selected_probability_st'}
        n=f'taxibj_{method}_seed7'
        c['experiment_plan']={'suite':'v25_scale_delta_exploration','variant':n,'method':description,
            'protocol':protocol(),'selection':'ID validation MAE only','scope':'nine independent from-scratch runs; seed7 screening'}
        result[n]={'variant':n,'name':method.lower()+'_v25_scale_delta','method':method,'dataset':'taxibj',
            'reference':reference,'config':c,'sources':{s:str(Path(data_root)/f'taxibj_{s}.npz') for s in ('train','val','test')}}
    return result


def freeze(project_root=ROOT):
    data_root=prepare(project_root/'data/TaxiBJ/v24_clean_abc_20260917',project_root/'data/TaxiBJ/v25_observed_multiscale')
    js=jobs(data_root,project_root)
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts/v24').glob('*.py'))+list((ROOT/'scripts/v25').glob('*.py'))
    files+=list((ROOT/'configs/v24').rglob('*.json'))+list((ROOT/'configs/v25').glob('*.json'))
    files+=[ROOT/'AGENTS.md',ROOT/'scripts/v25/README_SCALE_DELTA_EXPLORATION.md',ROOT/'tests/test_v25_scale_delta_exploration.py',ROOT/'tests/test_v25_observed_multiscale.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    data={}
    for p in [*(data_root/f'taxibj_{s}.npz' for s in ('train','val','test')),data_root/'manifest.json']:
        st=p.stat();data[str(p)]={'sha256':sha(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    ref_files={}
    for method in ('B1','B2'):
        n=f'taxibj_{method}_seed7';r=load(REFERENCE/'results'/f'{n}.json');run=Path(r['run_dir'])
        for label,path in [('receipt',REFERENCE/'results'/f'{n}.json'),('evaluation',REFERENCE/'evaluations'/f'{n}.json'),
                           ('config',run/'config.json'),('metadata',run/'training_metadata.json'),('history',run/'logs/metrics.jsonl')]:
            ref_files[f'{method}/{label}.json']={'path':str(path),'sha256':sha(path)}
    origin={'branch':subprocess.check_output(['git','branch','--show-current'],cwd=project_root,text=True).strip(),
            'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip()}
    payload={'project_root':str(project_root),'jobs':js,'source':sources,'data':data,'origin':origin,
             'order':[f'taxibj_{n}_seed7' for n in ORDER],'gpu':1,'reference_suite':str(REFERENCE),'references':ref_files}
    suite=project_root/'outputs/v25/experiments/scale_delta_exploration'/digest(payload)[:16]
    suite.mkdir(parents=True,exist_ok=True);base.frozen(suite/'plan.json',payload)
    for name,stamp in sources.items():
        target=suite/'source_snapshot'/name
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        if sha(target)!=stamp:raise RuntimeError('Frozen source mismatch '+name)
    for name,row in ref_files.items():
        target=suite/'references'/name
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['path'],target)
        if sha(target)!=row['sha256']:raise RuntimeError('Reference evidence changed '+name)
    return suite


def gpu1_only(gpus=(1,)):
    if list(gpus)!=[1]:raise ValueError('This queue is physical GPU1 only')
    # User authorized GPU1 execution without an exclusive-idle admission wait.
    # Shared GPU1 queue lock still prevents our own serial queues overlapping.


def worker(suite,payload):
    from report_scale_delta_exploration import export_report
    base.check_gpu_idle=gpu1_only
    original_subprocess=base.subprocess_run
    def run_child(command,logpath,gpu,visible=False):
        command=list(command)
        for i,value in enumerate(command):
            if str(value).endswith('/scripts/v24/train_four_direction.py'):
                command[i]=str(suite/'source_snapshot/scripts/v25/train_scale_delta.py')
        return original_subprocess(command,logpath,gpu,visible)
    base.subprocess_run=run_child
    js=copy.deepcopy(payload['jobs']);prepare_initialization(suite,js)
    # Existing B1/B2 constructors must match the complete frozen initialization.
    expected=load(Path(payload['reference_suite'])/'initialization_audit.json')['taxibj_B1_seed7']['full_sha256']
    audit=load(suite/'initialization_audit.json')
    if any(row['full_sha256']!=expected for row in audit.values()):raise RuntimeError('B1 initialization mismatch')
    base.frozen(suite/'resolved_jobs.json',js)
    for n,j in js.items():base.frozen(suite/'jobs'/f'{n}.json',j);base.frozen(suite/'configs'/f'{n}.json',j['config'])
    preflight=suite/'acceptance/preflight.json'
    if not preflight.exists() or load(preflight).get('status')!='passed':
        write(suite/'state.json',{'phase':'preflight','gpu':1,'time':time.time()})
        base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/preflight_scale_delta.py'),'--suite',str(suite)],suite/'launcher_logs/preflight.log',1,visible=True)
    if not (suite/'started.json').exists():write(suite/'started.json',{'time':time.time(),'gpu':1,'order':payload['order']})
    for n in payload['order']:
        j=js[n]
        try:
            for path,stamp in payload['data'].items():
                st=Path(path).stat()
                if (st.st_size,st.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Frozen data changed '+path)
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
    write(suite/'completed.json',{'time':time.time(),'count':9,'gpu':1,'report':str(report)})
    write(suite/'state.json',{'phase':'completed','time':time.time(),'report':str(report)})
    print(f'Completed nine runs: {suite}\nReport: {report}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if a.gpu!=1:p.error('GPU1 only')
    suite=a.suite.resolve() if a.suite else freeze();payload=load(suite/'plan.json');verify(suite,payload)
    for name,row in payload['references'].items():
        if sha(suite/'references'/name)!=row['sha256']:raise RuntimeError('Frozen reference changed '+name)
    if a.prepare_only:print(suite);return
    if a.summary_only:
        from report_scale_delta_exploration import export_report
        print(export_report(Path(payload['project_root']),suite,load(suite/'resolved_jobs.json')));return
    if a.worker:worker(suite,payload);return
    gpu_lock=Path(payload['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock'
    gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock,gpu_lock.open('a') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_scale_delta_exploration.py'),'--suite',str(suite),'--worker','--gpu','1'],check=True)
        except BaseException as error:
            write(suite/'state.json',{'phase':'failed','time':time.time(),'error':repr(error)});raise


if __name__=='__main__':main()
