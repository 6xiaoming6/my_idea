#!/usr/bin/env python3
"""Selectable TaxiBJ I01-I08 interventions; frozen previous suites stay immutable."""
import argparse,copy,fcntl,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
from prepare_data import sha
from run_first_validation import verify
from run_history_plane_interaction import prepare_initialization
import run_four_direction_exploration as base
REFERENCE=ROOT/'outputs/v25/experiments/first_validation/659d3ba9678e1dca'
METHODS={'I01':'H0仅F；CMFF各轮注入对应尺度观测；完整受控差分',
         'I02':'H0三尺度；四轮自由硬选择尺度、共享Top2；完整受控差分',
         'I03':'H0三尺度；CMFF共享Top2；完整受控差分＋自适应预测历史池',
         'I04':'H0三尺度；CMFF共享Top2；完整受控差分＋最终F聚合对完整M/C真值的L1损失',
         'I05':'H0三尺度；CMFF共享Top2；完整受控差分＋可见观测剩余区域自适应非负一致性校正',
         'I06':'H0三尺度；CMFF共享Top2；完整受控差分＋仅可见M/C观测的一致性软损失',
         'I07':'H0三尺度；CMFF共享Top2；完整受控差分＋固定等权剩余区域非负投影',
         'I08':'H0三尺度；CMFF共享Top2；完整受控差分＋自适应预测历史池＋自适应剩余区域投影'}


def jobs(reference=REFERENCE, methods=None):
    selected=list(METHODS) if methods is None else list(methods)
    if not selected or len(set(selected))!=len(selected) or any(m not in METHODS for m in selected):
        raise ValueError('Select distinct methods from I01-I08')
    origin=load(reference/'jobs/taxibj_B2_seed7.json');result={}
    for method in selected:
        description=METHODS[method]
        j=copy.deepcopy(origin);j.pop('common_initialization',None);c=j['config'];name=f'taxibj_{method}_seed7'
        c['model']['coe']['initial_observation_scales']='F' if method=='I01' else 'FMC'
        if method=='I02':c['model']['coe']['scale_delta_exploration']={'enabled':True,'path':'free','top_k_per_step':[2]*4,'delta_steps':[2,3,4],'same_scale_only':False,'scale_gradient':'selected_branch_st','top1_gradient':'selected_probability_st'}
        if method=='I03':c['model']['coe']['observed_history']={'enabled':True,'mode':'adaptive'}
        if method=='I04':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'full_target_loss'}
            c['loss']['lambda_observation_consistency']=0.1
            c['data']['return_multiscale_targets']=True
        if method=='I05':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'residual_projection','weight_bound':2.0}
        if method=='I06':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'observed_loss'}
            c['loss']['lambda_observation_consistency']=0.1
        if method=='I07':
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'residual_projection','weight_bound':2.0,'weight_mode':'uniform'}
        if method=='I08':
            c['model']['coe']['observed_history']={'enabled':True,'mode':'adaptive'}
            c['model']['coe']['observation_consistency']={'enabled':True,'mode':'residual_projection','weight_bound':2.0}
        c['experiment_plan'].update(suite='v25_taxibj_input_communication',variant=name,method=description,reference='B2',scope='three requested interventions; 100 epochs original B2 protocol')
        if method in ('I04','I05'):
            c['experiment_plan']['scope']='consistency followup; full-target soft loss versus observed-only adaptive residual projection; no history; 100 epochs original B2 protocol'
        if method in ('I06','I07','I08'):
            c['experiment_plan']['scope']='eight-run mechanism controls: observed-only soft loss, fixed projection, history plus learned projection; original B2 protocol'
        j.update(variant=name,method=method,name=method.lower()+'_v25_taxibj_input_communication',reference='B2')
        result[name]=j
    return result


def freeze(methods=None):
    js=jobs(methods=methods);old=load(REFERENCE/'plan.json')
    # Check historical data before inheriting a reference comparison.
    for path,stamp in old['data'].items():
        if sha(path)!=stamp['sha256']:raise RuntimeError('B2 dataset changed '+path)
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts/v24').glob('*.py'))+list((ROOT/'scripts/v25').glob('*.py'))
    files+=list((ROOT/'configs/v24').rglob('*.json'))+list((ROOT/'configs/v25').glob('*.json'))
    files += [ROOT/'AGENTS.md',ROOT/'scripts/v25/README_TAXIBJ_INPUT_COMMUNICATION.md',ROOT/'tests/test_v25_taxibj_input_communication.py',ROOT/'tests/test_v25_observed_multiscale.py',ROOT/'tests/test_v25_observation_consistency.py',ROOT/'tests/test_v25_consistency_controls.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))};refs={}
    for method in ['A2','B1','B2']:
        n=f'taxibj_{method}_seed7';r=load(REFERENCE/'results'/f'{n}.json');run=Path(r['run_dir'])
        for label,p in [('receipt',REFERENCE/'results'/f'{n}.json'),('evaluation',REFERENCE/'evaluations'/f'{n}.json'),('job',REFERENCE/'jobs'/f'{n}.json'),('config',run/'config.json'),('metadata',run/'training_metadata.json'),('history',run/'logs/metrics.jsonl')]:refs[f'{method}/{label}.json']={'path':str(p),'sha256':sha(p)}
    init=REFERENCE/'initialization/taxibj_B2_seed7.pth';refs['initialization.pth']={'path':str(init),'sha256':sha(init)}
    payload={'project_root':str(ROOT),'jobs':js,'order':list(js),'gpu':1,'source':sources,'data':old['data'],'references':refs,'reference_suite':str(REFERENCE),'origin':{'branch':subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()}}
    if any(j['method'] in ('I06','I07','I08') for j in js.values()):
        # These already frozen suites are read-only comparison dependencies.
        # Never import their jobs into this training order or retrain them here.
        previous=ROOT/'outputs/v25/experiments/taxibj_input_communication'
        payload['comparison_suites']={str(previous/name):sha(previous/name/'plan.json')
                                      for name in ('ea1ca776ade869d6','b5925acd4615e143')}
    suite=ROOT/'outputs/v25/experiments/taxibj_input_communication'/digest(payload)[:16];suite.mkdir(parents=True,exist_ok=True);base.frozen(suite/'plan.json',payload)
    for name,stamp in sources.items():
        target=suite/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(ROOT/name,target)
        if sha(target)!=stamp:raise RuntimeError('Frozen source mismatch '+name)
    for name,row in refs.items():
        target=suite/'references'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(row['path'],target)
        if sha(target)!=row['sha256']:raise RuntimeError('Frozen reference mismatch '+name)
    return suite


def gpu1_only(gpus=(1,)):
    if list(gpus)!=[1]:raise ValueError('Only physical GPU1 is authorized')


def worker(suite,payload):
    from report_taxibj_input_communication import export_report
    base.check_gpu_idle=gpu1_only
    js=copy.deepcopy(payload['jobs']);prepare_initialization(suite,js,suite/'references/initialization.pth')
    base.frozen(suite/'resolved_jobs.json',js)
    for n,j in js.items():base.frozen(suite/'jobs'/f'{n}.json',j);base.frozen(suite/'configs'/f'{n}.json',j['config'])
    if not (suite/'started.json').exists():write(suite/'started.json',{'gpu':1,'time':time.time(),'order':list(js)})
    acceptance=suite/'acceptance/preflight.json'
    if not acceptance.exists() or load(acceptance).get('status')!='passed':
        write(suite/'state.json',{'phase':'preflight','gpu':1,'time':time.time()})
        base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/preflight_taxibj_input_communication.py'),'--suite',str(suite)],suite/'launcher_logs/preflight.log',1,visible=True)
    for n,j in js.items():
        try:
            for path,stamp in payload['data'].items():
                st=Path(path).stat()
                if (st.st_size,st.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Data changed '+path)
            write(suite/'state.json',{'phase':'training','variant':n,'gpu':1,'time':time.time()})
            base.launch(suite,j,1);r=base.receipt(suite,j)
            write(suite/'state.json',{'phase':'evaluating','variant':n,'gpu':1,'time':time.time()})
            base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],j['config']['experiment_plan']['protocol'],j['sources']['test'],'evaluations',1)
            export_report(Path(payload['project_root']),suite,js)
        except BaseException as error:
            write(suite/'failures'/f'{n}.json',{'time':time.time(),'error':repr(error)})
            export_report(Path(payload['project_root']),suite,js);raise
    report=export_report(Path(payload['project_root']),suite,js)
    if not load(suite/'report_path.json')['complete']:raise RuntimeError('Final report audit failed')
    write(suite/'completed.json',{'count':len(js),'time':time.time(),'report':str(report)})
    write(suite/'state.json',{'phase':'completed','time':time.time(),'report':str(report)})
    print('Completed '+str(suite),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--methods',nargs='+',choices=list(METHODS),help='New suite only; use I06 I07 I08 after the existing frozen I01-I03 and I04-I05 suites');p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--summary-only',action='store_true');a=p.parse_args()
    if a.suite and a.methods:p.error('--methods cannot modify an existing --suite')
    suite=a.suite.resolve() if a.suite else freeze(a.methods);payload=load(suite/'plan.json');verify(suite,payload)
    for name,row in payload['references'].items():
        if sha(suite/'references'/name)!=row['sha256']:raise RuntimeError('Reference changed '+name)
    if a.prepare_only:print(suite);return
    if a.summary_only:
        from report_taxibj_input_communication import export_report
        print(export_report(Path(payload['project_root']),suite,load(suite/'resolved_jobs.json')));return
    if a.worker:worker(suite,payload);return
    gpu_lock=Path(payload['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock';gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock,gpu_lock.open('a') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,PYTHONFAULTHANDLER='1')
        try:subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_taxibj_input_communication.py'),'--suite',str(suite),'--worker'],env=env,check=True)
        except BaseException as error:
            write(suite/'state.json',{'phase':'failed','time':time.time(),'error':repr(error)});raise

if __name__=='__main__':main()
