#!/usr/bin/env python3
"""Validate the fixed CMFF backbone and bounded delta; no third-method search."""
import argparse,copy,fcntl,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
from run_backbone_exploration import jobs as w_jobs
from run_scale_rate_compare import filehash
from run_b3_c3 import load,write,digest
CONFIG=ROOT/'configs/v24/core_validation'
METHODS=tuple(f'K{i:02}' for i in range(1,9))


def jobs(datasets=('taxibj','bikenyc'),epochs=100,batch_size=32,seeds=(7,17,27),methods=METHODS):
    result={};templates=load(CONFIG/'variants.json')
    for dataset in datasets:
        origin=w_jobs(dataset,epochs,batch_size)['W01']
        for seed in seeds:
            for method in methods:
                if method not in METHODS[:4] and (dataset!='taxibj' or seed!=7):continue
                spec=templates[method];j=copy.deepcopy(origin);c=j['config'];coe=c['model']['coe']
                c['seed']=c['data']['loader_seed']=seed
                coe['expert_sharing']=spec['sharing']
                coe['id_priority'].update(scale_policy='fixed',fixed_epochs=0,
                    scale_soft_warmup_epochs=0,scale_soft_transition_epochs=0)
                coe['backbone_exploration']={'enabled':True}
                coe['core_validation']={'enabled':True,'path':spec['path'],'communication':spec['communication']}
                n=f'{dataset}_{method}_seed{seed}';ref=f'{dataset}_{spec["reference"]}_seed{seed}'
                c['experiment_plan']={'suite':'core_validation','variant':n,'method':spec['method'],'reference':ref,
                    'protocol':load(CONFIG/'test_protocol.json'),'selection':'ID validation MAE only',
                    'scope':'First two contributions only; no augmentation, consistency, or free scales'}
                j.update(variant=n,name=spec['name'],reference=ref,dataset=dataset,method=method)
                result[n]=j
    return result


def prepare_initialization(suite,all_jobs):
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    reference={};audit={}
    for n,j in all_jobs.items():
        seed=j['config']['seed']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model=DualBranchSTImputer.from_config(j['config'])
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        common={k:v for k,v in state.items() if 'step_pattern_experts.' not in k}
        if seed not in reference:reference[seed]=common
        if state_hash(reference[seed])!=state_hash(common):raise RuntimeError('Common constructor initialization differs: '+n)
        for key in state:
            src=key
            if 'step_pattern_experts.' in key:
                src='main_branch.pattern_experts.'+key.split('step_pattern_experts.',1)[1].split('.',1)[1]
            if not torch.equal(state[key],reference[seed][src]):raise RuntimeError('Independent pool did not copy template')
            state[key]=reference[seed][src].clone()
        model.load_state_dict(state,strict=True);sha=state_hash(state)
        path=suite/'initialization'/f'{n}.pth'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['sha256']!=sha or state_hash(saved['model'])!=sha:raise RuntimeError('Frozen initialization changed')
        else:
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
            torch.save({'model':state,'sha256':sha,'seed':seed},temp);temp.replace(path)
        j['common_initialization']=str(path)
        audit[n]={'seed':seed,'common_sha256':state_hash(common),'full_sha256':sha,
            'total_params':sum(p.numel() for p in model.parameters()),'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    base.frozen(suite/'initialization_audit.json',audit)


def report(suite,all_jobs):
    p=suite/'source_snapshot/scripts/v24/report_core_validation.py'
    spec=importlib.util.spec_from_file_location('frozen_core_report',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.export_report(ROOT,suite,all_jobs)


def run_queue(suite,payload,gpu):
    all_jobs=copy.deepcopy(payload['jobs']);prepare_initialization(suite,all_jobs)
    write(suite/'resolved_jobs.json',all_jobs)
    if not (suite/'started.json').exists():write(suite/'started.json',{'time':time.time(),'gpu':gpu})
    previous=None
    for n,j in all_jobs.items():
        block=(j['dataset'],j['config']['seed'])
        if previous is not None and block!=previous:report(suite,all_jobs)
        previous=block
        for path,stamp in payload['data'].items():
            st=Path(path).stat()
            if (st.st_size,st.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Dataset changed: '+path)
        base.frozen(suite/'configs'/f'{n}.json',j['config'])
        try:
            base.launch(suite,j,gpu);r=base.receipt(suite,j)
            base.evaluate_record(suite,n,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],
                j['config']['experiment_plan']['protocol'],j['sources']['test'],'evaluations',gpu)
        except BaseException as error:
            write(suite/'failures'/f'{n}.json',{'error':repr(error),'time':time.time()})
            report(suite,all_jobs);raise
    write(suite/'completed.json',{'time':time.time(),'count':len(all_jobs)})
    print(f'Results: {suite}\nReport: {report(suite,all_jobs)}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',nargs='+',choices=('taxibj','bikenyc'),default=['taxibj','bikenyc'])
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27]);p.add_argument('--variants',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--suite',type=Path);p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if a.gpu!=0 or min(a.epochs,a.batch_size)<1 or len(set(a.dataset))!=len(a.dataset) or len(set(a.seeds))!=len(a.seeds) or len(set(a.variants))!=len(a.variants):p.error('GPU0 only; positive budget and unique selections required')
    if a.suite:
        suite=a.suite.resolve();payload=load(suite/'plan.json')
    else:
        all_jobs=jobs(a.dataset,a.epochs,a.batch_size,a.seeds,a.variants)
        if not all_jobs:p.error('Selected combinations contain no jobs')
        source=base.manifest(a.dataset[0])
        for name in ('tests/test_v24_core_validation.py','scripts/v24/README_CORE_VALIDATION.md'):
            source[name]=filehash(ROOT/name)
        data={}
        for j in all_jobs.values():
            for path in j['sources'].values():
                if path not in data:
                    st=Path(path).stat();data[path]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':filehash(path)}
        payload={'jobs':all_jobs,'source':source,'data':data,'order':list(all_jobs)}
        suite=ROOT/'outputs/v24-COE/experiments/core_validation'/digest(payload)[:16]
    if a.dry_run:print(json.dumps({'suite':str(suite),**payload},ensure_ascii=False,indent=2));return
    if a.summary_only:print(report(suite,payload['jobs']));return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);suite.mkdir(exist_ok=True)
        for name,sha in payload['source'].items():
            target=suite/'source_snapshot'/name
            if not target.exists():
                if a.suite:raise RuntimeError('Missing frozen source '+name)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
            if filehash(target)!=sha:raise RuntimeError('Frozen source modified '+name)
        for path,stamp in payload['data'].items():
            if filehash(path)!=stamp['sha256']:raise RuntimeError('Dataset fingerprint changed')
        base.frozen(suite/'plan.json',payload)
        command=[sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/run_core_validation.py'),'--suite',str(suite),'--gpu',str(a.gpu)]
        code=subprocess.call(command,env=dict(os.environ,CORE_VALIDATION_FROZEN_PARENT='1'))
        if code:raise SystemExit(code)


if __name__=='__main__':
    if os.environ.get('CORE_VALIDATION_FROZEN_PARENT')=='1':
        p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
        a=p.parse_args();suite=a.suite.resolve();payload=load(suite/'plan.json')
        ROOT=Path(next(iter(payload['jobs'].values()))['config']['output_dir']).parents[1]
        run_queue(suite,payload,a.gpu)
    else:main()
