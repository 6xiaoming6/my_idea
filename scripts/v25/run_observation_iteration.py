#!/usr/bin/env python3
"""I42-I53: freeze, preflight, serial train, validation selection, test, report."""
import argparse,copy,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
from run_b3_c3 import load,write,digest
from prepare_data import sha
import run_four_direction_exploration as base
from observation_iteration_config import REFERENCE,CANDIDATES,METHODS,make_job,protocols,select_candidate

def gpu1_only(gpus):
    if list(gpus)!=[1]:raise ValueError('GPU1 only')
base.check_gpu_idle=gpu1_only

def freeze():
    jobs={j['variant']:j for j in [make_job(n) for n in CANDIDATES+['I52']]}
    templates={n:make_job('I53',selected=n) for n in CANDIDATES}
    refjob=load(REFERENCE/'jobs/taxibj_I37_seed7.json');receipt=load(REFERENCE/'results/taxibj_I37_seed7.json')
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts/v24').glob('*.py'))+list((ROOT/'scripts/v25').glob('*.py'))
    files+=list((ROOT/'tests').glob('test_v25*.py'))+list((ROOT/'configs/v24').rglob('*.json'))+list((ROOT/'configs/v25').rglob('*.json'))
    files += [ROOT/'AGENTS.md',ROOT/'scripts/v25/README_OBSERVATION_ITERATION.md']
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    old=load(REFERENCE/'plan.json');data=old['data']
    for path,stamp in data.items():
        if sha(path)!=stamp['sha256']:raise ValueError('Dataset changed '+path)
    reference=dict(checkpoint=str(Path(receipt['run_dir'])/'checkpoints/best.pth'),config_sha256=receipt['config_sha256'],
                   initialization=str(REFERENCE/'initialization/taxibj_I37_seed7.pth'),receipt=receipt,sources=refjob['sources'])
    for key in ('checkpoint','initialization'):reference[key+'_sha256']=sha(reference[key])
    test,dev,confirm=protocols()
    payload=dict(project_root=str(ROOT),jobs=jobs,candidate_templates=templates,order=list(jobs)+['taxibj_I53_seed17'],
        source=source,data=data,reference=reference,protocols=dict(test=test,development=dev,confirmation=confirm),
        selection=dict(candidates=CANDIDATES,id_ratio_max=1.03,ood_mae_ratio_max=.97,worst_category_ratio_max=1.05,
                       tie_break=['OOD_RMSE','ID_MAE','parameters','method'],selection_split='val',test_after_selection=True),
        origin=dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    suite=ROOT/'outputs/v25/experiments/observation_iteration'/digest(payload)[:16]
    suite.mkdir(parents=True,exist_ok=True);base.frozen(suite/'plan.json',payload)
    for relative,stamp in source.items():
        dest=suite/'source_snapshot'/relative
        if dest.exists():
            if sha(dest)!=stamp:raise ValueError('Frozen source changed')
        else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
    initialize(suite,payload)
    return suite

def initialize(suite,payload):
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    reference=torch.load(payload['reference']['initialization'],map_location='cpu',weights_only=False)
    refs={7:reference['model']}
    ref17=copy.deepcopy(payload['jobs']['taxibj_I52_seed17'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17);refs[17]=DualBranchSTImputer.from_config(ref17['config']).state_dict()
    audit={};resolved=copy.deepcopy(payload['jobs']);templates=copy.deepcopy(payload['candidate_templates'])
    all_jobs=list(resolved.items())+[(f'I53_from_{n}',j) for n,j in templates.items()]
    for key,j in all_jobs:
        seed=j['config']['seed']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model=DualBranchSTImputer.from_config(j['config'])
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        common=refs[seed]
        if any(k not in state or not torch.equal(v,state[k]) for k,v in common.items()):raise RuntimeError('Common initialization mismatch '+key)
        added=set(state)-set(common)
        if any(not k.startswith('main_branch.x_observation_') for k in added):raise RuntimeError('Unexpected new parameter')
        stamp=state_hash(state);path=suite/'initialization'/f'{key}.pth';path.parent.mkdir(exist_ok=True)
        if path.exists():
            if state_hash(torch.load(path,map_location='cpu',weights_only=False)['model'])!=stamp:raise RuntimeError('Initialization modified')
        else:torch.save(dict(model=state,sha256=stamp,seed=seed),path)
        j['common_initialization']=str(path)
        audit[key]=dict(common_sha256=state_hash(common),full_sha256=stamp,
                       parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    base.frozen(suite/'resolved_jobs.json',resolved);base.frozen(suite/'resolved_templates.json',templates)
    base.frozen(suite/'initialization_audit.json',audit)
    for key,j in all_jobs:base.frozen(suite/'configs'/f'{key}.json',j['config'])

def verify(suite,payload):
    for path,stamp in payload['source'].items():
        if sha(suite/'source_snapshot'/path)!=stamp:raise RuntimeError('Source changed '+path)
    for path,stamp in payload['data'].items():
        if sha(path)!=stamp['sha256']:raise RuntimeError('Data changed '+path)
    for key in ('checkpoint','initialization'):
        if sha(payload['reference'][key])!=payload['reference'][key+'_sha256']:raise RuntimeError('Reference changed')

def evaluate(suite,j,protocol,group):
    r=base.receipt(suite,j)
    return base.evaluate_record(suite,j['variant'],Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],
                                protocol,j['sources'][protocol['split']],group,1)

def reference_eval(suite,payload,protocol,group):
    r=payload['reference']
    return base.evaluate_record(suite,'taxibj_I37_seed7',r['checkpoint'],r['config_sha256'],protocol,r['sources'][protocol['split']],group,1)

def worker(suite,payload):
    jobs=load(suite/'resolved_jobs.json');templates=load(suite/'resolved_templates.json')
    pf=suite/'acceptance/preflight.json'
    if not pf.exists() or load(pf)['status']!='passed':
        raise RuntimeError('CPU and GPU1 preflight must pass before training')
    write(suite/'queue_state.json',dict(status='running',started=time.time()))
    try:
        for name,j in jobs.items():
            write(suite/'queue_state.json',dict(status='training',active=name,time=time.time()))
            base.launch(suite,j,1)
        # No test set has been evaluated by any new training above.
        dev=payload['protocols']['development'];reference=reference_eval(suite,payload,dev,'development')
        records={n:evaluate(suite,jobs[f'taxibj_{n}_seed7'],dev,'development') for n in CANDIDATES}
        audit=load(suite/'initialization_audit.json')
        selection=select_candidate(reference,records,{n:audit[f'taxibj_{n}_seed7']['parameters'] for n in CANDIDATES})
        selection['evidence_sha256']={n:sha(suite/'development'/f'taxibj_{n}_seed7.json') for n in CANDIDATES+['I37']}
        selection['definition_sha256']=digest(templates[selection['selected']])
        base.frozen(suite/'selection.json',selection)
        chosen=templates[selection['selected']];base.frozen(suite/'selected_job.json',chosen)
        base.launch(suite,chosen,1);jobs[chosen['variant']]=chosen
        # Test access is guarded by the immutable selection receipt above.
        for j in jobs.values():evaluate(suite,j,payload['protocols']['test'],'evaluations')
        reference_eval(suite,payload,payload['protocols']['test'],'evaluations')
        confirm=payload['protocols']['confirmation']
        reference_eval(suite,payload,confirm,'confirmation')
        for n in (f"taxibj_{selection['selected']}_seed7",'taxibj_I52_seed17','taxibj_I53_seed17'):
            evaluate(suite,jobs[n],confirm,'confirmation')
        from report_observation_iteration import export_report
        report=export_report(suite,payload,jobs,selection)
        write(suite/'queue_state.json',dict(status='finished',report=str(report),time=time.time()))
    except BaseException as error:
        write(suite/'queue_state.json',dict(status='failed',error=repr(error),time=time.time()));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--freeze',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.freeze:print(freeze(),flush=True);return
    if a.suite is None:raise ValueError('Explicit frozen suite required')
    suite=a.suite.resolve();payload=load(suite/'plan.json');verify(suite,payload)
    if a.worker:worker(suite,payload);return
    root=Path(payload['project_root']);locks=[]
    for path in (suite/'queue.lock',root/'outputs/v24-COE/experiments/gpu1.queue.lock'):
        path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_observation_iteration.py'),
                         '--suite',str(suite),'--worker'],suite/'launcher_logs/worker.log',1,visible=True)

if __name__=='__main__':main()
