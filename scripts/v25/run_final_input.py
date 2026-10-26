#!/usr/bin/env python3
"""Freeze/run I64-I75 on physical GPU1; old results remain read-only."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/v24'), str(ROOT/'scripts/v25')]
from run_b3_c3 import load, write, digest
from prepare_data import sha
import run_four_direction_exploration as base
from final_input_config import DESIGN, REFERENCE, PREVIOUS, CANDIDATES, make_job, protocols, choose


def gpu1_only(gpus):
    if list(gpus) != [1]: raise ValueError('Physical GPU1 only')
base.check_gpu_idle = gpu1_only


def freeze():
    jobs = {j['variant']: j for j in map(make_job, CANDIDATES)}
    data = load(REFERENCE/'plan.json')['data']
    for file, stamp in data.items():
        if sha(file) != stamp['sha256']: raise ValueError('Dataset changed '+file)
    references = {}
    for method, parent in [('I37', REFERENCE), ('I56', PREVIOUS), ('I57', PREVIOUS)]:
        name = f'taxibj_{method}_seed7'
        job = load(parent/'jobs'/f'{name}.json'); r = load(parent/'results'/f'{name}.json')
        if r['status'] != 'finished' or r['completed_epochs'] != 100:
            raise ValueError('Reference incomplete '+name)
        checkpoint = str(Path(r['run_dir'])/'checkpoints/best.pth')
        references[method] = dict(variant=name, checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
            config_sha256=r['config_sha256'], initialization=job['common_initialization'],
            initialization_sha256=sha(job['common_initialization']), sources=job['sources'], receipt=r)
    files = list((ROOT/'src').rglob('*.py'))
    files += list((ROOT/'scripts/v24').glob('*.py')) + list((ROOT/'scripts/v25').glob('*.py'))
    files += list((ROOT/'tests').glob('test_v25*.py'))
    files += list((ROOT/'configs/v24').rglob('*.json')) + list((ROOT/'configs/v25').rglob('*.json'))
    files += [ROOT/'AGENTS.md', ROOT/'scripts/v25/README_FINAL_INPUT.md', DESIGN, DESIGN.with_suffix('.md')]
    source = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}
    test, dev, confirm = protocols()
    payload = dict(project_root=str(ROOT), jobs=jobs, order=list(jobs), source=source, data=data,
        references=references, design=load(DESIGN),
        protocols=dict(test=test, development=dev, confirmation=confirm),
        selection=dict(candidates=CANDIDATES, id_ratio_max=1.03, ood_mae_ratio_max=.97,
                       worst_category_ratio_max=1.05, split='val', test_after_selection=True,
                       no_extra_training=True, development_after_each_train=True),
        origin=dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    suite = ROOT/'outputs/v25/experiments/final_input'/digest(payload)[:16]
    suite.mkdir(parents=True, exist_ok=True); base.frozen(suite/'plan.json', payload)
    for relative, stamp in source.items():
        target = suite/'source_snapshot'/relative
        if target.exists():
            if sha(target) != stamp: raise ValueError('Frozen source changed '+relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/relative, target)
    initialize(suite, payload)
    return suite


def initialize(suite, payload):
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    expected = torch.load(payload['references']['I37']['initialization'], map_location='cpu', weights_only=False)['sha256']
    jobs = copy.deepcopy(payload['jobs']); audit = {}
    for name, j in jobs.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7); model = DualBranchSTImputer.from_config(j['config'])
        state = {n: v.detach().cpu().clone() for n,v in model.state_dict().items()}
        stamp = state_hash(state)
        if stamp != expected: raise RuntimeError('Common initialization changed '+name)
        path = suite/'initialization'/f'{name}.pth'; path.parent.mkdir(exist_ok=True)
        if path.exists():
            if state_hash(torch.load(path,map_location='cpu',weights_only=False)['model']) != stamp:
                raise ValueError('Initialization changed '+name)
        else: torch.save(dict(model=state, sha256=stamp, seed=7), path)
        j['common_initialization'] = str(path)
        base.frozen(suite/'configs'/f'{name}.json', j['config'])
        audit[name] = dict(sha256=stamp, parameters=sum(p.numel() for p in model.parameters()),
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    base.frozen(suite/'resolved_jobs.json', jobs); base.frozen(suite/'initialization_audit.json', audit)


def verify(suite, payload):
    for name, stamp in payload['source'].items():
        if sha(suite/'source_snapshot'/name) != stamp: raise ValueError('Frozen source changed '+name)
    for name, stamp in payload['data'].items():
        if sha(name) != stamp['sha256']: raise ValueError('Dataset changed '+name)
    for ref in payload['references'].values():
        for key in ['checkpoint', 'initialization']:
            if sha(ref[key]) != ref[key+'_sha256']: raise ValueError('Reference changed '+key)


def evaluate_reference(suite, payload, method, protocol, group):
    r = payload['references'][method]
    return base.evaluate_record(suite,r['variant'],Path(r['checkpoint']),r['config_sha256'],
                                protocol,r['sources'][protocol['split']],group,1)


def evaluate_job(suite, job, protocol, group):
    r = base.receipt(suite, job)
    return base.evaluate_record(suite,job['variant'],Path(r['run_dir'])/'checkpoints/best.pth',
        r['config_sha256'],protocol,job['sources'][protocol['split']],group,1)


def worker(suite, payload):
    acceptance = suite/'acceptance/preflight.json'
    cpu = suite/'acceptance/cpu.json'
    if not acceptance.exists() or load(acceptance)['status'] != 'passed' or not cpu.exists() or load(cpu)['status'] != 'passed':
        raise RuntimeError('CPU and GPU preflight must pass before training')
    jobs = load(suite/'resolved_jobs.json')
    try:
        dev = payload['protocols']['development']; records = {}
        write(suite/'queue_state.json',dict(status='reference_development',time=time.time()))
        reference = evaluate_reference(suite,payload,'I37',dev,'development')
        for name, job in jobs.items():
            write(suite/'queue_state.json',dict(status='training',active=name,time=time.time()))
            base.launch(suite,job,1)
            write(suite/'queue_state.json',dict(status='development',active=name,time=time.time()))
            records[job['method']] = evaluate_job(suite,job,dev,'development')
        for method in ['I56','I57']: evaluate_reference(suite,payload,method,dev,'development')
        parameters = {j['method']: load(suite/'initialization_audit.json')[n]['parameters'] for n,j in jobs.items()}
        selection = choose(reference, records, parameters)
        selection['evidence_sha256'] = {n: digest(r) for n,r in records.items()}
        selection['reference_sha256'] = digest(reference)
        base.frozen(suite/'selection.json',selection)
        for group in ['evaluations','confirmation']:
            protocol = payload['protocols']['test' if group == 'evaluations' else 'confirmation']
            for method in payload['references']: evaluate_reference(suite,payload,method,protocol,group)
            for name, job in jobs.items():
                write(suite/'queue_state.json',dict(status=group,active=name,time=time.time()))
                evaluate_job(suite,job,protocol,group)
        from report_final_input import export_report
        path = export_report(suite,payload,jobs,selection)
        write(suite/'queue_state.json',dict(status='finished',report=str(path),time=time.time()))
    except BaseException as error:
        write(suite/'queue_state.json',dict(status='failed',error=repr(error),time=time.time()))
        raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); p.add_argument('--suite',type=Path)
    p.add_argument('--worker',action='store_true'); a=p.parse_args()
    if a.freeze: print(freeze(),flush=True); return
    if a.suite is None: raise ValueError('An explicit frozen suite is required')
    suite=a.suite.resolve(); payload=load(suite/'plan.json'); verify(suite,payload)
    if a.worker: worker(suite,payload); return
    locks=[]
    for path in [suite/'queue.lock',Path(payload['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock']:
        path.parent.mkdir(parents=True,exist_ok=True); f=path.open('a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(f)
    base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_final_input.py'),
                         '--suite',str(suite),'--worker'],suite/'launcher_logs/worker.log',1,visible=True)

if __name__ == '__main__': main()
