#!/usr/bin/env python3
"""Freeze and run I54-I63 on GPU1; evaluate development OOD after each run."""
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
from input_refinement_config import REFERENCE, PREVIOUS, CANDIDATES, make_job, protocols, select_candidate


def gpu1_only(gpus):
    if list(gpus) != [1]:
        raise ValueError('This suite uses physical GPU1 only')
base.check_gpu_idle = gpu1_only


def reference_record(suite, payload, seed):
    if seed == 27:
        job = load(suite/'resolved_jobs.json')['taxibj_I61_seed27']
        receipt = base.receipt(suite, job)
        return dict(checkpoint=str(Path(receipt['run_dir'])/'checkpoints/best.pth'),
                    config_sha256=receipt['config_sha256'], sources=job['sources'], variant=job['variant'])
    return payload['references'][str(seed)]


def freeze():
    jobs = {j['variant']: j for j in [make_job(n) for n in CANDIDATES + ['I61']]}
    templates = {f'{rep}_from_{n}': make_job(rep, selected=n) for rep in ['I62', 'I63'] for n in CANDIDATES}
    old = load(REFERENCE/'plan.json');data = old['data']
    for file, stamp in data.items():
        if sha(file) != stamp['sha256']: raise ValueError('Dataset changed '+file)
    references = {}
    for seed, parent, name in [(7, REFERENCE, 'taxibj_I37_seed7'), (17, PREVIOUS, 'taxibj_I52_seed17')]:
        job = load(parent/'jobs'/f'{name}.json');r = load(parent/'results'/f'{name}.json')
        checkpoint = str(Path(r['run_dir'])/'checkpoints/best.pth')
        references[str(seed)] = dict(variant=name, checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
            config_sha256=r['config_sha256'], initialization=job['common_initialization'],
            initialization_sha256=sha(job['common_initialization']), sources=job['sources'], receipt=r)
    files = list((ROOT/'src').rglob('*.py'))
    files += list((ROOT/'scripts/v24').glob('*.py')) + list((ROOT/'scripts/v25').glob('*.py'))
    files += list((ROOT/'tests').glob('test_v25*.py')) + list((ROOT/'configs/v24').rglob('*.json')) + list((ROOT/'configs/v25').rglob('*.json'))
    files += [ROOT/'AGENTS.md', ROOT/'scripts/v25/README_INPUT_REFINEMENT.md']
    source = {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    test, development, confirmation = protocols()
    payload = dict(project_root=str(ROOT), jobs=jobs, candidate_templates=templates,
        order=[*jobs, 'taxibj_I62_seed17', 'taxibj_I63_seed27'], source=source, data=data, references=references,
        protocols=dict(test=test, development=development, confirmation=confirmation),
        selection=dict(candidates=CANDIDATES, id_ratio_max=1.03, ood_mae_ratio_max=.97, worst_category_ratio_max=1.05,
                       split='val', test_after_selection=True, development_after_each_train=True),
        origin=dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    suite = ROOT/'outputs/v25/experiments/input_refinement'/digest(payload)[:16]
    suite.mkdir(parents=True, exist_ok=True);base.frozen(suite/'plan.json', payload)
    for relative, stamp in source.items():
        target = suite/'source_snapshot'/relative
        if target.exists():
            if sha(target) != stamp: raise ValueError('Frozen source changed')
        else:
            target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(ROOT/relative, target)
    initialize(suite, payload)
    return suite


def initialize(suite, payload):
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    references = {}
    for seed in [7, 17, 27]:
        cfg = copy.deepcopy(payload['jobs']['taxibj_I61_seed27']['config'])
        cfg['seed'] = seed;cfg['data']['loader_seed'] = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);reference = DualBranchSTImputer.from_config(cfg)
        state = {n:v.detach().cpu().clone() for n,v in reference.state_dict().items()}
        if seed != 27:
            old = torch.load(payload['references'][str(seed)]['initialization'],map_location='cpu',weights_only=False)
            if state_hash(state) != old['sha256']:
                raise RuntimeError('Original seed initialization changed '+str(seed))
        references[seed] = state
    jobs = copy.deepcopy(payload['jobs']);templates = copy.deepcopy(payload['candidate_templates']);audit = {}
    for key, j in [*jobs.items(), *templates.items()]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(j['config']['seed']);model = DualBranchSTImputer.from_config(j['config'])
        state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()};stamp = state_hash(state)
        if stamp != state_hash(references[j['config']['seed']]):
            raise RuntimeError('Input intervention changed common parameters '+key)
        path = suite/'initialization'/f'{key}.pth';path.parent.mkdir(exist_ok=True)
        if path.exists():
            if state_hash(torch.load(path,map_location='cpu',weights_only=False)['model']) != stamp:
                raise ValueError('Initialization changed')
        else:
            torch.save(dict(model=state, sha256=stamp, seed=j['config']['seed']), path)
        j['common_initialization'] = str(path)
        base.frozen(suite/'configs'/f'{key}.json', j['config'])
        audit[key] = dict(sha256=stamp, parameters=sum(p.numel() for p in model.parameters()),
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    base.frozen(suite/'resolved_jobs.json', jobs);base.frozen(suite/'resolved_templates.json', templates)
    base.frozen(suite/'initialization_audit.json', audit)


def verify(suite, payload):
    for name, stamp in payload['source'].items():
        if sha(suite/'source_snapshot'/name) != stamp: raise ValueError('Frozen source changed '+name)
    for name, stamp in payload['data'].items():
        if sha(name) != stamp['sha256']: raise ValueError('Dataset changed '+name)
    for ref in payload['references'].values():
        for key in ['checkpoint', 'initialization']:
            if sha(ref[key]) != ref[key+'_sha256']: raise ValueError('Reference changed '+key)


def evaluate_job(suite, job, protocol, group):
    r = base.receipt(suite, job)
    return base.evaluate_record(suite, job['variant'], Path(r['run_dir'])/'checkpoints/best.pth',
                               r['config_sha256'], protocol, job['sources'][protocol['split']], group, 1)


def evaluate_reference(suite, payload, seed, protocol, group):
    ref = reference_record(suite, payload, seed)
    return base.evaluate_record(suite, ref['variant'], Path(ref['checkpoint']), ref['config_sha256'],
                               protocol, ref['sources'][protocol['split']], group, 1)


def worker(suite, payload):
    acceptance = suite/'acceptance/preflight.json'
    if not acceptance.exists() or load(acceptance)['status'] != 'passed':
        raise RuntimeError('CPU and GPU preflight required before training')
    jobs = load(suite/'resolved_jobs.json');templates = load(suite/'resolved_templates.json')
    try:
        dev = payload['protocols']['development']
        write(suite/'queue_state.json', dict(status='reference_development',time=time.time()))
        reference = evaluate_reference(suite, payload, 7, dev, 'development')
        records = {}
        for name, job in jobs.items():
            write(suite/'queue_state.json', dict(status='training',active=name,time=time.time()))
            base.launch(suite, job, 1)
            # Show ID + OOD on the validation split as each model completes;
            # keep test untouched until the method selection is immutable.
            write(suite/'queue_state.json', dict(status='development',active=name,time=time.time()))
            record = evaluate_job(suite, job, dev, 'development')
            if job['method'] in CANDIDATES: records[job['method']] = record
        parameters = {n:load(suite/'initialization_audit.json')[f'taxibj_{n}_seed7']['parameters'] for n in CANDIDATES}
        selection = select_candidate(reference, records, parameters)
        selection['evidence_sha256'] = {n:sha(suite/'development'/f'taxibj_{n}_seed7.json') for n in CANDIDATES+['I37']}
        selection['template_sha256'] = {rep:digest(templates[f"{rep}_from_{selection['selected']}"]) for rep in ['I62','I63']}
        base.frozen(suite/'selection.json', selection)
        for rep in ['I62', 'I63']:
            job = templates[f"{rep}_from_{selection['selected']}"]
            base.frozen(suite/f'selected_{rep}.json', job)
            write(suite/'queue_state.json', dict(status='training',active=job['variant'],time=time.time()))
            base.launch(suite, job, 1);jobs[job['variant']] = job
            evaluate_job(suite, job, dev, 'development')
        evaluate_reference(suite, payload, 17, dev, 'development')
        for group in ['evaluations', 'confirmation']:
            protocol = payload['protocols']['test' if group=='evaluations' else 'confirmation']
            for seed in [7,17]: evaluate_reference(suite, payload, seed, protocol, group)
            for job in jobs.values():
                write(suite/'queue_state.json', dict(status=group,active=job['variant'],time=time.time()))
                evaluate_job(suite, job, protocol, group)
        from report_input_refinement import export_report
        path = export_report(suite, payload, jobs, selection)
        write(suite/'queue_state.json', dict(status='finished',report=str(path),time=time.time()))
    except BaseException as error:
        write(suite/'queue_state.json', dict(status='failed',error=repr(error),time=time.time()))
        raise


def main():
    p = argparse.ArgumentParser();p.add_argument('--freeze', action='store_true');p.add_argument('--suite', type=Path)
    p.add_argument('--worker', action='store_true');a=p.parse_args()
    if a.freeze: print(freeze(),flush=True);return
    if a.suite is None: raise ValueError('An explicit frozen suite is required')
    suite=a.suite.resolve();payload=load(suite/'plan.json');verify(suite,payload)
    if a.worker:worker(suite,payload);return
    locks=[]
    for path in [suite/'queue.lock',Path(payload['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock']:
        path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/run_input_refinement.py'),
                         '--suite',str(suite),'--worker'],suite/'launcher_logs/worker.log',1,visible=True)

if __name__=='__main__':main()
