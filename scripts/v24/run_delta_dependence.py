#!/usr/bin/env python3
"""Four frozen delta-dependence jobs, single GPU0, skip completed, epoch-state resume."""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
import run_history_followup as previous
from run_history_plane_interaction import prepare_initialization
from run_b3_c3 import load,write,digest
from run_scale_rate_compare import filehash
from run_team_accept_v4 import check_gpu_idle

PARENT='outputs/v24-COE/experiments/history_mechanism/a964cb1715943e5a'


def reference_jobs():
    parent=ROOT/PARENT;old=load(parent/'plan.json')
    refs={k:old['references'][k] for k in ['K04','X01','X05','X09','X10','Y12']}
    locations={k:old['reference_suites'][k] for k in refs}
    upstream=Path(old['upstream'])
    refs['X06']=load(upstream/'plan.json')['jobs']['taxibj_X06_seed7'];locations['X06']=str(upstream)
    refs['Z06']=old['jobs']['taxibj_Z06_seed7'];locations['Z06']=str(parent)
    return refs,locations


def jobs():
    refs,_=reference_jobs();result={}
    for label,spec in load(ROOT/'configs/v24/delta_dependence/variants.json').items():
        job=copy.deepcopy(refs[spec['parent']]);c=job['config'];name=f'taxibj_{label}_seed7'
        c['model']['coe']['core_validation']['communication']='none'
        c['model']['coe']['delta_dependence']=dict(enabled=True,parent=spec['parent'])
        c['experiment_plan'].update(suite='delta_dependence',variant=name,method=spec['method'],
            reference=' / '.join(spec['control']),factors=spec,scope='Z09-Z12 from-initialization no-delta controls')
        job.update(variant=name,method=label,name=f'{label.lower()}_delta_dependence',reference=' / '.join(spec['control']))
        result[name]=job
    return result


def make_plan():
    parent=ROOT/PARENT;old=load(parent/'plan.json');refs,locations=reference_jobs()
    source=base.manifest('taxibj')
    for rel in ('AGENTS.md','scripts/v24/README_DELTA_DEPENDENCE.md','tests/test_v24_delta_dependence.py',
                'tests/test_gpu_monitor.py'):
        source[rel]=filehash(ROOT/rel)
    return dict(jobs=jobs(),references=refs,reference_suites=locations,source=source,data=old['data'],root=str(ROOT),
        upstream=old['upstream'],core=old['core'],upstream_plan_sha256=old['upstream_plan_sha256'],core_plan_sha256=old['core_plan_sha256'],
        parent=str(parent),parent_plan_sha256=filehash(parent/'plan.json'),initialization=old['initialization'])


def validate(suite,payload):
    previous.validate(suite,payload)
    if filehash(Path(payload['parent'])/'plan.json')!=payload['parent_plan_sha256']:
        raise RuntimeError('Parent plan changed')
    for rel,sha in load(suite/'reference_manifest.json').items():
        if filehash(suite/rel)!=sha:raise RuntimeError('Frozen reference changed: '+rel)


def freeze_references(suite,payload):
    manifest={}
    for label,job in payload['references'].items():
        origin=Path(payload['reference_suites'][label]);name=job['variant'];r=base.receipt(origin,job)
        if not r:raise RuntimeError('Unfinished reference '+label)
        e=load(origin/'evaluations'/f'{name}.json');protocol=job['config']['experiment_plan']['protocol']
        if e.get('status')!='finished' or e['config_sha256']!=digest(job['config']) or e['protocol_sha256']!=digest(protocol) or set(e['sets'])!=set(protocol['evaluations']):
            raise RuntimeError('Invalid reference evaluation '+label)
        run=Path(r['run_dir'])
        files={'result.json':origin/'results'/f'{name}.json','evaluations.json':origin/'evaluations'/f'{name}.json',
            'config.json':run/'config.json','metrics.jsonl':run/'logs/metrics.jsonl',
            'training_metadata.json':run/'training_metadata.json','runtime.json':run/'runtime.json',
            'initialization.pth':origin/'initialization'/f'{name}.pth'}
        for filename,src in files.items():
            rel=f'references/{label}/{filename}';dst=suite/rel;sha=filehash(src)
            if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            if filehash(dst)!=sha:raise RuntimeError('Reference changed '+rel)
            manifest[rel]=sha
    base.frozen(suite/'reference_manifest.json',manifest)


def audit_parent_initialization(suite,js,payload):
    import torch
    from stmoe_imputer.utils.deterministic import state_hash
    audit={}
    for name,job in js.items():
        parent=job['config']['model']['coe']['delta_dependence']['parent']
        old=torch.load(suite/'references'/parent/'initialization.pth',map_location='cpu',weights_only=False)
        new=torch.load(job['common_initialization'],map_location='cpu',weights_only=False)
        if state_hash(old['model'])!=state_hash(new['model']):raise RuntimeError('Parent initialization mismatch '+name)
        expected=copy.deepcopy(payload['references'][parent]['config']);actual=copy.deepcopy(job['config'])
        expected['model']['coe']['core_validation']['communication']='none'
        actual['model']['coe'].pop('delta_dependence')
        expected.pop('experiment_plan');actual.pop('experiment_plan')
        if expected!=actual:raise RuntimeError('Unexpected parent config change '+name)
        audit[name]=dict(parent=parent,parent_full_sha256=state_hash(old['model']),new_full_sha256=state_hash(new['model']),
            method_changes=['core_validation.communication: conditional -> none','delta_dependence: explicit constructor'],
            metadata_changes=['experiment_plan'],trained_checkpoint_loaded=False)
    base.frozen(suite/'parent_initialization_audit.json',audit)


def report(suite,payload):
    from report_delta_dependence import export_report
    return export_report(Path(payload['root']),suite,payload)


def run_queue(suite,payload):
    validate(suite,payload)
    js=copy.deepcopy(payload['jobs']);prepare_initialization(suite,js,suite/'reference_initialization.pth')
    audit_parent_initialization(suite,js,payload)
    base.frozen(suite/'resolved_jobs.json',js)
    for name,job in js.items():base.frozen(suite/'configs'/f'{name}.json',job['config'])
    ap=suite/'acceptance/preflight.json'
    if not ap.exists() or load(ap).get('status')!='passed':
        write(suite/'state.json',dict(phase='GPU0_batch32_preflight',time=time.time()))
        base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/preflight_delta_dependence.py'),
            '--suite',str(suite)],suite/'launcher_logs/preflight.log',0,visible=True)
    audit=load(ap)
    if audit['status']!='passed' or audit['source_sha256']!=digest(payload['source']) or audit['configs_sha256']!=digest({n:j['config'] for n,j in js.items()}) or len(audit['cases'])!=8:
        raise RuntimeError('Invalid preflight evidence')
    if not (suite/'started.json').exists():write(suite/'started.json',dict(time=time.time(),gpu=0))
    report(suite,payload)
    for name,job in js.items():
        write(suite/'state.json',dict(phase='training_and_evaluating',variant=name,time=time.time()))
        for path,stamp in payload['data'].items():
            st=Path(path).stat()
            if (st.st_size,st.st_mtime_ns)!=(stamp['size'],stamp['mtime_ns']):raise RuntimeError('Data changed')
        previous.reserve_run(suite,job)
        try:
            if base.receipt(suite,job):print('Skip completed training:',name,flush=True)
            base.launch(suite,job,0);r=base.receipt(suite,job)
            base.evaluate_record(suite,name,Path(r['run_dir'])/'checkpoints/best.pth',r['config_sha256'],
                job['config']['experiment_plan']['protocol'],job['sources']['test'],'evaluations',0)
        except BaseException as error:
            write(suite/'failures'/f'{name}.json',dict(error=repr(error),time=time.time()))
            report(suite,payload);raise
        report(suite,payload)
    dest=report(suite,payload)
    if any(r['status']!='finished' for r in load(suite/'summary.json').values()):raise RuntimeError('Incomplete final audit')
    write(suite/'completed.json',dict(time=time.time(),count=len(js),report=str(dest)))
    write(suite/'state.json',dict(phase='completed',time=time.time()))
    print('Finished Z09-Z12. Report:',dest,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args();payload=load(a.suite/'plan.json') if a.suite else make_plan()
    suite=a.suite.resolve() if a.suite else ROOT/'outputs/v24-COE/experiments/delta_dependence'/digest(payload)[:16]
    if a.prepare_only:
        previous.freeze(suite,payload);freeze_references(suite,payload);validate(suite,payload);print(suite);return
    if a.worker:
        try:run_queue(suite,payload)
        except BaseException as error:
            write(suite/'queue_failure.json',dict(time=time.time(),error=repr(error)))
            write(suite/'state.json',dict(phase='failed',time=time.time(),error=repr(error)));raise
        return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock, (suite.parent.parent/'gpu0.queue.lock').open('a') as gpu_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_gpu_idle([0])
        if not a.suite:previous.freeze(suite,payload);freeze_references(suite,payload)
        validate(suite,payload)
        subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/run_delta_dependence.py'),'--suite',str(suite),'--worker'],check=True)


if __name__=='__main__':main()
