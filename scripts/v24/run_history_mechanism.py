#!/usr/bin/env python3
"""Eight frozen Z jobs, single GPU0, skip completed, epoch-state resume."""
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

PARENT='outputs/v24-COE/experiments/history_followup/47fea7c076e9a355'


def jobs():
    template=load(ROOT/PARENT/'plan.json')['jobs']['taxibj_Y12_seed7']
    result={}
    for label,spec in load(ROOT/'configs/v24/history_mechanism/variants.json').items():
        job=copy.deepcopy(template);c=job['config'];name=f'taxibj_{label}_seed7'
        c['model']['coe']['history_mechanism']=dict(enabled=True,**{k:spec[k] for k in ('interaction','history','representation')})
        c['data']['mask']['missing_rate']=spec['rate']
        for key in ('train_mask_diversity','eval_mask_diversity'):c['data'][key]['rates']=[spec['rate']]
        protocol=c['experiment_plan']['protocol'];protocol['rate']=spec['rate']
        for item in protocol['evaluations'].values():
            if 'rate' in item:item['rate']=spec['rate']
        c['experiment_plan'].update(suite='history_mechanism',variant=name,method=spec['method'],
            reference=' / '.join(spec['control']),factors=spec,scope='Z01-Z08 fixed mechanism exploration')
        job.update(variant=name,method=label,name=f'{label.lower()}_history_mechanism',reference=' / '.join(spec['control']))
        result[name]=job
    return result


def make_plan():
    parent=ROOT/PARENT;old=load(parent/'plan.json');refs={};locations={}
    for label in ['K04','X01','X04','X05','X09','X10','Y05','Y06','Y07','Y08','Y12']:
        origin=Path(old['core']) if label=='K04' else Path(old['upstream']) if label.startswith('X') else parent
        variant=f'taxibj_{label}_seed7'
        refs[label]=load(origin/'plan.json')['jobs'][variant]
        locations[label]=str(origin)
    source=base.manifest('taxibj')
    for rel in ('AGENTS.md','scripts/v24/README_HISTORY_MECHANISM.md','tests/test_v24_history_mechanism.py'):
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
            'training_metadata.json':run/'training_metadata.json','runtime.json':run/'runtime.json'}
        for filename,src in files.items():
            rel=f'references/{label}/{filename}';dst=suite/rel;sha=filehash(src)
            if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            if filehash(dst)!=sha:raise RuntimeError('Reference changed '+rel)
            manifest[rel]=sha
    base.frozen(suite/'reference_manifest.json',manifest)


def report(suite,payload):
    from report_history_mechanism import export_report
    return export_report(Path(payload['root']),suite,payload)


def run_queue(suite,payload):
    validate(suite,payload)
    js=copy.deepcopy(payload['jobs']);prepare_initialization(suite,js,suite/'reference_initialization.pth')
    base.frozen(suite/'resolved_jobs.json',js)
    for name,job in js.items():base.frozen(suite/'configs'/f'{name}.json',job['config'])
    ap=suite/'acceptance/preflight.json'
    if not ap.exists() or load(ap).get('status')!='passed':
        write(suite/'state.json',dict(phase='GPU0_batch32_preflight',time=time.time()))
        base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/preflight_history_mechanism.py'),
            '--suite',str(suite)],suite/'launcher_logs/preflight.log',0,visible=True)
    audit=load(ap)
    if audit['status']!='passed' or audit['source_sha256']!=digest(payload['source']) or audit['configs_sha256']!=digest({n:j['config'] for n,j in js.items()}) or len(audit['cases'])!=16:
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
    print('Finished Z01-Z08. Report:',dest,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args();payload=load(a.suite/'plan.json') if a.suite else make_plan()
    suite=a.suite.resolve() if a.suite else ROOT/'outputs/v24-COE/experiments/history_mechanism'/digest(payload)[:16]
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
        subprocess.run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v24/run_history_mechanism.py'),'--suite',str(suite),'--worker'],check=True)


if __name__=='__main__':main()
