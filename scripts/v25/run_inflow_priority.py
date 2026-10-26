#!/usr/bin/env python3
"""I28: fresh single-inflow I25, after I26 and before the held I27 waiter."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/v24'), str(ROOT/'scripts/v25')]
from run_after_suite import load, write, file_sha, digest, dependency_status, _completion_status, append_event
from run_interleaved import process, same_process, children
from run_before_priority_resume import check_ready

RUNNER = 'scripts/v25/run_inflow_priority.py'
NAME = 'taxibj_I28_seed7'
SUITES = ROOT/'outputs/v25/experiments/taxibj_input_communication'


def select_inflow(arrays):
    """Select the same physical variable in every scale, preserving all windows."""
    import numpy as np
    from stmoe_imputer.data.observed_multiscale import aggregate_sum
    if set(arrays) != {'x_f_gt', 'x_m_gt', 'x_c_gt'}:
        raise ValueError('Expected only the three preconstructed target arrays')
    result = {}
    fine = arrays['x_f_gt']
    if fine.ndim != 5 or fine.shape[1] != 2:
        raise ValueError('Expected the original two-variable BCTHW data')
    for key, factor in (('x_f_gt', 1), ('x_m_gt', 2), ('x_c_gt', 4)):
        value = arrays[key]
        expected = (*fine.shape[:3], fine.shape[-2]//factor, fine.shape[-1]//factor)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError('Invalid original scale '+key)
        result[key] = np.ascontiguousarray(value[:, 0:1], dtype=np.float32)
    for key, factor in (('x_m_gt', 2), ('x_c_gt', 4)):
        if not np.allclose(aggregate_sum(result['x_f_gt'], factor), result[key], rtol=1e-6, atol=1e-4):
            raise ValueError('Selected channel violates physical sum aggregation')
    return result


def make_job(original, sources):
    job = copy.deepcopy(original)
    job.pop('common_initialization', None)
    cfg = job['config']
    cfg['model']['c_in'] = 1
    cfg['data']['channel_selection'] = {'name': 'inflow', 'source_index': 0,
                                        'mode': 'preselected_npz_all_scales'}
    cfg['experiment_plan'].update(variant=NAME, reference='I25',
        method='I25单变量流入；F/M/C均仅保留原第0通道；独立seed7初始化从头训练',
        scope='Single-inflow task; same windows/mask seeds/I25 settings; no comparison to dual-channel aggregate MAE',
        initialization='Fresh seed7 c_in=1 model, no trained or two-channel initialization weights loaded')
    job.update(variant=NAME, method='I28', name='i28_v25_i25_inflow', reference='I25', sources=sources)
    return job


def freeze():
    import numpy as np
    import run_four_direction_exploration as base
    reference = SUITES/'2c07a1e3f7f177b7'
    original = load(reference/'jobs/taxibj_I25_seed7.json')
    reference_plan = load(reference/'plan.json')
    provenance = {'schema': 'v25_inflow_channel0_v1', 'channel': 'inflow', 'index': 0,
                  'channel_mapping_source': str(ROOT/'scripts/prepare_taxibj_npz.py'),
                  'channel_mapping_sha256': file_sha(ROOT/'scripts/prepare_taxibj_npz.py'),
                  'sources': {split: {'path': path, 'sha256': file_sha(path)}
                              for split, path in original['sources'].items()}}
    for row in provenance['sources'].values():
        if row['sha256'] != reference_plan['data'][row['path']]['sha256']:
            raise RuntimeError('I25 source data changed')
    directory = ROOT/'data/TaxiBJ/v25_inflow'/digest(provenance)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    sources, audit = {}, {}
    for split, source in provenance['sources'].items():
        with np.load(source['path'], allow_pickle=False) as z:
            selected = select_inflow({key: z[key] for key in z.files})
        target = directory/f'taxibj_{split}.npz'
        if target.exists():
            with np.load(target, allow_pickle=False) as z:
                if set(z.files) != set(selected) or any(not np.array_equal(z[k], v) for k,v in selected.items()):
                    raise RuntimeError('Existing inflow data changed')
        else:
            with target.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f, **selected)
            target.with_suffix('.tmp').replace(target)
        sources[split] = str(target)
        audit[split] = {'path': str(target), 'sha256': file_sha(target),
                        'shapes': {k: list(v.shape) for k,v in selected.items()},
                        'channel0_exact': True, 'aggregation_checked': True}
    base.frozen(directory/'manifest.json', {'provenance': provenance, 'outputs': audit})
    i26, i27 = SUITES/'8f67d8cc2cbf072e', SUITES/'6df045b55c2886df'
    hold = i27/'hold_for_i28_inflow.json'
    if not hold.exists():
        raise RuntimeError('I27 waiter must first be held at the I26 boundary')
    job = make_job(original, sources)
    files = list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts/v24').glob('*.py'))
    files += list((ROOT/'scripts/v25').glob('*.py'))
    files += [ROOT/'AGENTS.md', ROOT/'scripts/prepare_taxibj_npz.py',
              ROOT/'tests/test_v25_inflow.py', ROOT/'scripts/v25/README_INFLOW_PRIORITY.md']
    stamps = {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(set(files))}
    refs = {str(p): file_sha(p) for p in (reference/'plan.json', reference/'jobs/taxibj_I25_seed7.json',
            reference/'evaluations/taxibj_I25_seed7.json', directory/'manifest.json')}
    plan = {'project_root': str(ROOT), 'jobs': {NAME: job}, 'order': [NAME], 'gpu': 1,
            'source': stamps, 'references': refs, 'data': {},
            'run_after_suite': {'path': str(i26), 'plan_sha256': file_sha(i26/'plan.json')},
            'held_next': {'path': str(i27), 'plan_sha256': file_sha(i27/'plan.json'),
                          'hold': str(hold), 'hold_sha256': file_sha(hold)},
            'origin': {'branch': subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                       'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}}
    for path in sources.values():
        st=Path(path).stat()
        plan['data'][path]={'sha256':file_sha(path),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    suite = SUITES/digest(plan)[:16]
    suite.mkdir(parents=True, exist_ok=True)
    base.frozen(suite/'plan.json', plan)
    for name, expected in stamps.items():
        dest=suite/'source_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copyfile(ROOT/name,dest)
        if file_sha(dest)!=expected: raise RuntimeError('Snapshot mismatch '+name)
    return suite


def verify(suite, plan):
    if plan['gpu'] != 1: raise RuntimeError('Only GPU1 is authorized')
    for name, expected in plan['source'].items():
        if file_sha(suite/'source_snapshot'/name)!=expected: raise RuntimeError('Frozen source changed '+name)
    for path, expected in plan['references'].items():
        if file_sha(path)!=expected: raise RuntimeError('Reference changed '+path)
    for path, stamp in plan['data'].items():
        if file_sha(path)!=stamp['sha256']: raise RuntimeError('Inflow data changed '+path)


def held_next(plan):
    spec=plan['held_next']; suite=Path(spec['path'])
    if file_sha(suite/'plan.json')!=spec['plan_sha256'] or file_sha(spec['hold'])!=spec['hold_sha256']:
        raise RuntimeError('I27 hold evidence changed')
    identity=load(spec['hold'])
    if identity['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Host rebooted; requires explicit queue recovery')
    if not same_process(identity) or process(identity['pid'])['state'] not in ('T','t') or children(identity['pid']):
        raise RuntimeError('I27 waiter must remain held without training children')
    ready, detail, _ = check_ready(load(suite/'plan.json'))
    return identity, ready, detail


def prepare_initialization(suite, plan):
    import torch
    import run_four_direction_exploration as base
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    js=copy.deepcopy(plan['jobs'])
    job=js[NAME]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(job['config']['seed'])
        model=DualBranchSTImputer.from_config(job['config'])
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    stamp=state_hash(state); path=suite/'initialization'/f'{NAME}.pth'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False)
        if old['sha256']!=stamp or state_hash(old['model'])!=stamp: raise RuntimeError('Initialization changed')
    else:
        torch.save({'model':state,'sha256':stamp,'seed':job['config']['seed']},path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
    job['common_initialization']=str(path)
    base.frozen(suite/'initialization_audit.json',{'full_sha256':stamp,'seed':7,'c_in':1,
        'mode':'fresh single-channel random initialization; not equal to dual-channel initial tensors',
        'total_params':sum(p.numel() for p in model.parameters()),
        'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    base.frozen(suite/'resolved_jobs.json',js)
    base.frozen(suite/'jobs'/f'{NAME}.json',job)
    base.frozen(suite/'configs'/f'{NAME}.json',job['config'])
    return js


def execute(suite, plan):
    import run_four_direction_exploration as base
    from report_inflow_priority import export_report
    from run_taxibj_input_communication import gpu1_only
    identity, ready, detail = held_next(plan)
    if not ready: raise RuntimeError('I26 incomplete: '+detail)
    base.check_gpu_idle=gpu1_only
    with (suite.parent/'queue.lock').open('a') as lock, (Path(plan['project_root'])/'outputs/v24-COE/experiments/gpu1.queue.lock').open('a') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        js=prepare_initialization(suite,plan); job=js[NAME]
        try:
            acceptance=suite/'acceptance/preflight.json'
            if not acceptance.exists() or load(acceptance).get('status')!='passed':
                write(suite/'state.json',{'phase':'preflight','time':time.time(),'gpu':1})
                base.subprocess_run([sys.executable,'-u',str(suite/'source_snapshot/scripts/v25/preflight_taxibj_input_communication.py'),
                                     '--suite',str(suite)],suite/'launcher_logs/preflight.log',1,visible=True)
            write(suite/'state.json',{'phase':'training','variant':NAME,'time':time.time(),'gpu':1})
            base.launch(suite,job,1); receipt=base.receipt(suite,job)
            write(suite/'state.json',{'phase':'evaluating','variant':NAME,'time':time.time(),'gpu':1})
            base.evaluate_record(suite,NAME,Path(receipt['run_dir'])/'checkpoints/best.pth',receipt['config_sha256'],
                                 job['config']['experiment_plan']['protocol'],job['sources']['test'],'evaluations',1)
            report=export_report(Path(plan['project_root']),suite,js)
            if not load(suite/'report_path.json')['complete']:raise RuntimeError('I28 final report audit failed')
            write(suite/'completed.json',{'count':1,'time':time.time(),'report':str(report)})
            write(suite/'state.json',{'phase':'completed','time':time.time(),'report':str(report)})
            status,detail=_completion_status(suite,plan)
            if status!='ready':raise RuntimeError(detail)
        except BaseException as error:
            write(suite/'failures'/f'{NAME}.json',{'time':time.time(),'error':repr(error)})
            write(suite/'state.json',{'phase':'failed','time':time.time(),'error':repr(error)})
            export_report(Path(plan['project_root']),suite,js)
            raise


def run(suite, plan):
    def emit(phase, detail):
        event={'phase':phase,'detail':detail,'pid':os.getpid(),'time':time.time()}
        write(suite/'waiting.json',event);append_event(suite,event)
        print(json.dumps(event,ensure_ascii=False),flush=True)
    with (suite/'inflow_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (suite/'released_i27.json').exists():
                status,detail=_completion_status(suite,plan)
                if status!='ready':raise RuntimeError(detail)
                return
            last_log=0
            while True:
                identity, ready, detail=held_next(plan)
                status, audit=dependency_status(plan['run_after_suite'])
                if status=='blocked':raise RuntimeError(audit)
                if ready and status=='ready':break
                if time.monotonic()-last_log>=60:
                    emit('waiting_for_I26',detail);last_log=time.monotonic()
                time.sleep(5)
            emit('running_I28','I26 training, six evaluations and report complete; I27 remains held')
            subprocess.run([sys.executable,'-u',str(suite/'source_snapshot'/RUNNER),'--suite',str(suite),'--execute'],
                           cwd=plan['project_root'],check=True)
            status,detail=_completion_status(suite,plan)
            if status!='ready':raise RuntimeError('I28 audit failed: '+detail)
            identity,_,_=held_next(plan)
            # Training subprocess has exited and released both GPU1 queue locks.
            os.kill(identity['pid'],signal.SIGCONT)
            write(suite/'released_i27.json',{'identity':identity,'time':time.time(),'after':str(suite)})
            emit('completed','I28 complete; I27 waiter released; original remaining queue unchanged')
        except BaseException as error:
            emit('blocked',repr(error)+'; I27 remains held; no automatic retry')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();suite=a.suite.resolve() if a.suite else freeze()
    plan=load(suite/'plan.json');verify(suite,plan)
    if a.prepare_only:
        prepare_initialization(suite,plan)
        print(suite)
    elif a.execute:execute(suite,plan)
    else:run(suite,plan)
