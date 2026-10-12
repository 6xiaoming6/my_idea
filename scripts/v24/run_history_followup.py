#!/usr/bin/env python3
"""Resume the frozen X queue, then run 12 frozen Y jobs, serially on GPU0."""
import argparse
import copy
from datetime import datetime
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).resolve().parent)]
import run_four_direction_exploration as base
from run_b3_c3 import load, write, digest
from run_scale_rate_compare import filehash
from run_history_plane_interaction import prepare_initialization

UPSTREAM = 'outputs/v24-COE/experiments/history_plane_interaction/86284122bd9a18c2'
CORE = 'outputs/v24-COE/experiments/core_validation/684cad05994af9a4'


def jobs():
    template = load(ROOT/CORE/'plan.json')['jobs']['taxibj_K04_seed7']
    result = {}
    for method, spec in load(ROOT/'configs/v24/history_followup/variants.json').items():
        job = copy.deepcopy(template)
        c = job['config']; coe = c['model']['coe']; rate = spec['rate']
        name = f'taxibj_{method}_seed7'
        coe['core_validation']['communication'] = 'conditional' if spec['delta'] else 'none'
        coe['history_plane_interaction'] = dict(enabled=True, history='adaptive' if spec['history'] else 'none',
                                               residual=False, plane='none', interaction='none')
        if spec['global_plane'] != 'none' or spec['history_pair']:
            coe['history_followup'] = dict(enabled=True, global_plane=spec['global_plane'], history_pair=spec['history_pair'])
        c['loss']['lambda_coe_mid'] = spec['mid']
        c['data']['mask']['missing_rate'] = rate
        for key in ('train_mask_diversity', 'eval_mask_diversity'):
            if key in c['data']:
                c['data'][key]['rates'] = [rate]
        protocol = c['experiment_plan']['protocol']; protocol['rate'] = rate
        for item in protocol['evaluations'].values():
            if 'rate' in item:
                item['rate'] = rate
        c['experiment_plan'].update(suite='history_followup', variant=name, method=spec['method'],
            reference=spec['control'], scope='8 evidence-based controls + 4 exploratory combinations', factors=spec)
        job.update(variant=name, method=method, name=f'{method.lower()}_history_followup', reference=spec['control'])
        result[name] = job
    return result


def make_plan():
    js = jobs(); upstream = ROOT/UPSTREAM; core = ROOT/CORE
    references = {'K04':load(core/'plan.json')['jobs']['taxibj_K04_seed7']}
    references.update({f'X{i:02}':load(upstream/'plan.json')['jobs'][f'taxibj_X{i:02}_seed7'] for i in range(1, 11)})
    source = base.manifest('taxibj')
    for rel in ('AGENTS.md', 'scripts/v24/README_HISTORY_FOLLOWUP.md',
                'scripts/v24/README_HISTORY_PLANE_INTERACTION.md', 'tests/test_v24_history_followup.py'):
        source[rel] = filehash(ROOT/rel)
    data = {}
    for path in next(iter(js.values()))['sources'].values():
        p = Path(path); st = p.stat()
        data[path] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=filehash(p))
    initial = core/'initialization/taxibj_K04_seed7.pth'
    return dict(jobs=js, order=list(js), references=references, source=source, data=data, root=str(ROOT),
                upstream=str(upstream), upstream_plan_sha256=filehash(upstream/'plan.json'),
                core=str(core), core_plan_sha256=filehash(core/'plan.json'),
                initialization=dict(path=str(initial), sha256=filehash(initial)))


def freeze(suite, payload):
    suite.mkdir(parents=True, exist_ok=True)
    for rel, sha in payload['source'].items():
        dst = suite/'source_snapshot'/rel
        if not dst.exists():
            if filehash(ROOT/rel) != sha:
                raise RuntimeError('Source changed before freeze: '+rel)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/rel, dst)
        if filehash(dst) != sha:
            raise RuntimeError('Snapshot mismatch: '+rel)
    p = suite/'reference_initialization.pth'
    if not p.exists():
        shutil.copyfile(payload['initialization']['path'], p)
    base.frozen(suite/'plan.json', payload)
    validate(suite, payload)


def validate(suite, payload):
    for rel, sha in payload['source'].items():
        if filehash(suite/'source_snapshot'/rel) != sha:
            raise RuntimeError('Frozen source changed: '+rel)
    for path, stamp in payload['data'].items():
        if filehash(path) != stamp['sha256']:
            raise RuntimeError('Dataset changed: '+path)
    if filehash(suite/'reference_initialization.pth') != payload['initialization']['sha256']:
        raise RuntimeError('Reference initialization changed')
    for directory, key in [('upstream', 'upstream_plan_sha256'), ('core', 'core_plan_sha256')]:
        if filehash(Path(payload[directory])/'plan.json') != payload[key]:
            raise RuntimeError('Reference plan changed: '+directory)
    upstream = Path(payload['upstream'])
    for rel, sha in load(upstream/'plan.json')['source'].items():
        if filehash(upstream/'source_snapshot'/rel) != sha:
            raise RuntimeError('Upstream frozen source changed: '+rel)


def wait_for_upstream(suite, upstream):
    """The old queue owns this lock through training, evaluation and report export."""
    lockpath = upstream.parent/'queue.lock'
    with lockpath.open('a') as handle:
        waiting = False
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if not waiting:
                    print('Existing X queue owns its lock; waiting without launching duplicate training.', flush=True)
                    waiting = True
                write(suite/'state.json', dict(phase='waiting_for_existing_X_queue', time=time.time(), upstream=str(upstream)))
                time.sleep(10)


def finish_upstream(suite, payload):
    upstream = Path(payload['upstream'])
    wait_for_upstream(suite, upstream)
    # Old snapshot and old config remain authoritative. Its receipt checks skip completed work.
    if not (upstream/'completed.json').exists():
        write(suite/'state.json', dict(phase='resuming_unfinished_X_jobs', time=time.time()))
        script = upstream/'source_snapshot/scripts/v24/run_history_plane_interaction.py'
        subprocess.run([sys.executable, '-u', str(script), '--suite', str(upstream), '--gpu', '0'], check=True)
    if not (upstream/'completed.json').exists():
        raise RuntimeError('X queue did not finish')


def close_idle_upstream_session():
    """Close only the known old experiment session after its queue has finished."""
    target = '=v24-history-plane-10'
    listed = subprocess.run(['tmux','list-panes','-s','-t',target,'-F',
                             '#{pane_pid}|#{pane_dead}|#{pane_current_command}|#{pane_tty}'],
                            capture_output=True, text=True)
    if listed.returncode:
        return 'absent'
    records = [line.split(None,3) for line in subprocess.check_output(
        ['ps','-eo','pid=,ppid=,tty=,comm='], text=True).splitlines()]
    shells = {'bash','sh','zsh','fish','dash'}
    for line in listed.stdout.splitlines():
        pid, dead, command, tty = line.split('|')
        if dead == '1':
            if any(r[0]==pid or r[1]==pid for r in records):
                return 'kept_busy'
        elif (command not in shells
              or not any(r[0]==pid and r[3] in shells for r in records)
              or any(r[1]==pid or (r[2]==tty.removeprefix('/dev/') and r[0]!=pid) for r in records)):
            return 'kept_busy'
    # A reused session with new windows/tasks must not be closed.
    again = subprocess.run(['tmux','list-panes','-s','-t',target,'-F',
                            '#{pane_pid}|#{pane_dead}|#{pane_current_command}|#{pane_tty}'],
                           capture_output=True,text=True)
    if again.returncode or again.stdout != listed.stdout:
        return 'kept_changed'
    subprocess.run(['tmux','kill-session','-t',target], check=True)
    return 'closed_idle'


def freeze_references(suite, payload):
    sources = [('K04', Path(payload['core']), 'taxibj_K04_seed7')]
    sources += [(f'X{i:02}', Path(payload['upstream']), f'taxibj_X{i:02}_seed7') for i in range(1, 11)]
    manifest = {}
    for label, origin, name in sources:
        job = load(origin/'plan.json')['jobs'][name]
        r = base.receipt(origin, job)
        if r is None:
            raise RuntimeError('Reference training incomplete: '+label)
        epath = origin/'evaluations'/f'{name}.json'; e = load(epath)
        if (e.get('status') != 'finished' or e.get('config_sha256') != digest(job['config'])
                or e.get('protocol_sha256') != digest(job['config']['experiment_plan']['protocol'])
                or set(e.get('sets', {})) != set(job['config']['experiment_plan']['protocol']['evaluations'])):
            raise RuntimeError('Reference evaluations incomplete: '+label)
        run = Path(r['run_dir'])
        files = {'result.json':origin/'results'/f'{name}.json', 'evaluations.json':epath,
                 'config.json':origin/'configs'/f'{name}.json', 'metrics.jsonl':run/'logs/metrics.jsonl',
                 'training_metadata.json':run/'training_metadata.json', 'runtime.json':run/'runtime.json'}
        for filename, src in files.items():
            rel = f'references/{label}/{filename}'; dst = suite/rel; sha = filehash(src)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
            if filehash(dst) != sha:
                raise RuntimeError('Reference evidence changed: '+rel)
            manifest[rel] = sha
    base.frozen(suite/'reference_manifest.json', manifest)


def reserve_run(suite, job):
    pointer = suite/'runs'/f'{job["variant"]}.json'
    if not pointer.exists():
        c = job['config']; stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run = Path(c['output_dir'])/c['data']['dataset_name']/'custom'/f'{stamp}_{job["name"]}_seed{c["seed"]}'/'random'/f'rate{c["data"]["mask"]["missing_rate"]:g}'/f'{stamp}_seed{c["seed"]}_bs{c["data"]["batch_size"]}'
        base.frozen(pointer, dict(run_dir=str(run)))
    return Path(load(pointer)['run_dir'])


def report(suite, payload):
    from report_history_followup import export_report
    return export_report(Path(payload['root']), suite, payload)


def run_queue(suite, payload):
    validate(suite, payload)
    finish_upstream(suite, payload)
    freeze_references(suite, payload)
    print('Previous tmux session:', close_idle_upstream_session(), flush=True)
    js = copy.deepcopy(payload['jobs'])
    prepare_initialization(suite, js, suite/'reference_initialization.pth')
    base.frozen(suite/'resolved_jobs.json', js)
    for name, job in js.items():
        base.frozen(suite/'configs'/f'{name}.json', job['config'])
    acceptance = suite/'acceptance/preflight.json'
    if not acceptance.exists() or load(acceptance).get('status') != 'passed':
        write(suite/'state.json', dict(phase='GPU0_batch32_preflight', time=time.time()))
        base.subprocess_run([sys.executable, '-u', str(suite/'source_snapshot/scripts/v24/preflight_history_followup.py'),
                             '--suite', str(suite)], suite/'launcher_logs/preflight.log', 0, visible=True)
    audit = load(acceptance)
    if (audit.get('status') != 'passed' or audit.get('source_sha256') != digest(payload['source'])
            or audit.get('configs_sha256') != digest({n:j['config'] for n,j in js.items()})
            or len(audit.get('cases', [])) != 24):
        raise RuntimeError('Invalid GPU preflight evidence')
    if not (suite/'started.json').exists():
        write(suite/'started.json', dict(time=time.time(), gpu=0))
    for name, job in js.items():
        write(suite/'state.json', dict(phase='training_and_evaluating', variant=name, time=time.time()))
        for path, stamp in payload['data'].items():
            st = Path(path).stat()
            if (st.st_size, st.st_mtime_ns) != (stamp['size'], stamp['mtime_ns']):
                raise RuntimeError('Data changed during queue: '+path)
        reserve_run(suite, job)
        try:
            base.launch(suite, job, 0)
            r = base.receipt(suite, job)
            base.evaluate_record(suite, name, Path(r['run_dir'])/'checkpoints/best.pth', r['config_sha256'],
                                 job['config']['experiment_plan']['protocol'], job['sources']['test'], 'evaluations', 0)
        except BaseException as error:
            write(suite/'failures'/f'{name}.json', dict(error=repr(error), time=time.time()))
            report(suite, payload)
            raise
        report(suite, payload)
    path = report(suite, payload)
    if any(r['status'] != 'finished' for r in load(suite/'summary.json').values()):
        raise RuntimeError('Final evidence audit incomplete')
    write(suite/'completed.json', dict(time=time.time(), count=12, report=str(path)))
    write(suite/'state.json', dict(phase='completed', time=time.time()))
    print('Finished Y01-Y12. Report:', path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path); p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    payload = load(a.suite/'plan.json') if a.suite else make_plan()
    suite = a.suite.resolve() if a.suite else ROOT/'outputs/v24-COE/experiments/history_followup'/digest(payload)[:16]
    if a.prepare_only:
        freeze(suite, payload); print(suite); return
    suite.parent.mkdir(parents=True, exist_ok=True)
    if a.worker:
        try:
            run_queue(suite, payload)
        except BaseException as error:
            write(suite/'queue_failure.json', dict(error=repr(error), time=time.time()))
            write(suite/'state.json', dict(phase='failed', error=repr(error), time=time.time()))
            raise
        return
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not a.suite:
            freeze(suite, payload)
        validate(suite, payload)
        subprocess.run([sys.executable, '-u', str(suite/'source_snapshot/scripts/v24/run_history_followup.py'),
                        '--suite', str(suite), '--worker'], check=True)


if __name__ == '__main__':
    main()
