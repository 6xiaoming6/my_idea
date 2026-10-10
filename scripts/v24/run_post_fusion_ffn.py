#!/usr/bin/env python3
"""Two additional TaxiBJ seed7 FFN trainings; reuse frozen K04 as reference."""
import argparse
import copy
import fcntl
import importlib.util
import json
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

REFERENCE_REL = 'outputs/v24-COE/experiments/core_validation/684cad05994af9a4'
REFERENCE_NAME = 'taxibj_K04_seed7'


def jobs():
    origin = load(ROOT/REFERENCE_REL/'plan.json')['jobs'][REFERENCE_NAME]
    result = {}
    for method, spec in load(ROOT/'configs/v24/post_fusion_ffn/variants.json').items():
        j = copy.deepcopy(origin)
        c = j['config']
        c['model']['coe']['post_fusion_ffn'] = {
            'enabled': True, 'sharing': spec['sharing'], 'expansion': 4,
            'pre_norm': True, 'residual': True, 'placement': 'after_fine_restore',
        }
        name = f'taxibj_{method}_seed7'
        c['experiment_plan'].update(suite='post_fusion_ffn', variant=name,
            method=spec['method'], reference=REFERENCE_NAME,
            scope='Two FFN-sharing variants on CMFF controlled delta; no additional views or free scales')
        j.update(variant=name, method=method, name=spec['name'], reference=REFERENCE_NAME)
        result[name] = j
    return result


def prepare_initialization(suite, all_jobs, reference_initialization):
    import torch
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.deterministic import state_hash
    old = torch.load(reference_initialization, map_location='cpu', weights_only=False)
    if state_hash(old['model']) != old['sha256']:
        raise RuntimeError('Reference initialization hash mismatch')
    audit = {}
    first_ffn_hash = None
    for name, j in all_jobs.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(j['config']['seed'])
            model = DualBranchSTImputer.from_config(j['config'])
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        common = {k: v for k, v in state.items() if not k.startswith('main_branch.post_ffns.')}
        if state_hash(common) != old['sha256']:
            raise RuntimeError('FFN construction changed K04 backbone initialization')
        for k, v in old['model'].items():
            if not torch.equal(state[k], v):
                raise RuntimeError('Common parameter mismatch '+k)
            state[k] = v.clone()
        for block in model.main_branch.post_ffns:
            sha = state_hash(block.state_dict())
            first_ffn_hash = first_ffn_hash or sha
            if sha != first_ffn_hash:
                raise RuntimeError('FFN initial copies differ')
        full_sha = state_hash(state)
        path = suite/'initialization'/f'{name}.pth'
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['sha256'] != full_sha or state_hash(saved['model']) != full_sha:
                raise RuntimeError('Frozen FFN initialization changed')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            torch.save({'model': state, 'sha256': full_sha, 'seed': j['config']['seed']}, temp)
            temp.replace(path)
        j['common_initialization'] = str(path)
        audit[name] = {'common_sha256': old['sha256'], 'full_sha256': full_sha,
            'ffn_initial_sha256': first_ffn_hash,
            'ffn_count': len(model.main_branch.post_ffns),
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    base.frozen(suite/'initialization_audit.json', audit)


def report(suite, payload):
    spec = importlib.util.spec_from_file_location('frozen_ffn_report',
        suite/'source_snapshot/scripts/v24/report_post_fusion_ffn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_report(ROOT, suite, payload)


def validate_evidence(suite, payload):
    for rel, sha in payload['source'].items():
        if filehash(suite/'source_snapshot'/rel) != sha:
            raise RuntimeError('Frozen source changed: '+rel)
    for path, stamp in payload['data'].items():
        if filehash(path) != stamp['sha256']:
            raise RuntimeError('Dataset changed: '+path)
    for label, item in payload['reference']['files'].items():
        if filehash(suite/'reference'/label) != item['sha256']:
            raise RuntimeError('Frozen K04 reference changed: '+label)


def run_queue(suite, payload, gpu):
    validate_evidence(suite, payload)
    all_jobs = copy.deepcopy(payload['jobs'])
    prepare_initialization(suite, all_jobs, suite/'reference/initialization.pth')
    base.frozen(suite/'resolved_jobs.json', all_jobs)
    for n, j in all_jobs.items():
        base.frozen(suite/'configs'/f'{n}.json', j['config'])
    if not (suite/'started.json').exists():
        write(suite/'started.json', {'time': time.time(), 'gpu': gpu})
    for n, j in all_jobs.items():
        for path, stamp in payload['data'].items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (stamp['size'], stamp['mtime_ns']):
                raise RuntimeError('Dataset changed: '+path)
        try:
            base.launch(suite, j, gpu)
            r = base.receipt(suite, j)
            base.evaluate_record(suite, n, Path(r['run_dir'])/'checkpoints/best.pth',
                r['config_sha256'], j['config']['experiment_plan']['protocol'],
                j['sources']['test'], 'evaluations', gpu)
        except BaseException as error:
            write(suite/'failures'/f'{n}.json', {'error': repr(error), 'time': time.time()})
            report(suite, payload)
            raise
        report(suite, payload)
    write(suite/'completed.json', {'time': time.time(), 'count': len(all_jobs)})
    print(f'Results: {suite}\nReport: {report(suite, payload)}', flush=True)


def make_plan(preflight):
    all_jobs = jobs()
    ref = ROOT/REFERENCE_REL
    r = load(ref/'results'/f'{REFERENCE_NAME}.json')
    if r['status'] != 'finished' or r['completed_epochs'] != 100:
        raise RuntimeError('K04 reference incomplete')
    run = Path(r['run_dir'])
    mapping = {'config.json': ref/'configs'/f'{REFERENCE_NAME}.json',
        'result.json': ref/'results'/f'{REFERENCE_NAME}.json',
        'evaluations.json': ref/'evaluations'/f'{REFERENCE_NAME}.json',
        'initialization.pth': ref/'initialization'/f'{REFERENCE_NAME}.pth',
        'training_metadata.json': run/'training_metadata.json',
        'metrics.jsonl': run/'logs/metrics.jsonl', 'runtime.json': run/'runtime.json'}
    source = base.manifest('taxibj')
    for rel in ['scripts/v24/README_POST_FUSION_FFN.md', 'tests/test_v24_post_fusion_ffn.py']:
        source[rel] = filehash(ROOT/rel)
    data = {}
    for p in next(iter(all_jobs.values()))['sources'].values():
        stat = Path(p).stat()
        data[p] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': filehash(p)}
    acceptance = load(preflight)
    if acceptance.get('status') != 'passed':
        raise RuntimeError('GPU preflight has not passed')
    if acceptance.get('model_sha256') != filehash(ROOT/'src/stmoe_imputer/models/post_fusion_ffn_coe.py'):
        raise RuntimeError('Preflight model differs from launch model')
    if acceptance.get('configs_sha256') != digest({n:j['config'] for n,j in all_jobs.items()}):
        raise RuntimeError('Preflight configurations differ from launch')
    return {'jobs': all_jobs, 'source': source, 'data': data, 'order': list(all_jobs),
        'acceptance': acceptance, 'reference': {'name': REFERENCE_NAME, 'suite': str(ref),
            'files': {k:{'path':str(p), 'sha256':filehash(p)} for k,p in mapping.items()}}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--suite', type=Path)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    a = p.parse_args()
    if a.gpu != 0: p.error('GPU0 only')
    if a.suite:
        suite = a.suite.resolve(); payload = load(suite/'plan.json')
    else:
        if not a.preflight: p.error('--preflight is required for a new frozen suite')
        payload = make_plan(a.preflight)
        suite = ROOT/'outputs/v24-COE/experiments/post_fusion_ffn'/digest(payload)[:16]
    if a.dry_run:
        print(json.dumps({'suite':str(suite), 'count':len(payload['jobs']), 'jobs':payload['jobs']}, ensure_ascii=False, indent=2));return
    if a.summary_only:
        validate_evidence(suite, payload);print(report(suite, payload));return
    suite.parent.mkdir(parents=True, exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        suite.mkdir(exist_ok=True)
        for rel, sha in payload['source'].items():
            dst = suite/'source_snapshot'/rel
            if not dst.exists():
                if a.suite: raise RuntimeError('Missing snapshot '+rel)
                dst.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/rel, dst)
        for label, item in payload['reference']['files'].items():
            dst = suite/'reference'/label
            if not dst.exists():
                if a.suite: raise RuntimeError('Missing reference '+label)
                dst.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(item['path'], dst)
        base.frozen(suite/'plan.json', payload)
        base.frozen(suite/'acceptance/preflight.json', payload['acceptance'])
        validate_evidence(suite, payload)
        command = [sys.executable, '-u', str(suite/'source_snapshot/scripts/v24/run_post_fusion_ffn.py'),
                   '--suite', str(suite), '--gpu', '0']
        code = subprocess.call(command, env=dict(os.environ, POST_FFN_FROZEN_PARENT='1'))
        if code: raise SystemExit(code)


if __name__ == '__main__':
    if os.environ.get('POST_FFN_FROZEN_PARENT') == '1':
        p = argparse.ArgumentParser();p.add_argument('--suite', type=Path, required=True);p.add_argument('--gpu', type=int, required=True)
        a = p.parse_args();suite = a.suite.resolve();payload = load(suite/'plan.json')
        ROOT = Path(next(iter(payload['jobs'].values()))['config']['output_dir']).parents[1]
        run_queue(suite, payload, a.gpu)
    else:
        main()
