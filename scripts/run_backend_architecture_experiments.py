#!/usr/bin/env python3
"""Sequential V23 upgrade + backend architecture studies; full TRAIN/VAL/TEST.

Only NEW_MLP is shared across the two studies, never rerun or selected by TEST.
Incomplete runs restart from epoch one (best weights are retained only in RAM).
No historical scores are imported; a scientific fingerprint protects resume.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import hashlib
import math
import os
from pathlib import Path
import statistics
import zipfile

import run_dual_moe_comparison as common
import run_scale_completion_experiments as runner
import train_scale_completion as trainer

ROOT = common.ROOT
NOTE = (
    'Study 1: NEW_MLP vs OLD_D, plus predeclared OLD_L at TaxiBJ random@0.8. '
    'This tests the entire upgrade (experts/shared/layout/balance), not one-factor attribution. '
    'Study 2: ST_LOCAL/ST_DILATED vs NEW_MLP and parameter-matched MLP_WIDE. '
    'Front E8/K4, shared pointwise head, router inputs, loss and budget are fixed in study 2. '
    'Full original train/val/test; fresh joint end-to-end training; best VAL MAE chooses epoch; '
    'best CPU-memory weights restored for exactly one final TEST; no checkpoint files. '
    'Candidate choice uses VAL only, TEST is descriptive. One seed is exploratory evidence, '
    'not statistical significance or full convergence evidence. No deadline truncation. '
    'All routed heads are evaluated densely; Top-K is sparse mixing, not sparse dispatch.'
)


def validate(p):
    if p.get('protocol') != 'backend-architecture-v1':
        raise ValueError('Expected backend-architecture-v1')
    if not p['datasets'] or len(set(p['datasets'])) != len(p['datasets']) or any(d not in common.SPECS for d in p['datasets']):
        raise ValueError('Distinct supported datasets required')
    for key in ('batch_size', 'val_epoch', 'cpu_threads'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'{key} must be positive integer')
    if not p['seeds'] or len(set(p['seeds'])) != len(p['seeds']) or any(type(s) is not int or s < 0 for s in p['seeds']):
        raise ValueError('Distinct nonnegative integer seeds required')
    for d in p['datasets']:
        if type(p['dataset_epochs'][d]) is not int or p['dataset_epochs'][d] < 1:
            raise ValueError('Positive per-dataset epochs required')
    points = [(x['pattern'], x['rate']) for x in p['points']]
    if not points or len(points) != len(set(points)) or any(m not in ('fixed', 'random') or r not in (.2,.4,.6,.8) for m,r in points):
        raise ValueError('Distinct valid mask/rate points required')
    if not set(m for m,r in points) <= set(p['patterns']) or not set(r for m,r in points) <= set(p['rates']):
        raise ValueError('Identity patterns/rates must cover all points')
    for point in p.get('extra_old_l', []):
        if point['dataset'] not in p['datasets'] or (point['pattern'],point['rate']) not in points:
            raise ValueError('OLD_L point must be included in main points')
    if not math.isfinite(p['timing_safety_factor']) or p['timing_safety_factor'] < 1:
        raise ValueError('timing_safety_factor must be >=1')
    expected = {'OLD_D':'upgrade','NEW_MLP':'upgrade','MLP_WIDE':'structure',
                'ST_LOCAL':'structure','ST_DILATED':'structure'}
    if {k:v['stage'] for k,v in p['variants'].items()} != expected:
        raise ValueError('Retain all five predeclared variants/stages')


def identity(p):
    record = common.identity(p)
    record['protocol'] = p['protocol']
    for path in (Path(__file__), ROOT/'scripts/run_scale_completion_experiments.py',
                 ROOT/'scripts/train_scale_completion.py', ROOT/'configs/presets/dual_moe_shared_topk.json'):
        record['code'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def jobs(p, suite):
    # Finish all upgrade comparisons before starting structure exploration.
    for stage in ('upgrade', 'structure'):
        for dataset in p['datasets']:
            for point in p['points']:
                for seed in p['seeds']:
                    variants = {k:v for k,v in p['variants'].items() if v['stage'] == stage}
                    if stage == 'upgrade' and {'dataset':dataset, **point} in p.get('extra_old_l', []):
                        variants['OLD_L'] = {'patch': common.merge(p['variants']['OLD_D']['patch'],
                            {'model': {'dual_moe': {'completion_blend':'learned','completion_alpha':.5}}})}
                    for variant, spec in variants.items():
                        cfg, paths = trainer.build_config(dataset, point['pattern'], point['rate'],
                            'learned_regions', seed, p['dataset_epochs'][dataset], 'dual_moe_shared_topk')
                        cfg = common.merge(cfg, spec['patch'])
                        cfg = common.merge(cfg, {'output_dir':str(suite/'runs'),
                            'data':{'batch_size':p['batch_size'], 'num_workers':0, 'drop_last':False},
                            'train':{'val_epoch':p['val_epoch'], 'save_best_checkpoint':False,
                                     'early_stopping':{'enabled':False}}})
                        key = f'{dataset}_{point["pattern"]}_rate{point["rate"]:g}_{variant}_seed{seed}'
                        yield {'key':key, 'dataset':dataset, **point, 'seed':seed, 'variant':variant,
                               'stage':stage, 'name':f'{variant}_seed{seed}', 'cfg':cfg, 'paths':paths}


def data_manifest(all_jobs):
    import numpy as np
    counts = {}
    for job in all_jobs:
        for split, path in job['paths'].items():
            if str(path) in counts:
                continue
            with zipfile.ZipFile(path) as archive:
                key = 'x_f_gt.npy' if 'x_f_gt.npy' in archive.namelist() else 'x_f.npy'
                with archive.open(key) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1,0):np.lib.format.read_array_header_1_0,
                              (2,0):np.lib.format.read_array_header_2_0}[version]
                    shape, fortran, dtype = reader(stream)
            if len(shape) != 5 or min(shape) < 1 or dtype.hasobject or fortran:
                raise ValueError(f'Invalid NCTHW source: {path}')
            counts[str(path)] = shape[0]
        job['expected_train_samples'] = counts[str(job['paths']['train'])]
    return {'selection':'ALL original samples, no copies or truncation', 'sample_counts':counts}


def completed(job):
    audit = Path(job['cfg']['output_dir']).parent/'logs'/f'{job["key"]}.status.json'
    if audit.exists() and common.load(audit).get('status') != 'verified':
        return None
    return runner.completed(job)


def summarize(all_jobs, suite):
    rows = []
    for job in all_jobs:
        result = completed(job)
        row = {k:job[k] for k in ('dataset','pattern','rate','seed','variant','stage')}
        row.update(status='complete' if result else 'incomplete', epochs=job['cfg']['train']['epochs'])
        if result:
            row.update({k:result[k] for k in ('val_mae','test_mae','test_rmse','best_epoch','run_dir')})
            row['val_rmse'] = result['best_val_metrics']['rmse']
        rows.append(row)
    common.write_json(suite/'summary.json', rows)
    fields = ['dataset','pattern','rate','seed','variant','stage','status','epochs',
              'val_mae','val_rmse','test_mae','test_rmse','best_epoch','run_dir']
    with (suite/'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    lookup = {(r['dataset'],r['pattern'],r['rate'],r['seed'],r['variant']):r for r in rows}
    comparisons = [('NEW_MLP','OLD_D'), ('NEW_MLP','OLD_L'), ('MLP_WIDE','NEW_MLP'),
                   ('ST_LOCAL','NEW_MLP'), ('ST_DILATED','NEW_MLP'),
                   ('ST_LOCAL','MLP_WIDE'), ('ST_DILATED','MLP_WIDE'), ('ST_DILATED','ST_LOCAL')]
    paired = []
    for d,m,r,s in sorted({k[:4] for k in lookup}):
        for candidate, baseline in comparisons:
            a,b = lookup.get((d,m,r,s,candidate)),lookup.get((d,m,r,s,baseline))
            if not a or not b or a['status'] != 'complete' or b['status'] != 'complete':
                continue
            item = dict(dataset=d,pattern=m,rate=r,seed=s,candidate=candidate,baseline=baseline)
            for metric in ('val_mae','val_rmse','test_mae','test_rmse'):
                item[metric+'_delta'] = a[metric]-b[metric]
                item[metric+'_pct'] = 100*(a[metric]-b[metric])/b[metric] if b[metric] else None
            paired.append(item)
    aggregate = []
    for candidate,baseline in comparisons:
        part = [r for r in paired if r['candidate']==candidate and r['baseline']==baseline]
        if part:
            aggregate.append({'candidate':candidate,'baseline':baseline,'completed_pairs':len(part),
                'val_mae_wins':sum(r['val_mae_delta']<0 for r in part),
                'test_mae_wins':sum(r['test_mae_delta']<0 for r in part),
                'mean_val_mae_pct':statistics.mean(r['val_mae_pct'] for r in part if r['val_mae_pct'] is not None)
                    if any(r['val_mae_pct'] is not None for r in part) else None})
    common.write_json(suite/'comparison.json', {'note':NOTE,'negative_delta_is_better':True,
                       'paired':paired,'aggregate':aggregate})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/presets/dual_moe_backend_architecture.json')
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--calibrate',action='store_true',help='Time disposable models only; no formal runs')
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--stage',choices=('all','upgrade','structure'),default='all')
    args = parser.parse_args()
    if not args.gpu.isdigit(): parser.error('One GPU index required')
    if sum((args.dry_run,args.calibrate,args.summary_only)) > 1: parser.error('Choose one diagnostic mode')
    p = common.load(common.resolve(args.config)); validate(p)
    record = identity(p); suite = common.resolve(p['output_dir'])/common.digest(record)[:16]
    all_jobs = list(jobs(p,suite)); manifest = data_manifest(all_jobs)
    selected = [j for j in all_jobs if args.stage=='all' or j['stage']==args.stage]
    # Structure comparisons depend on the same NEW_MLP anchor, never on a new run.
    if args.stage=='structure':
        selected = [j for j in all_jobs if j['variant']=='NEW_MLP']+selected
    print(f'[suite] {suite}\n[plan] {len(all_jobs)} total, {len(selected)} requested; '
          f'epochs={p["dataset_epochs"]}; FULL train/val/test; VAL every {p["val_epoch"]}',flush=True)
    print(f'[target] {p.get("target_time")} Asia/Shanghai, advisory only; no truncation',flush=True)
    if args.dry_run:
        for j in selected: print(f'{j["stage"]}: {j["key"]} TRAIN={j["expected_train_samples"]}')
        return
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu; os.environ['PYTHONUNBUFFERED']='1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]=str(p['cpu_threads'])
    suite.mkdir(parents=True,exist_ok=True)
    with (suite/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        common.write_json(suite/'protocol.json',record)
        common.write_json(suite/'data_manifest.json',manifest)
        common.write_json(suite/'plan.json',[{k:j[k] for k in ('key','stage','dataset','pattern','rate','seed','variant','expected_train_samples')} for j in all_jobs])
        for j in all_jobs: common.write_json(suite/'configs'/f'{j["key"]}.json',j['cfg'])
        if args.summary_only:
            rows=summarize(all_jobs,suite)
            print(f'[summary] {sum(r["status"]=="complete" for r in rows)}/{len(rows)}; {suite/"summary.csv"}')
            return
        remaining=[j for j in selected if completed(j) is None]
        if not remaining:
            summarize(all_jobs,suite); print('[done] All requested experiments complete.'); return
        # Calibrate each outstanding dataset/architecture freshly on THIS GPU.
        timings=runner.calibrate(p,suite,remaining)
        seconds=sum(timings[f'{j["dataset"]}/{j["variant"]}']['seconds_per_run'] for j in remaining)
        eta=datetime.now(runner.TZ)+timedelta(seconds=seconds)
        print(f'[ETA] {len(remaining)} runs, {seconds/3600:.2f} h, {eta:%Y-%m-%d %H:%M %Z}; not guaranteed',flush=True)
        target=p.get('target_time')
        if target and eta>datetime.strptime(target,'%Y-%m-%d %H:%M').replace(tzinfo=runner.TZ):
            print('[WARN] Estimate exceeds advisory target; all configured epochs will still run.',flush=True)
        if args.calibrate: return
        try:
            for i,j in enumerate(selected,1):
                if identity(p)!=record or common.load(common.resolve(args.config))!=p:
                    raise RuntimeError('Code/data/config changed. Rerun creates a separate scientific suite.')
                if completed(j):
                    print(f'[{i}/{len(selected)}] SKIP complete {j["key"]}',flush=True); continue
                print(f'[{i}/{len(selected)}] {j["stage"]} RUN {j["key"]}',flush=True)
                audit=suite/'logs'/f'{j["key"]}.status.json'
                common.write_json(audit,{'status':'running','fingerprint':common.digest(record)})
                runner.launch(j,suite)
                if identity(p)!=record or common.load(common.resolve(args.config))!=p:
                    common.write_json(audit,{'status':'invalid','reason':'Code/data/config changed during run'})
                    raise RuntimeError('Code/data/config changed during training; do not interpret mixed results.')
                common.write_json(audit,{'status':'verified','fingerprint':common.digest(record)})
                summarize(all_jobs,suite)
        finally:
            summarize(all_jobs,suite)
        print(f'[done] {suite/"summary.csv"}\n[paired] {suite/"comparison.json"}',flush=True)


if __name__=='__main__': main()
