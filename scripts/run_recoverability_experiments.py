#!/usr/bin/env python3
"""V23 recoverability: CPU geometry check, then matched single-GPU full runs.

No automatic training on import, no test-based selection, no deadline truncation.
Resume skips only verified complete TRAIN/VAL/best-restored TEST runs. Interrupted
jobs restart from epoch one because best weights are intentionally RAM-only.
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
import sys

import run_dual_moe_comparison as common
import run_scale_completion_experiments as runner
import run_backend_architecture_experiments as infrastructure
import train_scale_completion as trainer

ROOT = common.ROOT
VARIANTS = {'A_ST_DILATED': 'off', 'B_SCALAR': 'scalar', 'C_MATRIX': 'matrix',
            'D_RECOVERY': 'constrained', 'E_FIT': 'fit_only'}
PAIRS = [('D_RECOVERY', 'A_ST_DILATED'), ('B_SCALAR', 'A_ST_DILATED'),
         ('C_MATRIX', 'B_SCALAR'), ('D_RECOVERY', 'B_SCALAR'),
         ('D_RECOVERY', 'C_MATRIX'), ('D_RECOVERY', 'E_FIT'), ('E_FIT', 'A_ST_DILATED')]
NOTE = (
    'A=unchanged ST_DILATED. B=scalar coverage + constant fit + isotropically attenuated prior. '
    'C=matrix fit + unrestricted coefficient prior. D=matrix fit + Q-restricted coefficient prior. '
    'E=matrix fit without learned coefficient prior (its matching head is multiplied by zero). '
    'B/C/D/E have identical parameter shapes and auxiliary weight; C vs D isolates prior restriction. '
    'E is a neural model with a classical local-fit branch, NOT a standalone classical baseline. '
    'Fixed normalized [1,x,y,time] basis; learned memberships remain end-to-end trainable. '
    'No exact recoverability/data-consistency guarantee for the final neural output; weakness is not '
    'calibrated uncertainty. Synthetic geometry checks establish algebra, not real-data efficacy. '
    'Full original TRAIN/VAL/TEST; matched per-dataset epochs and seeds; best VAL MAE in CPU RAM '
    'restored for exactly one final TEST; no checkpoint files. Select by VAL, TEST descriptive only. '
    'Single seed is exploratory, not significance. Negative paired differences indicate improvement. '
    'No raw cross-dataset MAE averaging. New fingerprint means a fresh comparison, no historical reuse.'
)


def resolve_profile(raw, profile='full'):
    # Profiles live in the same JSON. Preserve the full budget; never silently
    # select different epochs after inspecting accuracy or partially running.
    p = {key: value for key, value in raw.items() if key != 'profiles'}
    if profile != 'full':
        if profile not in raw.get('profiles', {}):
            raise ValueError(f'Missing profile {profile}')
        p = common.merge(p, raw['profiles'][profile])
    return p


def deadline_time(value):
    return datetime.strptime(value, '%Y-%m-%d %H:%M').replace(tzinfo=runner.TZ) if value else None


def meets_deadline(eta, target, buffer_minutes=0):
    return target is None or eta + timedelta(minutes=buffer_minutes) <= target


def validate(p):
    if p.get('protocol') != 'recoverability-v1' or p.get('variants') != VARIANTS:
        raise ValueError('Keep the five predeclared recoverability controls')
    if not p['datasets'] or len(set(p['datasets'])) != len(p['datasets']) or any(d not in common.SPECS for d in p['datasets']):
        raise ValueError('Distinct supported datasets required')
    for key in ('batch_size', 'val_epoch', 'cpu_threads'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not p['seeds'] or len(set(p['seeds'])) != len(p['seeds']) or any(type(s) is not int or s < 0 for s in p['seeds']):
        raise ValueError('Distinct nonnegative seeds required')
    for d in p['datasets']:
        if type(p['dataset_epochs'][d]) is not int or p['dataset_epochs'][d] < 1:
            raise ValueError('Positive per-dataset epoch budgets required')
    if p['patterns'] != ['fixed', 'random'] or not p['rates'] or len(set(p['rates'])) != len(p['rates']) or any(r not in (.2, .4, .6, .8) for r in p['rates']):
        raise ValueError('Use both patterns and distinct supported rates')
    points = [(x['pattern'], x['rate']) for x in p['points']]
    if len(points) != len(set(points)) or set(points) != {(m, r) for m in p['patterns'] for r in p['rates']}:
        raise ValueError('Cross both fixed/random with ALL declared rates (avoid confounding)')
    if set(p['recoverability']) != {'ridge', 'strength'}:
        raise ValueError('Declare common ridge and strength, no per-candidate tuning')
    for name, value in [*p['recoverability'].items(), ('auxiliary_weight', p['auxiliary_weight']),
                        ('timing_safety_factor', p['timing_safety_factor'])]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Positive finite {name} required')
    if p['timing_safety_factor'] < 1:
        raise ValueError('Timing safety factor must be >=1')
    if p.get('target_time'):
        deadline_time(p['target_time'])
    buffer = p.get('finish_buffer_minutes', 0)
    if type(buffer) is not int or buffer < 0:
        raise ValueError('finish_buffer_minutes must be a nonnegative integer')


def identity(p):
    record = common.identity(p)
    record['protocol'] = p['protocol']
    for path in (Path(__file__), Path(runner.__file__), Path(infrastructure.__file__), Path(trainer.__file__),
                 ROOT/'configs/presets/dual_moe_st_dilated.json'):
        record['code'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def jobs(p, suite):
    for dataset in p['datasets']:
        for point in p['points']:
            for seed in p['seeds']:
                for variant, mode in p['variants'].items():
                    cfg, paths = trainer.build_config(dataset, point['pattern'], point['rate'],
                        'learned_regions', seed, p['dataset_epochs'][dataset], 'dual_moe_st_dilated')
                    cfg = common.merge(cfg, {
                        'output_dir': str(suite/'runs'),
                        'model': {'dual_moe': {'recoverability': {'mode': mode, **p['recoverability']}}},
                        'loss': {'dual_moe_recoverability_weight': 0. if mode == 'off' else p['auxiliary_weight']},
                        'data': {'batch_size': p['batch_size'], 'num_workers': 0, 'drop_last': False},
                        'train': {'val_epoch': p['val_epoch'], 'save_best_checkpoint': False,
                                  'early_stopping': {'enabled': False}}})
                    key = f'{dataset}_{point["pattern"]}_rate{point["rate"]:g}_{variant}_seed{seed}'
                    yield dict(key=key, dataset=dataset, **point, seed=seed, variant=variant,
                               mode=mode, name=f'{variant}_seed{seed}', cfg=cfg, paths=paths)


def completed(job):
    audit = Path(job['cfg']['output_dir']).parent/'logs'/f'{job["key"]}.status.json'
    try:
        verified = common.load(audit).get('status') == 'verified'
    except (OSError, ValueError, AttributeError):
        verified = False
    if not verified:
        return None
    result = runner.completed(job)
    if result and job['mode'] != 'off':
        for metrics in (result['best_val_metrics'], result['test_metrics']):
            for scale in ('mid', 'coarse'):
                for suffix in ('count', 'branch_mae', 'weakness', 'mae'):
                    value = metrics.get(f'recovery_{scale}_all_{suffix}')
                    if value is None or not math.isfinite(value):
                        return None
    return result


def summarize(all_jobs, suite):
    rows = []
    for j in all_jobs:
        found = completed(j)
        row = {k: j[k] for k in ('dataset', 'pattern', 'rate', 'seed', 'variant', 'mode')}
        row.update(status='complete' if found else 'incomplete', epochs=j['cfg']['train']['epochs'])
        if found:
            row.update({k: found[k] for k in ('val_mae', 'test_mae', 'test_rmse', 'best_epoch', 'run_dir')})
            row['val_rmse'] = found['best_val_metrics']['rmse']
            row['diagnostics'] = {stage: {k: v for k, v in metrics.items() if k.startswith(('recovery_', 'l_recoverability'))}
                                  for stage, metrics in [('best_val', found['best_val_metrics']), ('test', found['test_metrics'])]}
        rows.append(row)
    common.write_json(suite/'summary.json', rows)
    fields = ['dataset', 'pattern', 'rate', 'seed', 'variant', 'mode', 'status', 'epochs',
              'val_mae', 'val_rmse', 'test_mae', 'test_rmse', 'best_epoch', 'run_dir']
    with (suite/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    lookup = {(r['dataset'], r['pattern'], r['rate'], r['seed'], r['variant']): r for r in rows}
    paired = []
    for d, m, r, s in sorted({key[:4] for key in lookup}):
        for candidate, control in PAIRS:
            a, b = lookup.get((d, m, r, s, candidate)), lookup.get((d, m, r, s, control))
            if not a or not b or a['status'] != 'complete' or b['status'] != 'complete':
                continue
            item = dict(dataset=d, pattern=m, rate=r, seed=s, candidate=candidate, control=control)
            for metric in ('val_mae', 'val_rmse', 'test_mae', 'test_rmse'):
                item[metric+'_delta'] = a[metric]-b[metric]
                item[metric+'_pct'] = 100*(a[metric]-b[metric])/b[metric] if b[metric] else None
            paired.append(item)
    groups = []
    for d, m, r in sorted({key[:3] for key in lookup}):
        expected = sorted({key[3] for key in lookup if key[:3] == (d, m, r)})
        for candidate, control in PAIRS:
            part = [x for x in paired if (x['dataset'], x['pattern'], x['rate'], x['candidate'], x['control']) == (d, m, r, candidate, control)]
            item = dict(dataset=d, pattern=m, rate=r, candidate=candidate, control=control,
                        expected_seeds=expected, completed_seeds=sorted(x['seed'] for x in part),
                        status='complete' if len(part) == len(expected) else 'incomplete')
            for metric in ('val_mae', 'test_mae', 'test_rmse'):
                values = [x[metric+'_pct'] for x in part if x[metric+'_pct'] is not None]
                if values:
                    item[metric+'_pct_mean'] = statistics.mean(values)
                    item[metric+'_pct_std'] = statistics.stdev(values) if len(values) > 1 else None
                    item[metric+'_wins'] = sum(x[metric+'_delta'] < 0 for x in part)
            groups.append(item)
    common.write_json(suite/'comparison.json', {'note': NOTE, 'paired': paired, 'groups': groups})
    return rows


def geometry_check(ridge):
    """Controlled affine fields with equal observed counts; no neural training."""
    import torch
    sys.path.insert(0, str(ROOT/'src'))
    from stmoe_imputer.models.recoverability import spacetime_basis, observation_system, ridge_decomposition, aligned_evaluate
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        phi = spacetime_basis(3, 5, 5, 'cpu')
        coeff = torch.randn(64, 4, 1)
        values = torch.matmul(phi, coeff).reshape(64, 1, 3, 5, 5)
        patterns = {}
        for name in ('spread', 'spatial_line', 'single_time'):
            m = torch.zeros(1, 1, 3, 5, 5)
            if name == 'spread':
                for t in range(3):
                    for y, x in ((0,0), (0,4), (4,0), (4,4)): m[0,0,t,y,x] = 1
            elif name == 'spatial_line':
                for t in range(3): m[0,0,t,2,[0,1,3,4]] = 1
            else:
                for index in (0,2,4,5,7,9,15,17,19,20,22,24): m[0,0,1,index//5,index%5] = 1
            patterns[name] = m
        shared_missing = ~torch.stack([v.bool() for v in patterns.values()]).any(0)
        rows = []
        for name, single_mask in patterns.items():
            m = single_mask.expand(64,-1,-1,-1,-1)
            a = torch.ones(64, 1, 3, 25, 1)
            g, rhs, cov = observation_system(a, m, values, phi)
            obs, q = ridge_decomposition(g, rhs, ridge)
            prediction = aligned_evaluate(a, obs, phi).reshape_as(values)
            weak = (phi * torch.matmul(phi, q[0,0,0])).sum(-1) / phi.square().sum(-1)
            missing = shared_missing.expand_as(values)
            errors = (prediction-values)[missing]
            rows.append(dict(pattern=name, observed_count=int(single_mask.sum()), coverage=float(cov[0,0,0]),
                             gram_rank=int(torch.linalg.matrix_rank(g[0,0,0])),
                             gram_eigenvalues=torch.linalg.eigvalsh(g[0,0,0]).tolist(),
                             common_missing_count=int(shared_missing.sum()),
                             common_missing_mae=float(errors.abs().mean()), common_missing_rmse=float(errors.square().mean().sqrt()),
                             common_missing_weakness=float(weak[shared_missing.reshape(-1)].mean())))
    return {'note': 'Classical local affine/ridge mechanism check with FIXED single-region assignments. '
                    'All three layouts have 12 observed values. Errors use the SAME hidden query positions. '
                    'Zero prior, 64 random affine fields, seed42; no fitting/tuning on TEST and no real-data or model-performance claim.',
            'ridge': ridge, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/presets/dual_moe_recoverability_experiments.json')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--profile', choices=('full', 'tonight'), default='full',
                        help='full=60 long-budget jobs; tonight=30 matched high-missing screening jobs')
    parser.add_argument('--deadline', help='Advisory target only, never blocks training: Asia/Shanghai YYYY-MM-DD HH:MM')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--mechanism-only', action='store_true', help='CPU affine geometry check only, no training')
    modes.add_argument('--calibrate', action='store_true', help='Disposable timing models, no formal training')
    modes.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    if not args.gpu.isdigit(): parser.error('One GPU index required')
    raw = common.load(common.resolve(args.config))
    p = resolve_profile(raw, args.profile); validate(p)
    try:
        target = deadline_time(args.deadline or p.get('target_time'))
    except ValueError:
        parser.error('Deadline format: YYYY-MM-DD HH:MM (Asia/Shanghai)')
    record = identity(p); suite = common.resolve(p['output_dir'])/common.digest(record)[:16]
    all_jobs = list(jobs(p, suite)); manifest = infrastructure.data_manifest(all_jobs)
    print(f'[suite] {suite}\n[plan] profile={args.profile}; {len(all_jobs)} full-data TRAIN/VAL/TEST jobs; '
          f'epochs={p["dataset_epochs"]}; VAL every {p["val_epoch"]}; seed={p["seeds"]}', flush=True)
    if target:
        print(f'[target] {target:%Y-%m-%d %H:%M %Z}; reserve {p.get("finish_buffer_minutes",0)} min; '
              'advisory only; exceeding this time never blocks or stops training', flush=True)
    if args.dry_run:
        for j in all_jobs: print(f'{j["key"]}: TRAIN={j["expected_train_samples"]}, mode={j["mode"]}')
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONUNBUFFERED'] = '1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(p['cpu_threads'])
    suite.mkdir(parents=True, exist_ok=True)
    with (suite/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common.write_json(suite/'protocol.json', record)
        common.write_json(suite/'data_manifest.json', manifest)
        common.write_json(suite/'plan.json', [{k:j[k] for k in ('key','dataset','pattern','rate','seed','variant','mode','expected_train_samples')} for j in all_jobs])
        for j in all_jobs: common.write_json(suite/'configs'/f'{j["key"]}.json', j['cfg'])
        if args.summary_only:
            rows = summarize(all_jobs, suite)
            print(f'[summary] {sum(r["status"]=="complete" for r in rows)}/{len(rows)}: {suite/"summary.csv"}')
            return
        geometry = geometry_check(p['recoverability']['ridge'])
        common.write_json(suite/'geometry.json', geometry)
        with (suite/'geometry.log').open('w') as stream:
            stream.write(geometry['note']+'\n')
            for row in geometry['rows']:
                line = f'{row["pattern"]}: observations={row["observed_count"]}, rank={row["gram_rank"]}, common-query RMSE={row["common_missing_rmse"]:.6f}, weakness={row["common_missing_weakness"]:.6f}'
                stream.write(line+'\n'); print('[geometry] '+line, flush=True)
        if args.mechanism_only: return
        remaining = [j for j in all_jobs if completed(j) is None]
        if not remaining:
            summarize(all_jobs, suite); print('[done] All experiments already verified.'); return
        timings = runner.calibrate(p, suite, remaining)
        seconds = sum(timings[f'{j["dataset"]}/{j["variant"]}']['seconds_per_run'] for j in remaining)
        eta = datetime.now(runner.TZ)+timedelta(seconds=seconds)
        print(f'[ETA] {len(remaining)} runs; {seconds/3600:.2f} h; {eta:%Y-%m-%d %H:%M %Z} (estimate only)', flush=True)
        within_target = meets_deadline(eta, target, p.get('finish_buffer_minutes', 0))
        common.write_json(suite/'schedule.json', {'profile':args.profile,
            'target':target.isoformat() if target else None, 'estimated_finish':eta.isoformat(),
            'remaining_runs':len(remaining), 'accepted':True, 'deadline_mode':'advisory',
            'within_target':within_target,
            'finish_buffer_minutes':p.get('finish_buffer_minutes',0)})
        if not within_target:
            print('[WARN] Estimate plus reserve exceeds the advisory target. '
                  'This does NOT block training; all configured jobs and epochs are retained.', flush=True)
        if args.calibrate: return
        try:
            for i, j in enumerate(all_jobs, 1):
                if identity(p) != record or resolve_profile(common.load(common.resolve(args.config)), args.profile) != p:
                    raise RuntimeError('Code/data/config changed. Rerun creates a NEW suite; do not mix results.')
                if completed(j):
                    print(f'[{i}/{len(all_jobs)}] SKIP verified {j["key"]}', flush=True); continue
                print(f'[{i}/{len(all_jobs)}] RUN {j["key"]}', flush=True)
                audit = suite/'logs'/f'{j["key"]}.status.json'
                common.write_json(audit, {'status':'running', 'fingerprint':common.digest(record)})
                runner.launch(j, suite)
                if identity(p) != record or resolve_profile(common.load(common.resolve(args.config)), args.profile) != p:
                    common.write_json(audit, {'status':'invalid', 'reason':'Code/data/config changed during training'})
                    raise RuntimeError('Scientific fingerprint changed during training; results not accepted.')
                common.write_json(audit, {'status':'verified', 'fingerprint':common.digest(record)})
                if not completed(j):
                    common.write_json(audit, {'status':'invalid', 'reason':'Missing/invalid recoverability diagnostics'})
                    raise RuntimeError(f'Recoverability audit failed: {audit}')
                summarize(all_jobs, suite)
        finally:
            summarize(all_jobs, suite)
        print(f'[done] {suite/"summary.csv"}\n[paired] {suite/"comparison.json"}', flush=True)


if __name__ == '__main__': main()
