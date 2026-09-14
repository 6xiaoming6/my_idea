#!/usr/bin/env python3
"""Matched scale-completion comparison, single GPU, restartable complete jobs.

No test-based candidate selection and no deadline-driven epoch truncation.
--calibrate only performs short timing updates; it never launches formal runs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timedelta
import hashlib
import io
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile
from zoneinfo import ZoneInfo

import run_dual_moe_comparison as common

ROOT = common.ROOT
TZ = ZoneInfo('Asia/Shanghai')


def is_topk(p):
    return is_stability(p) or p.get('protocol') in ('dual-topk-full-2x2-v1', 'front-expert-count-full-v1', 'front-routing-matched-v1', 'target-readout-three-dataset-v1', 'backend-routing-matched-v1', 'dual-moe-coverage-v1', 'backend-confirmation-v1')


def is_target_study(p):
    return is_stability(p) or p.get('protocol') in ('target-readout-three-dataset-v1', 'backend-routing-matched-v1', 'dual-moe-coverage-v1', 'backend-confirmation-v1')


def is_stability(p):
    return p.get('protocol') == 'backend-stability-v1'


def stability_variants():
    return {name: {'role': 'main', 'patch': {
        'model': {'dual_moe': {'aggregation_mode': 'topk', 'aggregation_top_k': 4,
            'completion_mode': mode, 'completion_top_k': 3, 'completion_blend': blend,
            'completion_alpha': .5, 'backend_diagnostics': True}},
        'loss': {'dual_moe_aggregation_balance_weight': .001, 'dual_moe_completion_balance_weight': 0.}}}
        for name,mode,blend in [('U','uniform','none'), ('D','topk','none'),
                                ('H','topk','fixed'), ('L','topk','learned')]}


STABILITY_NOTE = ('U=equal, D=dense conditional, H=0.5 equal + 0.5 conditional, '
    'L=global sigmoid alpha initialized 0.5, jointly learned. Front E8/Top4 unchanged but '
    'jointly retrained, not frozen. Backend balance zero; same full TRAIN/VAL/TEST budget. '
    'All runs fresh (new scientific code/diagnostics), no historical reuse. '
    'Negative paired change is better. Compare H/D and H/U, L/H and L/D and L/U; '
    'VAL selects best epochs; TEST descriptive, never tune from TEST. Three seeds are '
    'descriptive, not significance proof. This is NOT a front/back 2x2 ablation. '
    'Density bins use mask-only border-corrected spatial 3x3 fractions including center: '
    '[0,1/3), [1/3,2/3), [2/3,1]. Expert oracle is retrospective best SINGLE expert per '
    'missing channel element, not deployable or a lower bound on convex fusion. '
    'Gradient norms are preclip total-objective norms, not isolated main-loss gradients.')


def stability_jobs(p, suite):
    for job in confirmation_jobs(p, suite):
        job.pop('confirmation_study')
        job['stability_study'], job['stage'] = True, 'backend_stability'
        yield job


def is_confirmation(p):
    return p.get('protocol') == 'backend-confirmation-v1'


def confirmation_jobs(p, suite):
    for point in p['points']:
        for seed in p['seeds']:
            sub = {**p, 'protocol': 'target-readout-three-dataset-v1', 'datasets': [point['dataset']],
                   'patterns': ['random'], 'rate': point['rate'], 'seeds': [seed], 'ablation_seed': seed}
            for job in full_topk_jobs(sub, suite):
                d, rate = job['dataset'], job['rate']
                job['key'] = f'{d}_random_rate{rate:g}_{job["variant"]}_seed{seed}'
                job['cfg']['data']['mask']['train_csv'] = str(suite/f'data/{d}/random_rate{rate:g}_train.csv')
                job['confirmation_study'], job['stage'] = True, 'backend_confirmation'
                yield job


def reference_files(p):
    """Reference changes affect the new suite identity, not historical files."""
    if not is_confirmation(p) or not p.get('reuse_enabled', True):
        return []
    root = common.resolve(p['reuse_suite'])
    paths = [root/'protocol.json', root/'data/BikeNYC/selection.json',
             root/'data/BikeNYC/train.npz', root/'data/BikeNYC/random_rate0.4_train.csv']
    for name in ('Q10_seed42', 'Q11_seed42', 'SB_seed42', 'Q10_seed2026', 'Q11_seed2026'):
        for run in sorted((root/f'runs/BikeNYC/ablation/{name}/random/rate0.4').glob('*')):
            paths += [run/'config.json']+[run/'logs'/f for f in ('metrics.jsonl','train.log','val.log','test.log')]
    return [{'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            if path.is_file() else {'path': str(path), 'missing': True} for path in paths]


def comparable_config(cfg):
    # Ignore ONLY regenerated artifact locations. Batch/epochs/LR/loss/masks,
    # selection limit and all remaining data settings must be exactly equal.
    return common.merge(cfg, {'output_dir': '<output>', 'data': {
        'mask': {'train_csv': '<train-mask>'}, 'training_selection': {'manifest': '<manifest>'}}})


def bind_reuse(p, all_jobs):
    """Fail closed to fresh training; never copy scores or weights into new runs."""
    audit = []
    if not is_confirmation(p) or not p.get('reuse_enabled', True):
        return audit
    root = common.resolve(p['reuse_suite'])
    try:
        old = common.load(root/'protocol.json')
        if old.get('protocol') != 'dual-moe-coverage-v1':
            raise ValueError('Reference must be the declared coverage protocol')
        if old['policy']['cpu_threads'] != p['cpu_threads']:
            raise ValueError('CPU thread policy differs from reference')
        now = identity(p)
        # The only intentional code difference is this scheduler's new protocol
        # and reuse/summary support. ALL model, loss, data and training code and
        # preset hashes must match. A changed scientific implementation reruns.
        scheduler = str(Path(__file__).relative_to(ROOT))
        old_code = {k:v for k,v in old['code'].items() if k != scheduler}
        new_code = {k:v for k,v in now['code'].items() if k != scheduler}
        if old_code != new_code:
            raise ValueError('Scientific code/preset hash mismatch')
        sources = {s['path']: s for s in old['sources']}
        needed = common.identity({**p, 'datasets':['BikeNYC'], 'rates':[.4]})['sources']
        if any(sources.get(s['path']) != s for s in needed):
            raise ValueError('Reference source data/config identity mismatch')
        refs = {(j['dataset'],j['pattern'],j['rate'],j['seed'],j['variant']):j for j in jobs(old['policy'],root)}
        # Prove old TRAIN cache is full, unmodified source data, not merely a
        # manifest claiming all samples. BikeNYC is deliberately the only reuse dataset.
        import numpy as np
        manifest = common.load(root/'data/BikeNYC/selection.json')
        n = p['expected_train_samples']['BikeNYC']
        if (manifest['indices'] != list(range(n)) or manifest['original_train_windows'] != n
                or manifest['selected_train_windows'] != n):
            raise ValueError('Reference is not full TRAIN')
        with np.load(ROOT/'data/BikeNYC/bikenyc_train.npz', allow_pickle=False) as source, np.load(root/'data/BikeNYC/train.npz', allow_pickle=False) as cache:
            key = 'x_f_gt' if 'x_f_gt' in source else 'x_f'
            if not np.array_equal(source[key], cache['x_f_gt']):
                raise ValueError('Reference TRAIN cache differs from source')
        if (root/'data/BikeNYC/random_rate0.4_train.csv').read_bytes() != (ROOT/'data/BikeNYC/random_mask/0.4/train.csv').read_bytes():
            raise ValueError('Reference mask cache differs from source')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [{'status':'disabled', 'reason':str(exc), 'reference_suite':str(root)}]
    for job in all_jobs:
        if job['dataset'] != 'BikeNYC':
            continue
        key = tuple(job[k] for k in ('dataset','pattern','rate','seed','variant'))
        ref = refs.get(key)
        result = completed(ref) if ref is not None else None
        if result and comparable_config(ref['cfg']) == comparable_config(job['cfg']):
            job['reuse_job'], job['reuse_suite'] = ref, str(root)
            audit.append({'key':job['key'], 'status':'eligible', 'run_dir':result['run_dir'],
                          'reference_suite':str(root), 'reason':'Scientific hashes, exact settings, full data/masks and best-memory test verified'})
        else:
            audit.append({'key':job['key'], 'status':'fresh', 'reason':'Missing/incomplete reference or different settings'})
    return audit


def is_coverage_study(p):
    return p.get('protocol') == 'dual-moe-coverage-v1'


def coverage_variants():
    # No backend load balance. SF/FD both dense with zero front balance:
    # FD vs SF isolates conditional vs global weights without Top-K confounding.
    settings = {'Q00': ('uniform', 4, 'uniform', 0.),
                'Q10': ('topk', 4, 'uniform', .001),
                'Q01': ('uniform', 4, 'topk', 0.),
                'Q11': ('topk', 4, 'topk', .001),
                'SF': ('static', 8, 'topk', 0.),
                'FD': ('topk', 8, 'topk', 0.),
                'SB': ('topk', 4, 'static', .001)}
    return {name: {'role': 'main' if name.startswith('Q') else 'ablation',
                   'patch': {'model': {'dual_moe': {'aggregation_mode': front,
                       'aggregation_top_k': k, 'completion_mode': back, 'completion_top_k': 3}},
                       'loss': {'dual_moe_aggregation_balance_weight': wf,
                                'dual_moe_completion_balance_weight': 0.}}}
            for name, (front, k, back, wf) in settings.items()}


def resolve_policy(p, budget=None):
    if not is_coverage_study(p):
        if budget is not None:
            raise ValueError('--budget is only supported by the coverage protocol')
        return p
    selected = budget or p['default_budget']
    if selected not in p['budgets']:
        raise ValueError(f'Unknown budget: {selected}')
    return {**common.merge(p, p['budgets'][selected]), 'selected_budget': selected}


def coverage_jobs(p, suite):
    # Complete matched four-way blocks, all 24 conditions first. Supplemental
    # points are predeclared, never selected after inspecting test rankings.
    points = [(d, m, r) for m in p['patterns'] for r in p['rates'] for d in p['datasets']]
    plans = [('coverage', p['seeds'][0], point, ['Q00', 'Q10', 'Q01', 'Q11']) for point in points]
    plans += [('replication', seed, (d, point['pattern'], point['rate']), ['Q00', 'Q10', 'Q01', 'Q11'])
              for seed in p['seeds'][1:] for point in p['mechanism_points'] for d in p['datasets']]
    plans += [('mechanism', p['seeds'][0], (d, point['pattern'], point['rate']), ['SF', 'FD', 'SB'])
              for point in p['mechanism_points'] for d in p['datasets']]
    for stage, seed, (dataset, pattern, rate), variants in plans:
        sub = {**p, 'protocol': 'target-readout-three-dataset-v1', 'datasets': [dataset],
               'patterns': [pattern], 'rate': rate, 'seeds': [seed], 'ablation_seed': seed,
               'variants': {v: p['variants'][v] for v in variants}}
        for job in full_topk_jobs(sub, suite):
            job['key'] = f'{dataset}_{pattern}_rate{rate:g}_{job["variant"]}_seed{seed}'
            job['cfg']['data']['mask']['train_csv'] = str(suite/f'data/{dataset}/{pattern}_rate{rate:g}_train.csv')
            job['coverage_study'], job['stage'] = True, stage
            yield job


def summarize_coverage(all_jobs, suite):
    # Reuse paired/interaction statistics, but NEVER mix rates in a lookup or
    # mean. Historical single-rate protocols and their summaries stay unchanged.
    rows, grouped, paired, interactions, paired_groups = [], [], [], [], []
    for rate in sorted({j['rate'] for j in all_jobs}):
        part = suite/'analysis'/f'rate{rate:g}'
        subset = [j for j in all_jobs if j['rate'] == rate]
        rows.extend(summarize(subset, part))
        analysis = common.load(part/'comparison.json')
        for key, dest in [('groups', grouped), ('paired', paired), ('interaction', interactions), ('paired_groups', paired_groups)]:
            dest.extend({**r, 'rate': rate} for r in analysis[key])
    common.write_json(suite/'summary.json', rows)
    fields = ['dataset', 'pattern', 'rate', 'seed', 'variant', 'status', 'val_mae', 'test_mae',
              'test_rmse', 'best_epoch', 'epochs', 'run_dir']
    confirmation = any(j.get('confirmation_study') for j in all_jobs)
    if confirmation:
        fields += ['result_origin', 'reference_suite']
    with (suite/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    effects = []
    # Per-dataset macro percentage changes across coverage points; don't average
    # raw MAE across differently scaled datasets, or give replicated points more weight.
    first_seed = all_jobs[0]['seed']
    for dataset in ([] if confirmation else sorted({j['dataset'] for j in all_jobs})):
        for candidate, control in [('Q11', 'Q01'), ('Q11', 'Q10'), ('Q11', 'Q00')]:
            values = [r for r in paired if r['dataset'] == dataset and r['seed'] == first_seed
                      and r['candidate'] == candidate and r['control'] == control]
            expected = len({(j['pattern'], j['rate']) for j in all_jobs if j['dataset'] == dataset and j['variant'] == candidate})
            effect = {'dataset': dataset, 'candidate': candidate, 'control': control,
                      'expected_points': expected, 'complete_points': len(values),
                      'status': 'complete' if len(values) == expected else 'incomplete'}
            for metric in ('val_mae', 'test_mae', 'test_rmse'):
                numbers = [r[metric+'_change_percent'] for r in values]
                if numbers and all(v is not None for v in numbers):
                    effect[metric] = {'macro_change_percent': statistics.mean(numbers),
                                      'wins': sum(v < 0 for v in numbers),
                                      'improvements_at_least_1_percent': sum(v <= -1 for v in numbers),
                                      'regressions_at_least_1_percent': sum(v >= 1 for v in numbers)}
            effects.append(effect)
    diagnostics = []
    for row in rows:
        if row['status'] != 'complete':
            continue
        cfg = common.load(Path(row['run_dir'])/'config.json')
        import json
        entries = [json.loads(line) for line in (Path(row['run_dir'])/'logs/metrics.jsonl').read_text().splitlines()]
        valid = [r for r in entries if r.get('val') is not None]
        tail = valid[-min(5, len(valid)):]
        diagnostics.append({k: row[k] for k in ('dataset', 'pattern', 'rate', 'seed', 'variant')} |
                           {'best_epoch': row['best_epoch'], 'epochs': row['epochs'],
                            'best_in_last_10_percent': row['best_epoch'] >= .9*row['epochs'],
                            'last_val_mae': valid[-1]['val']['mae'],
                            'tail_val_mae_mean': statistics.mean(r['val']['mae'] for r in tail),
                            'tail_improvement_percent': 100*(tail[-1]['val']['mae']/tail[0]['val']['mae']-1) if tail[0]['val']['mae'] else None,
                            'train_seconds': sum(r.get('perf', {}).get('train_time_sec', 0) for r in entries),
                            'train_samples_limit': cfg['data']['training_selection']['limit']})
    common.write_json(suite/'comparison.json', {
        'note': 'All rates kept separate. Q11/Q01=front routing+balance contribution; Q11/Q10=backend conditional fusion. '
                'FD/SF=conditional vs global DENSE front weights, both zero front balance. Q11/SB=conditional vs global backend weights. '
                'Q11/FD also changes front sparsity AND balance; not pure sparsity. Uniform retains learned organization experts, not spatial average pooling. '
                'Coverage is one seed; replication is only at predeclared points. No significance or novel-idea guarantee. Select by VAL; TEST descriptive. '
                'Negative paired changes are better. No raw cross-dataset MAE averaging. Partial results are not final evidence.',
        'groups': grouped, 'paired': paired, 'paired_groups': paired_groups,
        'interaction': interactions, 'coverage_effects': effects})
    if confirmation:
        analysis = common.load(suite/'comparison.json')
        analysis['note'] = ('Backend confirmation: Q10=equal, SB=global learned weights, Q11=conditional dense weights; fixed frontend settings, jointly retrained. '
                            'Paired changes <0 are better. Q11/SB tests conditionality; SB/Q10 tests learning global weights; Q11/Q10 tests combined gain. '
                            'No 2x2 interaction. Rates never pooled. Two seeds are descriptive, not a significance test. '
                            'Reused rows retain original run_dir and reference_suite; they are not new independent runs. '
                            'Best epoch selected by VAL MAE, full TEST once. All groups must finish; tail convergence must be checked.')
        common.write_json(suite/'comparison.json', analysis)
    if any(j.get('stability_study') for j in all_jobs):
        analysis = common.load(suite/'comparison.json')
        analysis['note'] = STABILITY_NOTE
        analysis['coverage_effects'] = []
        common.write_json(suite/'comparison.json', analysis)
        common.write_json(suite/'mechanism_summary.json', [
            {**{k: row[k] for k in ('dataset','pattern','rate','seed','variant','status')},
             **{stage: {k:v for k,v in row.get(source, {}).items()
                        if k.startswith('backend_diag_') or k.startswith('completion_missing_')}
                for stage,source in [('best_val','best_val_metrics'),('test','test_metrics')]}}
            for row in rows])
    common.write_json(suite/'diagnostics.json', diagnostics)
    return rows


def is_backend_study(p):
    return p.get('protocol') == 'backend-routing-matched-v1'


def uses_train_selection(p):
    return is_routing_study(p) or is_target_study(p)


def training_epochs(p, dataset):
    return p['dataset_epochs'][dataset] if is_target_study(p) else p['epochs']


def is_expert_sweep(p):
    return p.get('protocol') == 'front-expert-count-full-v1'


def is_routing_study(p):
    return p.get('protocol') == 'front-routing-matched-v1'


def validate(p):
    if p.get('protocol') is not None and not is_topk(p):
        raise ValueError('Unknown comparison protocol')
    if type(p.get('save_best_checkpoint', True)) is not bool:
        raise ValueError('save_best_checkpoint must be true or false')
    if p.get('deadline_mode', 'strict') not in ('strict', 'advisory'):
        raise ValueError('deadline_mode must be strict or advisory')
    for k in ('train_windows', 'val_epoch', 'batch_size', 'cpu_threads') + (() if is_target_study(p) else ('epochs',)):
        if type(p[k]) is not int or p[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    for k, allowed in [('datasets', common.SPECS), ('patterns', ('fixed', 'random'))]:
        if not p[k] or len(set(p[k])) != len(p[k]) or any(v not in allowed for v in p[k]):
            raise ValueError(f'Invalid {k}')
    if not p['seeds'] or len(set(p['seeds'])) != len(p['seeds']) or any(type(s) is not int or s < 0 for s in p['seeds']):
        raise ValueError('Distinct nonnegative integer seeds required')
    if p['ablation_seed'] not in p['seeds']:
        raise ValueError('Ablations require a matched main seed')
    if ((not (is_coverage_study(p) or is_confirmation(p) or is_stability(p)) and p['rate'] not in (.2, .4, .6, .8))
            or not math.isfinite(p['timing_safety_factor']) or p['timing_safety_factor'] < 1):
        raise ValueError('Invalid rate or timing factor')
    if is_topk(p):
        if is_target_study(p):
            if ('epochs' in p or set(p.get('dataset_epochs', {})) != set(p['datasets'])
                    or any(type(n) is not int or n < p['val_epoch'] for n in p['dataset_epochs'].values())):
                raise ValueError('Declare positive dataset_epochs for each dataset, no ambiguous global epochs')
            if (set(p.get('expected_train_samples', {})) != set(p['datasets'])
                    or any(type(n) is not int or not 1 <= n <= p['train_windows'] for n in p['expected_train_samples'].values())):
                raise ValueError('Declare expected TRAIN counts per dataset')
            if is_stability(p):
                if (p['datasets'] != ['BikeNYC','TaxiBJ'] or p['patterns'] != ['random']
                        or p.get('rates') != [.4,.8] or 'rate' in p or p['seeds'] != [42,2026,3407]
                        or p['ablation_seed'] != 42 or p.get('points') != [
                            {'dataset':'BikeNYC','rate':.4}, {'dataset':'TaxiBJ','rate':.4}, {'dataset':'TaxiBJ','rate':.8}]
                        or p['variants'] != stability_variants() or p.get('reuse_enabled', False)
                        or p['expected_train_samples'] != {'BikeNYC':511, 'TaxiBJ':2491}):
                    raise ValueError('Keep all 36 matched U/D/H/L jobs and full TRAIN; no historical reuse')
                return
            if is_confirmation(p):
                expected = {v:{**coverage_variants()[v], 'role':'main'} for v in ('Q10','SB','Q11')}
                if (set(p['datasets']) != {'TaxiBJ','BikeNYC'} or p['patterns'] != ['random']
                        or p.get('rates') != [.4,.8] or 'rate' in p or p['seeds'] != [42,2026]
                        or p['ablation_seed'] != 42 or p.get('points') != [
                            {'dataset':'BikeNYC','rate':.4}, {'dataset':'TaxiBJ','rate':.4}, {'dataset':'TaxiBJ','rate':.8}]
                        or p['variants'] != expected or type(p.get('reuse_enabled',True)) is not bool
                        or not isinstance(p.get('reuse_suite'),str)):
                    raise ValueError('Keep all 18 matched Q10/SB/Q11 jobs at the three predeclared points')
                return
            if is_coverage_study(p):
                if (set(p['datasets']) != set(common.SPECS) or p['patterns'] != ['fixed', 'random']
                        or p.get('rates') != [.2, .4, .6, .8] or 'rate' in p
                        or p['seeds'] != [42, 2026] or p['ablation_seed'] != 42
                        or p.get('mechanism_points') != [{'pattern': 'random', 'rate': .4}]
                        or p['variants'] != coverage_variants()):
                    raise ValueError('Keep the complete 96-point matrix, 12 replication jobs and 9 predeclared mechanism jobs')
                return
            if is_backend_study(p):
                if set(p.get('dataset_seeds', {})) != set(p['datasets']):
                    raise ValueError('Declare dataset_seeds for every backend-study dataset')
                for seeds in p['dataset_seeds'].values():
                    if (not seeds or any(type(s) is not int or s not in p['seeds'] for s in seeds)
                            or len(set(seeds)) != len(seeds)):
                        raise ValueError('Dataset seeds must be distinct members of the global seeds list')
                if set().union(*(set(s) for s in p['dataset_seeds'].values())) != set(p['seeds']):
                    raise ValueError('Each global seed must be scheduled for a dataset')
                expected = {'BU': ('uniform', 2, 0.), 'BD': ('topk', 3, 0.),
                            'BK0': ('topk', 2, 0.), 'BK1': ('topk', 2, .01)}
                if set(p['variants']) != set(expected):
                    raise ValueError('Keep BU/BD/BK0/BK1 for every dataset and scheduled seed')
                for name, (mode, k, weight) in expected.items():
                    patch = {'model': {'dual_moe': {'completion_mode': mode, 'completion_top_k': k}},
                             'loss': {'dual_moe_completion_balance_weight': weight}}
                    if p['variants'][name]['role'] != 'main' or p['variants'][name]['patch'] != patch:
                        raise ValueError(f'{name}: change only declared backend routing/balance; keep frontend and training budget')
                return
            expected = {'Q00': ('uniform', 'uniform', 0., 0.),
                        'Q01': ('uniform', 'topk', 0., .01),
                        'Q10': ('topk', 'uniform', .001, 0.),
                        'Q11': ('topk', 'topk', .001, .01)}
            if set(p['variants']) != set(expected):
                raise ValueError('Keep all four matched target readout controls Q00/Q01/Q10/Q11')
            for name, (front, back, wf, wb) in expected.items():
                patch = {'model': {'dual_moe': {'aggregation_mode': front, 'completion_mode': back}},
                         'loss': {'dual_moe_aggregation_balance_weight': wf, 'dual_moe_completion_balance_weight': wb}}
                if p['variants'][name]['role'] != 'main' or p['variants'][name]['patch'] != patch:
                    raise ValueError(f'{name}: only declared routing/balance factors may differ, not data/epochs')
            return
        if is_routing_study(p):
            if (any(d not in ('TaxiBJ', 'BikeNYC') for d in p['datasets'])
                    or set(p['expected_train_samples']) != set(p['datasets'])
                    or any(type(n) is not int or not 1 <= n <= p['train_windows'] for n in p['expected_train_samples'].values())):
                raise ValueError('Declare expected selected TRAIN counts for TaxiBJ/BikeNYC; full VAL/TEST are fixed')
            expected = {
                'U8': ('main', 'uniform', 8, 'topk', 0., .01),
                'K2': ('main', 'topk', 2, 'topk', .01, .01),
                'K4': ('main', 'topk', 4, 'topk', .01, .01),
                'D8': ('main', 'topk', 8, 'topk', 0., .01),
                'K4_NB': ('ablation', 'topk', 4, 'topk', 0., .01),
                'D8_BU': ('ablation', 'topk', 8, 'uniform', 0., 0.),
            }
            if set(p['variants']) != set(expected):
                raise ValueError('Keep the four routing comparisons and two mechanism controls')
            for name, (role, mode, k, backend, wf, wb) in expected.items():
                patch = {'model': {'dual_moe': {'aggregation_experts': 8, 'aggregation_mode': mode,
                                                'aggregation_top_k': k, 'completion_mode': backend,
                                                'completion_top_k': 2}},
                         'loss': {'dual_moe_aggregation_balance_weight': wf,
                                  'dual_moe_completion_balance_weight': wb}}
                if p['variants'][name]['role'] != role or p['variants'][name]['patch'] != patch:
                    raise ValueError(f'{name}: keep matched E=8, declared routing and balance factors')
            return
        if p['datasets'] != ['TaxiBJ'] or p['train_windows'] != 2491:
            raise ValueError('Full Top-K comparison requires TaxiBJ and exactly 2491 training samples')
        if is_expert_sweep(p):
            counts, k = p['expert_counts'], p['aggregation_top_k']
            if (not counts or any(type(e) is not int or not 3 <= e <= 16 for e in counts)
                    or len(set(counts)) != len(counts) or 3 not in counts
                    or type(k) is not int or not 1 < k < min(counts)):
                raise ValueError('Use distinct expert counts 3..16 including reference 3, and fixed 1<K<min(E)')
            expected = {}
            for e in counts:
                for label, mode, weight in [('U', 'uniform', 0.), ('K', 'topk', .01)]:
                    expected[f'E{e}_{label}'] = {
                        'model': {'dual_moe': {'aggregation_experts': e, 'aggregation_top_k': k,
                                               'aggregation_mode': mode, 'completion_mode': 'topk'}},
                        'loss': {'dual_moe_aggregation_balance_weight': weight,
                                 'dual_moe_completion_balance_weight': .01}}
            if set(p['variants']) != set(expected):
                raise ValueError('Each expert count requires both uniform and Top-K controls')
            for name, patch in expected.items():
                spec = p['variants'][name]
                if spec['role'] != 'main' or spec['patch'] != patch:
                    raise ValueError(f'{name}: only front expert count/routing/balance may vary')
            return
        expected = {'T00': ('uniform', 'uniform', 0., 0.),
                    'T10': ('topk', 'uniform', .01, 0.),
                    'T01': ('uniform', 'topk', 0., .01),
                    'T11': ('topk', 'topk', .01, .01),
                    'T11_NB': ('topk', 'topk', 0., 0.)}
        if set(p['variants']) != set(expected):
            raise ValueError('Keep the predeclared five Top-K comparisons')
        for name, (front, back, front_w, back_w) in expected.items():
            spec = p['variants'][name]
            patch = {'model': {'dual_moe': {'aggregation_mode': front, 'completion_mode': back}},
                     'loss': {'dual_moe_aggregation_balance_weight': front_w,
                              'dual_moe_completion_balance_weight': back_w}}
            if spec['role'] != 'main' or spec['patch'] != patch:
                raise ValueError(f'{name}: change only the declared routing/balance factors')
        return
    if set(p['variants']) != {'B00', 'B01', 'N01', 'N00', 'NU', 'NG', 'NB2'}:
        raise ValueError('Keep the seven predeclared comparisons')
    for name, spec in p['variants'].items():
        if spec['role'] != ('main' if name in ('B00', 'B01', 'N01') else 'ablation'):
            raise ValueError('Do not silently change seed coverage')
        # Structural/loss hypotheses are configurable; training budgets cannot
        # vary by candidate. All candidates inherit the same preset and seed.
        if set(spec['patch']) - {'model', 'loss'}:
            raise ValueError('Variant patches may only change model/loss')


def identity(p):
    record = common.identity(p)
    paths = [Path(__file__), ROOT/'configs/presets/scale_completion.json']
    record['code'].update({str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    record['protocol'] = 'v23-scale-completion-matched-v1'
    if is_topk(p):
        for path in (ROOT/'configs/presets/dual_moe_topk.json', ROOT/'scripts/train_scale_completion.py'):
            record['code'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        record['protocol'] = p['protocol']
    if is_target_study(p):
        path = ROOT/'configs/presets/dual_moe_target.json'
        record['code'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if is_confirmation(p):
        record['reference_files'] = reference_files(p)
    return record


def full_topk_jobs(p, suite):
    import train_scale_completion as trainer
    seeds = [p['ablation_seed']] + [s for s in p['seeds'] if s != p['ablation_seed']]
    for seed in seeds:
        for dataset in p['datasets']:
            if is_backend_study(p) and seed not in p['dataset_seeds'][dataset]:
                continue
            for pattern in p['patterns']:
                for variant, spec in p['variants'].items():
                    if is_routing_study(p) and spec['role'] == 'ablation' and seed != p['ablation_seed']:
                        continue
                    cfg, paths = trainer.build_config(dataset, pattern, p['rate'], 'learned_regions',
                                                      seed, training_epochs(p, dataset), 'dual_moe_target' if is_target_study(p) else 'dual_moe_topk')
                    cfg = common.merge(cfg, spec['patch'])
                    cfg = common.merge(cfg, {
                        'output_dir': str(suite/'runs'),
                        'data': {'batch_size': p['batch_size'], 'num_workers': 0, 'drop_last': False},
                        'train': {'val_epoch': p['val_epoch'], 'early_stopping': {'enabled': False},
                                  **({'save_best_checkpoint': p['save_best_checkpoint']}
                                     if 'save_best_checkpoint' in p else {})}})
                    if uses_train_selection(p):
                        paths['train'] = suite/f'data/{dataset}/train.npz'
                        cfg['data']['mask']['train_csv'] = str(suite/f'data/{dataset}/{pattern}_train.csv')
                        cfg['data']['training_selection'] = {'strategy': 'evenly_spaced', 'limit': p['train_windows'],
                                                            'manifest': str(suite/f'data/{dataset}/selection.json')}
                    yield {'dataset': dataset, 'pattern': pattern, 'rate': p['rate'], 'seed': seed,
                           'variant': variant, 'name': f'{variant}_seed{seed}',
                           'key': f'{dataset}_{pattern}_{variant}_seed{seed}', 'cfg': cfg, 'paths': paths,
                           'expected_train_samples': p['expected_train_samples'][dataset] if uses_train_selection(p) else p['train_windows'],
                           **({'target_study': True, 'epochs': cfg['train']['epochs']} if is_target_study(p) else {}),
                           **({'backend_study': True} if is_backend_study(p) else {}),
                           **({'expert_count': cfg['model']['dual_moe']['aggregation_experts'],
                               'front_mode': cfg['model']['dual_moe']['aggregation_mode']}
                              if is_expert_sweep(p) else {}),
                           **({'routing_study': True, 'front_k': cfg['model']['dual_moe']['aggregation_top_k'],
                               'role': spec['role']} if is_routing_study(p) else {})}


def verify_full_data(p, all_jobs):
    """Inspect NPZ headers, not data copies: never silently truncate/pad TRAIN."""
    import numpy as np
    counts = {}
    for job in all_jobs:
        for split, path in job['paths'].items():
            if str(path) in counts:
                continue
            with zipfile.ZipFile(path) as archive:
                name = 'x_f_gt.npy' if 'x_f_gt.npy' in archive.namelist() else 'x_f.npy'
                with archive.open(name) as f:
                    version = np.lib.format.read_magic(f)
                    readers = {(1, 0): np.lib.format.read_array_header_1_0,
                               (2, 0): np.lib.format.read_array_header_2_0}
                    shape, fortran, dtype = readers[version](f)
            if len(shape) != 5 or shape[0] < 1 or fortran or dtype.hasobject:
                raise ValueError(f'Invalid NCTHW data: {path}: {shape}')
            if split == 'train' and shape[0] != p['train_windows']:
                raise ValueError(f'Expected exactly {p["train_windows"]} TRAIN samples, got {shape[0]}: {path}')
            counts[str(path)] = shape[0]
    return counts


def verify_routing_sources(p):
    """Check source headers before subset materialization, including dry runs."""
    counts = {}
    for dataset in p['datasets']:
        folder, prefix, _ = common.SPECS[dataset]
        paths = {s: ROOT/f'data/{folder}/{prefix}_{s}.npz' for s in ('train', 'val', 'test')}
        # Use the same numeric NCTHW checks without a hard TRAIN size assertion.
        raw = verify_full_data(p, [{'paths': {f'source_{s}': path for s, path in paths.items()}}])
        original = raw[str(paths['train'])]
        selected = min(original, p['train_windows'])
        if selected != p['expected_train_samples'][dataset]:
            raise ValueError(f'{dataset}: expected {p["expected_train_samples"][dataset]} selected TRAIN, actual {selected}')
        counts[dataset] = {'original_train': original, 'selected_train': selected,
                           'val': raw[str(paths['val'])], 'test': raw[str(paths['test'])]}
    return counts


def jobs(p, suite):
    """Main seed 42 first, ablations next, independent main seeds last."""
    if is_stability(p):
        yield from stability_jobs(p, suite)
        return
    if is_confirmation(p):
        yield from confirmation_jobs(p, suite)
        return
    if is_coverage_study(p):
        yield from coverage_jobs(p, suite)
        return
    if is_backend_study(p):
        # Finish both seeds of the primary dataset before the slower regression
        # dataset, without changing any job's initialization or budget.
        yield from sorted(full_topk_jobs(p, suite), key=lambda j: (
            p['datasets'].index(j['dataset']), p['seeds'].index(j['seed']),
            p['patterns'].index(j['pattern']), list(p['variants']).index(j['variant'])))
        return
    if is_topk(p):
        yield from full_topk_jobs(p, suite)
        return
    schedule = [(p['ablation_seed'], list(p['variants']))]
    schedule += [(s, ['B00', 'B01', 'N01']) for s in p['seeds'] if s != p['ablation_seed']]
    for seed, names in schedule:
        for dataset in p['datasets']:
            folder, prefix, base = common.SPECS[dataset]
            for pattern in p['patterns']:
                for variant in names:
                    cfg = common.merge(common.load(ROOT/f'configs/datasets/{base}.json'), common.load(ROOT/'configs/presets/scale_completion.json'))
                    cfg = common.merge(cfg, p['variants'][variant]['patch'])
                    cfg = common.merge(cfg, {
                        'seed': seed, 'device': 'cuda:0', 'output_dir': str(suite/'runs'),
                        'data': {'batch_size': p['batch_size'], 'num_workers': 0, 'drop_last': False,
                                 'mask': {'pattern': pattern, 'missing_rate': p['rate'],
                                          'train_csv': str(suite/f'data/{dataset}/{pattern}_train.csv'),
                                          **{f'{s}_csv': str(ROOT/f'data/{folder}/{pattern}_mask/{p["rate"]:g}/{s}.csv') for s in ('val', 'test')}},
                                 'training_selection': {'strategy': 'evenly_spaced', 'limit': p['train_windows'],
                                                        'manifest': str(suite/f'data/{dataset}/selection.json')}},
                        'train': {'epochs': p['epochs'], 'val_epoch': p['val_epoch'], 'early_stopping': {'enabled': False}},
                    })
                    paths = {'train': suite/f'data/{dataset}/train.npz',
                             **{s: ROOT/f'data/{folder}/{prefix}_{s}.npz' for s in ('val', 'test')}}
                    yield {'dataset': dataset, 'pattern': pattern, 'rate': p['rate'], 'seed': seed, 'variant': variant,
                           'name': f'{variant}_seed{seed}', 'key': f'{dataset}_{pattern}_{variant}_seed{seed}',
                           'cfg': cfg, 'paths': paths}


def completed(job):
    result = common.completed_run(job['cfg'], job['name'])
    if result is None:
        if job.get('reuse_job') is not None:
            ref = completed(job['reuse_job'])
            if ref:
                return {**ref, 'result_origin':'reused', 'reference_suite':job['reuse_suite']}
        return None
    import json
    try:
        if 'expected_train_samples' in job:
            train_log = (Path(result['run_dir'])/'logs/train.log').read_text()
            if f'  train_samples: {job["expected_train_samples"]}\n' not in train_log:
                return None
        entries = [json.loads(s) for s in (Path(result['run_dir'])/'logs/metrics.jsonl').read_text().splitlines()]
        records = [r for r in entries if 'epoch' in r]
        epochs, interval = job['cfg']['train']['epochs'], job['cfg']['train']['val_epoch']
        expected = [e for e in range(1, epochs+1) if e % interval == 0 or e == epochs]
        if [r['epoch'] for r in records if r.get('val') is not None] != expected:
            return None
        for r in records:
            for stage in ('train', 'val'):
                if r.get(stage) is not None and not all(math.isfinite(r[stage][k]) for k in ('mae', 'rmse', 'loss')):
                    return None
        best = next(r['val'] for r in records if r['epoch'] == result['best_epoch'])
        result['best_val_metrics'] = best
        result['test_metrics'] = next(r['metrics'] for r in entries if r.get('stage') == 'test')
        if job.get('confirmation_study'):
            result.update(result_origin='trained', reference_suite=None)
        return result
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


def summarize(all_jobs, suite):
    if any(j.get('coverage_study') or j.get('confirmation_study') or j.get('stability_study') for j in all_jobs) and len({j['rate'] for j in all_jobs}) > 1:
        return summarize_coverage(all_jobs, suite)
    rows = []
    for job in all_jobs:
        result = completed(job)
        rows.append({k: job[k] for k in ('dataset', 'pattern', 'rate', 'seed', 'variant')} |
                    {'status': 'complete' if result else 'incomplete', **(result or {}),
                     **{k: job[k] for k in ('expert_count', 'front_mode', 'front_k', 'role', 'epochs') if k in job}})
    common.write_json(suite/'summary.json', rows)
    fields = ['dataset', 'pattern', 'rate', 'seed', 'variant', 'status', 'val_mae', 'test_mae', 'test_rmse', 'best_epoch', 'run_dir']
    expert_counts = sorted({j['expert_count'] for j in all_jobs if 'expert_count' in j})
    if expert_counts:
        fields += ['expert_count', 'front_mode']
    routing_study = any(j.get('routing_study') for j in all_jobs)
    target_study = any(j.get('target_study') for j in all_jobs)
    backend_study = any(j.get('backend_study') for j in all_jobs)
    if target_study:
        fields += ['epochs']
    if any(j.get('confirmation_study') for j in all_jobs):
        fields += ['result_origin', 'reference_suite']
    if routing_study:
        fields += ['front_k', 'role']
    with (suite/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    groups, pairs = [], []
    for dataset, pattern, variant in sorted({(r['dataset'], r['pattern'], r['variant']) for r in rows}):
        selected = [r for r in rows if (r['dataset'], r['pattern'], r['variant']) == (dataset, pattern, variant)]
        ok = [r for r in selected if r['status'] == 'complete']
        group = {'dataset': dataset, 'pattern': pattern, 'variant': variant, 'complete': len(ok), 'expected': len(selected)}
        if ok:
            for k in ('val_mae', 'test_mae', 'test_rmse'):
                group[k+'_mean'] = statistics.mean(r[k] for r in ok)
                group[k+'_std'] = statistics.stdev(r[k] for r in ok) if len(ok) > 1 else None
        groups.append(group)
    comparisons = [('B01', 'B00'), ('N01', 'B01'), ('N01', 'B00'), ('N00', 'N01'), ('NU', 'N01'), ('NG', 'N01'), ('NB2', 'N01')]
    if {j['variant'] for j in all_jobs} == {'T00', 'T10', 'T01', 'T11', 'T11_NB'}:
        comparisons = [('T10', 'T00'), ('T01', 'T00'), ('T11', 'T10'),
                       ('T11', 'T01'), ('T11', 'T00'), ('T11', 'T11_NB')]
    if expert_counts:
        comparisons = [(f'E{e}_K', f'E{e}_U') for e in expert_counts]
        comparisons += [(f'E{e}_{m}', f'E3_{m}') for e in expert_counts if e != 3 for m in ('U', 'K')]
    if routing_study:
        comparisons = [('K2', 'U8'), ('K4', 'U8'), ('D8', 'U8'), ('K4', 'K2'),
                       ('D8', 'K2'), ('D8', 'K4'), ('K4', 'K4_NB'), ('D8', 'D8_BU')]
    if target_study:
        comparisons = [('Q11', 'Q01'), ('Q11', 'Q10'), ('Q01', 'Q00'), ('Q10', 'Q00'), ('Q11', 'Q00')]
    if any(j.get('coverage_study') for j in all_jobs):
        comparisons += [('FD', 'SF'), ('Q11', 'SB'), ('Q11', 'FD'), ('SF', 'Q01')]
    if backend_study:
        comparisons = [('BD', 'BU'), ('BK0', 'BD'), ('BK1', 'BK0'),
                       ('BK1', 'BU'), ('BK0', 'BU'), ('BK1', 'BD')]
    if any(j.get('confirmation_study') for j in all_jobs):
        comparisons = [('Q11','SB'), ('SB','Q10'), ('Q11','Q10')]
    if any(j.get('stability_study') for j in all_jobs):
        comparisons = [('D','U'), ('H','D'), ('H','U'), ('L','H'), ('L','D'), ('L','U')]
    lookup = {(r['dataset'], r['pattern'], r['seed'], r['variant']): r for r in rows if r['status'] == 'complete'}
    for row in rows:
        if row['status'] != 'complete': continue
        for candidate, control in comparisons:
            if row['variant'] != candidate: continue
            ref = lookup.get((row['dataset'], row['pattern'], row['seed'], control))
            if ref:
                pairs.append({k: row[k] for k in ('dataset', 'pattern', 'seed')} |
                             {'candidate': candidate, 'control': control,
                              **{k+'_change_percent': 100*(row[k]/ref[k]-1) if ref[k] else None for k in ('val_mae', 'test_mae', 'test_rmse')}})
    interaction = []
    for (dataset, pattern, seed, variant), row in lookup.items():
        if variant != ('Q11' if target_study else 'T11'): continue
        names = ('Q00', 'Q10', 'Q01') if target_study else ('T00', 'T10', 'T01')
        refs = [lookup.get((dataset, pattern, seed, name)) for name in names]
        if all(refs):
            a, b, c = refs
            interaction.append({'dataset': dataset, 'pattern': pattern, 'seed': seed,
                                **{k: row[k]-b[k]-c[k]+a[k] for k in ('val_mae', 'test_mae', 'test_rmse')}})
    front_gain = []
    for row in rows:
        if row.get('front_mode') != 'topk' or row['status'] != 'complete': continue
        key = (row['dataset'], row['pattern'], row['seed'])
        uniform = lookup.get((*key, f'E{row["expert_count"]}_U'))
        reference_k, reference_u = lookup.get((*key, 'E3_K')), lookup.get((*key, 'E3_U'))
        if uniform:
            gain = {k: uniform[k]-row[k] for k in ('val_mae', 'test_mae', 'test_rmse')}
            front_gain.append({**dict(zip(('dataset', 'pattern', 'seed'), key)), 'expert_count': row['expert_count'],
                               'uniform_minus_topk': gain,
                               'gain_change_vs_E3': {k: v-(reference_u[k]-reference_k[k]) for k, v in gain.items()}
                               if reference_k and reference_u else None})
    paired_groups = []
    if routing_study or target_study:
        for dataset, pattern in sorted({(r['dataset'], r['pattern']) for r in rows}):
            for candidate, control in comparisons:
                planned = [{r['seed'] for r in rows if (r['dataset'], r['pattern'], r['variant']) == (dataset, pattern, variant)} for variant in (candidate, control)]
                expected = sorted(planned[0] & planned[1])
                subset = [r for r in pairs if (r['dataset'], r['pattern'], r['candidate'], r['control']) == (dataset, pattern, candidate, control)]
                group = {'dataset': dataset, 'pattern': pattern, 'candidate': candidate, 'control': control,
                         'expected_seeds': expected, 'complete_seeds': sorted(r['seed'] for r in subset),
                         'status': 'complete' if len(subset) == len(expected) and expected else 'incomplete'}
                if subset:
                    for key in ('val_mae', 'test_mae', 'test_rmse'):
                        values = [r[key+'_change_percent'] for r in subset]
                        if all(v is not None for v in values):
                            group[key+'_change_percent_mean'] = statistics.mean(values)
                            group[key+'_change_percent_std'] = statistics.stdev(values) if len(values)>1 else None
                            group[key+'_wins'] = sum(v<0 for v in values)
                paired_groups.append(group)
    note = 'Paired percent change: negative is better. Front gain=uniform minus Top-K: positive is better; gain_change_vs_E3 positive means a larger routing advantage. Partial groups are not final results. Test is descriptive, not for tuning. Interaction = T11-T10-T01+T00; negative means super-additive error reduction, not statistical significance. Uniform means equal expert fusion, not fixed spatial average pooling. Expert-count sweeps fix K/backend but vary capacity and K/E; they do not isolate routing from its balance regularizer. In routing study D8 is full learned softmax (Top-8 of 8), with front balance disabled because hard-count balance would be constant; dense hard loads being uniform is NOT evidence of non-collapsed gate weights. Two seeds are exploratory, not a significance claim.'
    if target_study:
        note = ('Target readout 2x2: paired percent change <0 is better; interaction=Q11-Q10-Q01+Q00. '
                'Each dataset has its own epoch budget, equal across its four controls. '
                'Uniform retains learned aggregation experts and disables that side balance loss; this tests routing+balance, not pure routing or fixed spatial average pooling. '
                'No test-based tuning or raw cross-dataset MAE averaging. A single seed is exploratory, not evidence of significance; incomplete pairs are not final conclusions.')
    if backend_study:
        note = ('Fixed frontend architecture/settings, jointly retrained end-to-end, NOT frozen weights. '
                'BU=uniform, BD=full softmax (Top-3 of 3, zero balance), BK0=Top-2 without backend balance, BK1=Top-2 with .01 balance. '
                'BD vs BU tests learned fusion; BK0 vs BD tests hard sparsity; BK1 vs BK0 tests balance. '
                'BD/BK0/BK1 share router initialization under each seed; BD hard-load uniformity does not prove probability balance. '
                'Budgets and train samples match within each dataset; dataset seed counts may differ. '
                'Negative paired change is better. Select by VAL, test is descriptive. No 2x2 interaction applies. '
                'Do not compare reduced TaxiBJ training data directly to prior 2048-window scores. No significance claim from one/two seeds.')
    if any(j.get('confirmation_study') for j in all_jobs):
        note = ('Backend confirmation: Q10 equal, SB global learned, Q11 conditional dense; same jointly trained frontend. '
                'Negative paired change is better; Q11/SB conditionality, SB/Q10 global weighting, Q11/Q10 combined gain. '
                'Two seeds are descriptive, VAL selects best epoch and TEST is descriptive. '
                'Reused rows point to original runs, not independent new results. No 2x2 interaction applies.')
    if any(j.get('stability_study') for j in all_jobs):
        note = STABILITY_NOTE
    common.write_json(suite/'comparison.json', {'note': note, 'groups': groups, 'paired': pairs, 'interaction': interaction, 'front_routing_gain': front_gain, 'paired_groups': paired_groups})
    return rows


def estimate_run_seconds(cfg, counts, train_s, eval_s, safety_factor):
    """Use the resolved per-job budget, including the final off-calendar VAL."""
    epochs, interval = cfg['train']['epochs'], cfg['train']['val_epoch']
    return (epochs*counts[0]*train_s + (math.ceil(epochs/interval)*counts[1]+counts[2])*eval_s)*safety_factor+30


def calibrate(p, suite, all_jobs):
    import torch
    from torch.utils.data import Subset
    sys.path.insert(0, str(ROOT/'src'))
    from stmoe_imputer.data import build_datasets, build_test_dataset, build_loader
    from stmoe_imputer.engine import train_one_epoch, evaluate, build_optimizer
    from stmoe_imputer.models import DualBranchSTImputer
    if not torch.cuda.is_available(): raise RuntimeError('Activate difftdi with CUDA available')
    torch.set_num_threads(p['cpu_threads'])
    timings = {}
    for job in all_jobs:
        key = f'{job["dataset"]}/{job["variant"]}'
        if key in timings: continue
        cfg, paths = job['cfg'], job['paths']
        train, val = build_datasets(cfg, str(paths['train']), str(paths['val']))
        test = build_test_dataset(cfg, str(paths['test']))
        counts = [math.ceil(len(d)/p['batch_size']) for d in (train, val, test)]
        loaders = [build_loader(Subset(d, range(min(len(d), 8*p['batch_size']))), cfg, False) for d in (train, val)]
        model = DualBranchSTImputer.from_config(cfg).cuda(); optimizer = build_optimizer(model, cfg)
        def timed(training):
            torch.cuda.synchronize(); start = time.perf_counter()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                if training: train_one_epoch(model, loaders[0], optimizer, torch.device('cuda'), cfg, 1)
                else: evaluate(model, loaders[1], torch.device('cuda'), cfg)
            torch.cuda.synchronize()
            return (time.perf_counter()-start)/len(loaders[0 if training else 1])
        timed(True); timed(False)
        train_s, eval_s = timed(True), timed(False)
        epochs, interval = cfg['train']['epochs'], cfg['train']['val_epoch']
        seconds = estimate_run_seconds(cfg, counts, train_s, eval_s, p['timing_safety_factor'])
        timings[key] = {'seconds_per_run': seconds, 'train_batch_seconds': train_s, 'eval_batch_seconds': eval_s,
                        'epochs': epochs, 'val_epoch': interval,
                        'batches_train_val_test': counts, 'trainable_params': sum(x.numel() for x in model.parameters() if x.requires_grad)}
        print(f'[timing] {key}: {seconds/60:.1f} min/run (safety included)', flush=True)
        del model, optimizer, train, val, test, loaders
        torch.cuda.empty_cache()
    common.write_json(suite/'timing.json', {'gpu': torch.cuda.get_device_name(), 'measured_at': datetime.now(TZ).isoformat(), 'timings': timings})
    return timings


def launch(job, suite):
    path = suite/'configs'/f'{job["key"]}.json'; common.write_json(path, job['cfg'])
    command = [sys.executable, '-u', str(ROOT/'scripts/train.py'), '-c', str(path), '--name', f'ablation_{job["name"]}', '--no_plot', '--quiet']
    for split, source in job['paths'].items(): command.extend([f'--{split}_npz', str(source)])
    raw = suite/'logs'/f'{job["key"]}.log'; raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                if line.strip() and not line.lstrip().startswith(('train epoch ', 'val epoch ', 'test best epoch ')):
                    print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
        finally:
            process.stdout.close()
    if code or not completed(job):
        raise RuntimeError(f'Incomplete/failed {job["key"]}, exit={code}; inspect {raw}. Rerun skips only completed jobs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/presets/scale_completion_experiments.json')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--no-reuse', action='store_true', help='Backend confirmation: rerun all 18 jobs, do not import eligible historical results')
    parser.add_argument('--budget', choices=('full', 'extended'), help='Coverage study: full TRAIN deadline budget, or longer full TRAIN without deadline')
    parser.add_argument('--deadline', help='Asia/Shanghai YYYY-MM-DD HH:MM; admission estimate, not a guaranteed hard stop')
    args = parser.parse_args()
    if not args.gpu.isdigit(): parser.error('One GPU index required')
    p = resolve_policy(common.load(common.resolve(args.config)), args.budget)
    if args.no_reuse:
        if not is_confirmation(p): parser.error('--no-reuse is only for backend confirmation')
        p['reuse_enabled'] = False
    validate(p)
    deadline_text = args.deadline or p.get('deadline')
    deadline = datetime.strptime(deadline_text, '%Y-%m-%d %H:%M').replace(tzinfo=TZ) if deadline_text else None
    strict_deadline = p.get('deadline_mode', 'strict') == 'strict'
    record = identity(p); suite = common.resolve(p['output_dir'])/common.digest(record)[:16]
    all_jobs = list(jobs(p, suite))
    reuse_audit = bind_reuse(p, all_jobs)
    if is_confirmation(p):
        count = sum(j.get('reuse_job') is not None for j in all_jobs)
        print(f'[reuse] {count}/{len(all_jobs)} eligible historical runs; {len(all_jobs)-count} require local training unless already complete', flush=True)
        for item in reuse_audit:
            if item['status'] == 'disabled': print(f'[reuse disabled] {item["reason"]}', flush=True)
    if uses_train_selection(p):
        counts = verify_routing_sources(p)
        print(f'[data] matched TRAIN selection, FULL val/test: {counts}', flush=True)
        if is_coverage_study(p) or is_confirmation(p) or is_stability(p):
            if any(c['original_train'] != c['selected_train'] for c in counts.values()):
                raise ValueError('Coverage study requires ALL TRAIN samples; do not silently cap the selected full-data budget')
            print(f'[data] ALL original TRAIN samples retained; budget={p.get("selected_budget", p["protocol"])}', flush=True)
    elif is_topk(p):
        counts = verify_full_data(p, all_jobs)
        print(f'[data] full source NPZs, sample counts: {counts}', flush=True)
    train_budget = f'FULL exactly {p["train_windows"]}' if is_topk(p) and not uses_train_selection(p) else f'<= {p["train_windows"]}'
    epoch_budget = p['dataset_epochs'] if is_target_study(p) else p['epochs']
    print(f'[suite] {suite}\n[budget] {len(all_jobs)} jobs; {epoch_budget} epochs, val every {p["val_epoch"]}, train {train_budget}; FULL val/test', flush=True)
    if args.dry_run:
        for job in all_jobs:
            origin = 'REUSE' if job.get('reuse_job') else 'RUN'
            print(f'{origin} {job["key"]}' if is_confirmation(p) else job['key'])
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu; os.environ['PYTHONUNBUFFERED'] = '1'
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = str(p['cpu_threads'])
    import fcntl
    suite.mkdir(parents=True, exist_ok=True)
    with (suite/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common.write_json(suite/'protocol.json', record)
        if is_confirmation(p):
            common.write_json(suite/'reuse_manifest.json', reuse_audit)
        if is_coverage_study(p) or is_confirmation(p) or is_stability(p):
            common.write_json(suite/'plan.json', [{k: j[k] for k in (
                'key', 'dataset', 'pattern', 'rate', 'seed', 'variant', 'stage', 'epochs', 'expected_train_samples')}
                for j in all_jobs])
        if args.summary_only:
            rows = summarize(all_jobs, suite)
            print(f'[summary] {sum(r["status"] == "complete" for r in rows)}/{len(rows)} complete: {suite/"summary.csv"}')
            return
        if is_topk(p) and not uses_train_selection(p):
            common.write_json(suite/'data_manifest.json', {'selection': 'full original NPZ, no subset', 'counts': counts})
        else:
            common.prepare(p, suite)
        remaining = [j for j in all_jobs if completed(j) is None]
        if not remaining and not args.calibrate:
            for i, job in enumerate(all_jobs, 1):
                print(f'[{i}/{len(all_jobs)}] SKIP complete {job["key"]}', flush=True)
            summarize(all_jobs, suite)
            print(f'[done] {suite/"summary.csv"}\n[paired] {suite/"comparison.json"}', flush=True)
            return
        # Every launch uses fresh timings when a deadline matters. Calibration
        # weights are discarded, never used as experiment initialization.
        timing = calibrate(p, suite, all_jobs) if args.calibrate or deadline or not (suite/'timing.json').exists() else common.load(suite/'timing.json')['timings']
        seconds = sum(timing[f'{j["dataset"]}/{j["variant"]}']['seconds_per_run'] for j in remaining)
        eta = datetime.now(TZ)+timedelta(seconds=seconds)
        print(f'[ETA] {len(remaining)} remaining, {seconds/3600:.2f} h, {eta:%Y-%m-%d %H:%M %Z}; estimate, NOT guaranteed', flush=True)
        if args.calibrate: return
        if deadline and eta > deadline:
            if strict_deadline:
                raise SystemExit('Estimated queue exceeds deadline; no formal jobs started. Change the common JSON budget for ALL groups, or use a later deadline.')
            print(f'[WARN] ETA exceeds advisory target {deadline:%Y-%m-%d %H:%M %Z}; continue ALL planned epochs, no budget truncation.', flush=True)
        try:
            for i, job in enumerate(all_jobs, 1):
                if is_topk(p) and identity(p) != record:
                    raise RuntimeError('Code/data/preset changed during this comparison; stop rather than mix implementations. Rerun creates a new fingerprinted suite.')
                done = completed(job)
                if done:
                    label = 'REUSE verified' if isinstance(done, dict) and done.get('result_origin') == 'reused' else 'SKIP complete'
                    print(f'[{i}/{len(all_jobs)}] {label} {job["key"]}', flush=True); continue
                if strict_deadline and deadline and datetime.now(TZ)+timedelta(seconds=timing[f'{job["dataset"]}/{job["variant"]}']['seconds_per_run']) > deadline:
                    raise RuntimeError('Deadline admission stopped new jobs. No partial training is counted as completed.')
                print(f'[{i}/{len(all_jobs)}] RUN {job["key"]}', flush=True)
                launch(job, suite)
                summarize(all_jobs, suite)
        finally:
            summarize(all_jobs, suite)
        print(f'[done] {suite/"summary.csv"}\n[paired] {suite/"comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
