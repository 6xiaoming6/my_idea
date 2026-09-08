#!/usr/bin/env python3
"""V22 follow-up: seed replication, fixed-geometry controls, validation-only routing audit.

No edits to the historical backbone/launcher: original fingerprints remain valid.
Control workers explicitly register a separate builder before entering the existing
trainer. This file's hash is included in every control's effective configuration.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile

from train import ROOT, DATASETS, build_config, load_suite, resolve
from run_experiments import completed


def load_policy(path):
    policy = json.loads(resolve(path).read_text())
    if policy['epochs'] < 1 or len(set(policy['seeds'])) != len(policy['seeds']):
        raise ValueError('Positive epoch budget and distinct seeds are required')
    return policy, load_suite(policy['suite'], policy['profile'])


def control_config(suite, variant, dataset, pattern, rate, seed, epochs, world_size):
    if variant not in {'uniform_fixed_shared', 'uniform_fixed_independent'}:
        raise ValueError(f'Unknown control: {variant}')
    suite = copy.deepcopy(suite)
    suite['variants'][variant] = {'model': {'v22': {
        'mode': 'uniform', 'geometry_control': variant,
        'verification_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }}}
    return build_config(suite, variant, dataset, pattern, rate, seed, epochs, world_size=world_size)


def install_control_builder():
    """Register only in control workers; never alter the normal V22 process."""
    import torch
    from torch import nn
    from stmoe_imputer.models.registry import MODEL_REGISTRY
    from stmoe_imputer.models.v_single.v22_coarsening_moe import V22CoarseningMoE, CoarseningScale

    class IndependentFixedScale(CoarseningScale):
        def __init__(self, original):
            # Preserve initialized common components and their state-dict names.
            nn.Module.__init__(self)
            self.mode, self.active = original.mode, original.active
            self.coarseners = original.coarseners
            self.coarse_input, self.coarse_block = original.coarse_input, original.coarse_block
            self.empty_token, self.router = original.empty_token, original.router
            self.branch_inputs = nn.ModuleList([copy.deepcopy(self.coarse_input) for _ in range(3)])
            self.branch_blocks = nn.ModuleList([copy.deepcopy(self.coarse_block) for _ in range(3)])
            for module in [*self.branch_inputs, *self.branch_blocks]:
                for child in module.modules():
                    if hasattr(child, 'reset_parameters'):
                        child.reset_parameters()
            self.coarse_input.requires_grad_(False)
            self.coarse_block.requires_grad_(False)

        def forward(self, features, evidence, mask):
            b, d, t, h, w = features.shape
            restore = lambda z, hh, ww: z.reshape(b, t, hh, ww, -1).permute(0, 4, 1, 2, 3).contiguous()
            # All three use exactly the same fixed partition and observation statistics.
            mean, stats, assignment, (hc, wc), penalty, diag = self.coarseners[0](features, evidence, mask)
            present = stats[..., 1:2]
            mean = mean * present + self.empty_token * (1 - present)
            packed = restore(torch.cat((mean, stats), -1), hc, wc)
            views = []
            for stem, block in zip(self.branch_inputs, self.branch_blocks):
                coarse = block(stem(packed))
                flat = coarse.permute(0, 2, 3, 4, 1).reshape(b * t, hc * wc, d).float()
                views.append(restore(self.coarseners[0].prolong(flat, assignment), h, w))
            diagnostics = {f'e{i}_{k}': v for i in range(3) for k, v in diag.items()}
            diagnostics.update({f'route_e{i}': features.new_tensor(1 / 3) for i in range(3)})
            diagnostics['route_entropy'] = features.new_tensor(math.log(3))
            return (sum(views) / 3).to(features.dtype), penalty, features.new_zeros(()), diagnostics

    def builder(cfg):
        control = cfg['model']['v22'].get('geometry_control')
        if control not in {'uniform_fixed_shared', 'uniform_fixed_independent'}:
            raise ValueError('Control worker requires explicit geometry_control')
        model = V22CoarseningMoE.from_config(cfg)
        for scale in model.scales:
            # Keep parameter initialization order identical; inactive assignment
            # projections are frozen so DDP sees no unused trainable parameters.
            for coarsener in scale.coarseners:
                coarsener.radius = 0
                coarsener.requires_grad_(False)
        if control == 'uniform_fixed_independent':
            model.scales = nn.ModuleList([IndependentFixedScale(scale) for scale in model.scales])
        return model

    MODEL_REGISTRY['v22_coarsening_moe'] = builder


def worker():
    # torchrun re-enters this file on each rank. Nothing is registered by the
    # scheduling parent, and the historical train/val/best/test loop is reused.
    sys.path.insert(0, str(ROOT / 'src'))
    install_control_builder()
    import train_ddp
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    train_ddp.main()


def jobs(policy, suite, devices, controls=False):
    variants = policy['controls'] if controls else policy['variants']
    seeds = policy['control_seeds'] if controls else policy['seeds']
    for dataset, pattern, rate in suite['default_points']:
        for variant in variants:
            for seed in seeds:
                builder = control_config if controls else build_config
                cfg, paths = builder(suite, variant, dataset, pattern, rate, seed,
                                     policy['epochs'], world_size=len(devices))
                yield variant, dataset, pattern, str(rate), seed, cfg, paths


def launch_control(job, args, interpreter):
    variant, _, _, _, _, cfg, paths = job
    with tempfile.TemporaryDirectory(prefix='v22_control_') as directory:
        path = Path(directory) / 'config.json'
        path.write_text(json.dumps(cfg, indent=2))
        command = interpreter + ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   f'--nproc_per_node={len(args.gpus)}', '--max_restarts=0',
                   str(Path(__file__).resolve()), '--worker', '-c', str(path),
                   '--name', f'ablation_v22_{variant}', '--no_plot', '--quiet']
        command += [a for split, p in paths.items() for a in (f'--{split}_npz', p)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(args.gpus), PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = str(args.cpu_threads)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None} if values else None


def summarize(policy, suite, devices):
    rows, audits = [], []
    for control in (False, True):
        for variant, dataset, pattern, rate, seed, cfg, _ in jobs(policy, suite, devices, control):
            old = completed(cfg, variant)
            rows.append(dict(variant=variant, dataset=dataset, pattern=pattern, rate=rate, seed=seed,
                             status='complete' if old else 'missing_or_incompatible', **(old or {})))
            if old and variant in policy['diagnostic_variants']:
                for path in sorted((Path(old['run']) / 'logs').glob('route_validation_*.json'), reverse=True):
                    try:
                        audit = json.loads(path.read_text())
                        if audit.get('reference_matches') and audit.get('checkpoint_epoch') == old['best_epoch']:
                            audits.append(dict(dataset=dataset, pattern=pattern, seed=seed,
                                               path=str(path), conditions=audit['conditions'],
                                               scale_alignment=audit['scale_alignment']))
                            break
                    except (OSError, ValueError, KeyError):
                        continue
    groups, paired = [], []
    for dataset, pattern, rate in suite['default_points']:
        for variant in policy['variants'] + policy['controls']:
            subset = [r for r in rows if (r['dataset'], r['pattern'], r['rate'], r['variant']) ==
                      (dataset, pattern, str(rate), variant) and r['status'] == 'complete']
            groups.append(dict(dataset=dataset, pattern=pattern, rate=rate, variant=variant,
                               seeds=[r['seed'] for r in subset],
                               **{k: stats([r[k] for r in subset]) for k in ('val_mae', 'test_mae', 'test_rmse')}))
        for left, right in [('uniform', 'fixed_stats'), ('moe', 'uniform'),
                            ('uniform', 'uniform_fixed_shared'), ('uniform', 'uniform_fixed_independent')]:
            lookup = {(r['variant'], r['seed']): r for r in rows if r['dataset'] == dataset and
                      r['pattern'] == pattern and r['rate'] == str(rate) and r['status'] == 'complete'}
            seeds = sorted(s for v, s in lookup if v == left and (right, s) in lookup)
            paired.append(dict(dataset=dataset, pattern=pattern, rate=rate, comparison=f'{left} minus {right}',
                               seeds=seeds, negative_is_better=True,
                               **{k: stats([lookup[left, s][k] - lookup[right, s][k] for s in seeds])
                                  for k in ('val_mae', 'test_mae', 'test_rmse')}))
    result = {'runs': rows, 'groups': groups, 'paired_differences': paired, 'validation_routing_audits': audits,
              'note': 'Sample SD; paired seeds only. No significance claim with n=1/3; controls are not parameter matched.'}
    output = ROOT / 'outputs/v22'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'verification_summary_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['V22 verification summary', 'MAE/RMSE: original units, missing entries only.',
             'SD is sample SD across seeds; n=1 has no estimated SD.',
             'dataset pattern variant seeds test_mae(mean +/- sd) test_rmse(mean +/- sd)']
    for group in groups:
        def format_stat(value):
            if value is None:
                return 'MISSING'
            sd = 'NA' if value['sample_std'] is None else f"{value['sample_std']:.6f}"
            return f"{value['mean']:.6f}+/-{sd}"
        lines.append(f"{group['dataset']} {group['pattern']} {group['variant']} {group['seeds']} "
                     f"{format_stat(group['test_mae'])} {format_stat(group['test_rmse'])}")
    lines.append(f'Complete runs: {sum(r["status"] == "complete" for r in rows)}/{len(rows)}; routing audits: {len(audits)}')
    path.with_suffix('.log').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Saved {path}')


def diagnose(run, policy, device_name, threads):
    """Validation ONLY. Forced-branch errors are interventions, not expert heads."""
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from torch.utils.data import DataLoader
    from stmoe_imputer.data.npz_dataset import FlowNPZDataset
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.models.v_single.v22_coarsening_moe import CoarseningScale
    from stmoe_imputer.metrics import MaskedMetricAccumulator
    from stmoe_imputer.utils.checkpoint import load_checkpoint
    from stmoe_imputer.utils.device import move_batch_to_device

    torch.set_num_threads(threads)
    run = Path(run)
    cfg = json.loads((run / 'config.json').read_text())
    if cfg['model']['v22']['mode'] not in ('moe', 'moe_no_stats'):
        raise ValueError('Routing audit requires a learned router checkpoint')
    device = torch.device(device_name)
    model = DualBranchSTImputer.from_config(cfg).to(device).eval()
    checkpoint = load_checkpoint(run / 'checkpoints/best.pt', model, map_location=device)
    scales = [m for m in model.modules() if isinstance(m, CoarseningScale)]
    _, folder, prefix = DATASETS[cfg['data']['dataset_name']]
    sc, mask = cfg['data']['scales'], cfg['data']['mask']
    dataset = FlowNPZDataset(ROOT / folder / f'{prefix}_val.npz', mask_cfg=mask,
                fine_to_mid=sc['fine_to_mid'], fine_to_coarse=sc['fine_to_coarse'],
                pooling_mode=sc.get('pooling_mode', 'avg'), pyramid_mode=sc.get('pyramid_mode', 'legacy'),
                seed=cfg['seed'] + 20000, mask_csv=resolve(mask['val_csv']))
    loader = DataLoader(dataset, batch_size=policy['diagnostic_batch_size'], shuffle=False, num_workers=0)
    conditions = ['learned', 'uniform', 'shuffled'] + [f's{i}_e{e}' for i in range(len(scales)) for e in range(3)]
    acc = {name: MaskedMetricAccumulator() for name in conditions}
    alignment = [dict(count=0, agreement=0., regret=0., weighted_regret=0., uniform_regret=0., oracle_error=0.,
                      entropy=0., usage=[0., 0., 0.]) for _ in scales]
    mode, gates, shuffled = ['learned'], {}, {}

    def hook(index):
        def intervene(module, inputs, logits):
            if mode[0] == 'learned':
                gates[index] = logits.detach().softmax(1)
                return logits
            if mode[0] == 'uniform':
                return torch.zeros_like(logits)
            if mode[0] == 'shuffled':
                return shuffled[index].clamp_min(1e-30).log()
            if mode[0].startswith(f's{index}_e'):
                out = torch.full_like(logits, -80.)
                out[:, int(mode[0][-1])] = 0.
                return out
            return logits
        return intervene

    handles = [scale.router.register_forward_hook(hook(i)) for i, scale in enumerate(scales)]
    try:
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                batch = move_batch_to_device(batch, device)
                errors = {}
                for condition in conditions:
                    mode[0] = condition
                    # Targets are deliberately not passed to the inference model.
                    inputs = {k: v for k, v in batch.items() if not k.endswith('_gt')}
                    pred = model(inputs)['x_hat_final']
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError(f'Non-finite {condition} prediction')
                    acc[condition].update(pred, batch['x_f_gt'], batch['m_f'])
                    if condition == 'learned':
                        for i, gate in gates.items():
                            # Deterministic per-example permutation of T/H/W;
                            # preserve expert usage but destroy position matching.
                            flat = gate.flatten(2)
                            generator = torch.Generator().manual_seed(policy['diagnostic_shuffle_seed'] + step * 100 + i)
                            perm = torch.randperm(flat.shape[-1], generator=generator).to(device)
                            shuffled[i] = flat[:, :, perm].reshape_as(gate)
                    elif condition.startswith('s') and condition != 'shuffled':
                        errors[condition] = (pred - batch['x_f_gt']).abs().mean(1)
                missing = batch['m_f'][:, 0] < .5
                for i, gate in gates.items():
                    err = torch.stack([errors[f's{i}_e{e}'] for e in range(3)], 1)
                    oracle, best = err.min(1)
                    selected = gate.argmax(1)
                    chosen = err.gather(1, selected[:, None]).squeeze(1)
                    a = alignment[i]
                    a['count'] += int(missing.sum())
                    # Treat tied minima as correct, not arbitrary argmin labels.
                    a['agreement'] += float(((chosen - oracle).abs() <= 1e-6)[missing].sum())
                    a['regret'] += float((chosen - oracle)[missing].sum())
                    a['weighted_regret'] += float(((gate * err).sum(1) - oracle)[missing].sum())
                    a['uniform_regret'] += float((err.mean(1) - oracle)[missing].sum())
                    a['oracle_error'] += float(oracle[missing].sum())
                    a['entropy'] += float((-(gate * gate.clamp_min(1e-30).log()).sum(1))[missing].sum())
                    for e in range(3):
                        a['usage'][e] += float(gate[:, e][missing].sum())
                print(f'[route val] {run.name} batch {step + 1}/{len(loader)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    metrics = {name: a.compute() for name, a in acc.items()}
    for a in alignment:
        if not a['count']:
            raise ValueError('No missing validation entries')
        for key in ('agreement', 'regret', 'weighted_regret', 'uniform_regret', 'oracle_error', 'entropy'):
            a[key] /= a['count']
        a['usage'] = [value / a['count'] for value in a['usage']]
    reference = checkpoint['metrics']['val_mae']
    tolerance = policy['reference_tolerance']
    result = dict(run=str(run), split='validation_only', checkpoint_epoch=checkpoint['epoch'],
                  samples=len(dataset), conditions=metrics, scale_alignment=alignment,
                  reference_val_mae=reference,
                  diagnostic_device=str(device), torch_version=str(torch.__version__),
                  batch_size=policy['diagnostic_batch_size'], shuffle_seed=policy['diagnostic_shuffle_seed'],
                  reference_tolerance=tolerance,
                  reference_mae_absolute_difference=abs(reference - metrics['learned']['mae']),
                  reference_matches=math.isclose(reference, metrics['learned']['mae'],
                                                 rel_tol=tolerance['relative'], abs_tol=tolerance['absolute']),
                  caveat='Forced routing changes an entire scale before a nonlinear shared decoder. Errors are intervention proxies, NOT standalone expert errors. Oracle is post-hoc validation diagnostic, not deployable prediction. No test split loaded.')
    path = run / 'logs' / f'route_validation_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    payload = json.dumps(result, indent=2, allow_nan=False)
    path.write_text(payload)
    path.with_suffix('.log').write_text(payload + '\n')
    print(f'Saved {path}; reference_matches={result["reference_matches"]}', flush=True)
    if not result['reference_matches']:
        raise RuntimeError('Checkpoint validation MAE not reproduced; inspect diagnostic before interpretation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('multiseed', 'controls', 'diagnose', 'summary'))
    parser.add_argument('--config', default='configs/v22/verification.json')
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--diagnostic-device', default='cuda:0', help='Single-device inference, or cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--conda-env', default='difftdi')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.cpu_threads < 1 or len(set(args.gpus)) != len(args.gpus) or any(not x.isdigit() for x in args.gpus):
        parser.error('Distinct numeric GPUs and positive threads required')
    if len(args.gpus) < 2:
        parser.error('Follow-up protocol preserves the original DDP setup; supply at least two GPUs')
    policy, suite = load_policy(args.config)
    interpreter = [sys.executable] if args.conda_env == 'current' else ['conda', 'run', '--no-capture-output', '-n', args.conda_env, 'python']
    if args.stage == 'summary':
        if args.dry_run:
            print('WOULD write timestamped verification summary under outputs/v22')
        else:
            summarize(policy, suite, args.gpus)
        return
    pending = 0
    for job in jobs(policy, suite, args.gpus, args.stage == 'controls'):
        variant, dataset, pattern, rate, seed, cfg, _ = job
        if args.stage == 'diagnose' and variant not in policy['diagnostic_variants']:
            continue
        old = completed(cfg, variant)
        label = f'{variant} {dataset} {pattern}@{rate} seed={seed}'
        if args.stage == 'diagnose':
            if not old:
                print(f'[MISSING checkpoint] {label}', flush=True)
                continue
            command = interpreter + [str(Path(__file__).resolve()), '--audit-run', old['run'],
                                     str(resolve(args.config)), args.diagnostic_device, str(args.cpu_threads)]
        elif old:
            print(f'[SKIP complete] {label}: {old["run"]}', flush=True)
            continue
        elif args.stage == 'controls':
            pending += 1
            print(f'[{"WOULD RUN" if args.dry_run else "RUN"} control] {label}; epochs={cfg["train"]["epochs"]}', flush=True)
            if not args.dry_run:
                launch_control(job, args, interpreter)
            continue
        else:
            command = interpreter + ['scripts/v22/train.py', '--config', str(resolve(policy['suite'])),
                '--profile', policy['profile'], '--variant', variant, '--dataset', dataset, '--mask', pattern,
                '--rate', rate, '--seed', str(seed), '--epochs', str(policy['epochs']), '--gpus', *args.gpus,
                '--cpu-threads', str(args.cpu_threads), '--conda-env', 'current']
        pending += 1
        print(f'[{"WOULD RUN" if args.dry_run else "RUN"}] {label}\n{shlex.join(command)}', flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True, env=dict(os.environ, PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1'))
    print(f'[stage {args.stage}] {"planned" if args.dry_run else "executed"}={pending}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker()
    elif len(sys.argv) > 1 and sys.argv[1] == '--audit-run':
        policy, _ = load_policy(sys.argv[3])
        diagnose(sys.argv[2], policy, sys.argv[4], int(sys.argv[5]))
    else:
        main()
