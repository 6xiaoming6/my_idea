#!/usr/bin/env python3
"""Train the optimized V23 scale-MoE using the existing best-val/final-test loop."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import run_dual_moe_comparison as common

ROOT = common.ROOT


def build_config(dataset, pattern, rate, aggregation, seed=42, epochs=None, preset=None):
    folder, prefix, base = common.SPECS[dataset]
    if preset is not None and (preset not in {'dual_moe_topk', 'dual_moe_target', 'dual_moe_shared_topk', 'dual_moe_st_dilated', 'dual_moe_recoverability'} or aggregation != 'learned_regions'):
        raise ValueError('Dual MoE presets require learned_regions aggregation')
    preset = preset+'.json' if preset else ('scale_completion_grid.json' if aggregation == 'regular_grid' else 'scale_completion.json')
    cfg = common.merge(common.load(ROOT/f'configs/datasets/{base}.json'), common.load(ROOT/'configs/presets'/preset))
    masks = {f'{split}_csv': str(ROOT/f'data/{folder}/{pattern}_mask/{rate:g}/{split}.csv') for split in ('train','val','test')}
    cfg = common.merge(cfg, {'seed':seed, 'device':'cuda:0',
                             'data':{'mask':{'pattern':pattern, 'missing_rate':rate, **masks}}})
    dataset_epochs = cfg['train'].pop('dataset_epochs', {})
    if dataset in dataset_epochs:
        cfg['train']['epochs'] = dataset_epochs[dataset]
    if epochs is not None:
        if epochs < 1:raise ValueError('epochs must be positive')
        cfg['train']['epochs'] = epochs
    paths = {s: ROOT/f'data/{folder}/{prefix}_{s}.npz' for s in ('train','val','test')}
    for p in [*paths.values(), *(Path(p) for p in masks.values())]:
        if not p.is_file():raise FileNotFoundError(p)
    return cfg,paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=tuple(common.SPECS),required=True)
    parser.add_argument('--mask',choices=('fixed','random'),required=True)
    parser.add_argument('--rate',type=float,choices=(.2,.4,.6,.8),default=.4)
    parser.add_argument('--aggregation',choices=('learned_regions','regular_grid'),default='learned_regions')
    parser.add_argument('--preset',choices=('dual_moe_topk','dual_moe_target','dual_moe_shared_topk','dual_moe_st_dilated','dual_moe_recoverability'),help='Stable: dual_moe_st_dilated; experimental: dual_moe_recoverability; historical defaults preserved')
    parser.add_argument('--front-mode', choices=('uniform','topk'), help='Matched target-MoE ablation; all aggregation experts are retained')
    parser.add_argument('--back-mode', choices=('uniform','topk'), help='Matched target-MoE ablation; all three completion experts are retained')
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--train-windows',type=int,help='Optional evenly spaced TRAIN subset; omit for full training data')
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not args.gpu.isdigit() or args.cpu_threads<1:parser.error('One GPU index and positive CPU threads required')
    if args.train_windows is not None and args.train_windows<1:parser.error('--train-windows must be positive')
    if (args.front_mode or args.back_mode) and args.preset != 'dual_moe_target':
        parser.error('--front-mode/--back-mode require --preset dual_moe_target')
    cfg,paths=build_config(args.dataset,args.mask,args.rate,args.aggregation,args.seed,args.epochs,args.preset)
    options=cfg['model']['dual_moe']
    for side, mode in [('aggregation', args.front_mode), ('completion', args.back_mode)]:
        if mode:
            options[f'{side}_mode'] = mode
            if mode == 'uniform':
                cfg['loss'][f'dual_moe_{side}_balance_weight'] = 0.
    run_name = 'full'
    if args.preset == 'dual_moe_target':
        run_name = 'ablation_Q'+str(int(options['aggregation_mode'] == 'topk'))+str(int(options['completion_mode'] == 'topk'))
    elif args.preset in ('dual_moe_shared_topk', 'dual_moe_st_dilated', 'dual_moe_recoverability'):
        run_name = f'shared1_E{options["completion_experts"]}_K{options["completion_top_k"]}'
        if args.preset == 'dual_moe_st_dilated':
            run_name = 'ST_DILATED_'+run_name
        elif args.preset == 'dual_moe_recoverability':
            run_name = 'RECOVERY_'+run_name
            print(f'[recoverability] {options["recoverability"]}; local model-relative constraint, not calibrated uncertainty')
    print(f'[model] {options["design"]}, aggregation={args.aggregation}, E={options["aggregation_experts"] if args.aggregation=="learned_regions" else 1}')
    if args.preset:
        print(f'[routing] aggregation={options["aggregation_mode"]}, K={options["aggregation_top_k"]}; completion={options["completion_mode"]}, K={options["completion_top_k"]}')
        if options.get('completion_layout') == 'routed_shared':
            print(f'[completion] {options["completion_experts"]} routed experts + 1 always-on shared expert; Top-K excludes shared; sparse mixing, dense computation')
            print(f'[expert] routed={options.get("completion_expert_type", "mlp")}, hidden={options.get("completion_expert_hidden", options["dim"])}; shared=pointwise MLP')
        print(f'[balance] front={cfg["loss"]["dual_moe_aggregation_balance_weight"]}, back={cfg["loss"]["dual_moe_completion_balance_weight"]}; name={run_name}')
    print(f'[budget] train={args.train_windows or "FULL"}, FULL val/test; epochs={cfg["train"]["epochs"]}, val_epoch={cfg["train"]["val_epoch"]}, batch={cfg["data"]["batch_size"]}')
    print(f'[output] {cfg["output_dir"]}',flush=True)
    if args.dry_run:return
    if args.train_windows is not None:
        selection={'datasets':[args.dataset],'patterns':[args.mask],'rate':args.rate,'train_windows':args.train_windows}
        sources=[common.stamp(paths['train']),common.stamp(Path(cfg['data']['mask']['train_csv']))]
        cache=ROOT/'outputs/v23/scale_completion/data'/common.digest({'selection':selection,'sources':sources})[:16]
        data=common.prepare(selection,cache)[args.dataset]
        paths['train']=data/'train.npz';cfg['data']['mask']['train_csv']=str(data/f'{args.mask}_train.csv')
        cfg['data']['training_selection']={'strategy':'evenly_spaced','limit':args.train_windows,'manifest':str(data/'selection.json')}
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=args.gpu;env['PYTHONUNBUFFERED']='1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]=str(args.cpu_threads)
    with tempfile.TemporaryDirectory(prefix='scale_completion_') as tmp:
        path=Path(tmp)/'config.json';common.write_json(path,cfg)
        command=[sys.executable,'-u',str(ROOT/'scripts/train.py'),'-c',str(path),'--name',run_name,'--no_plot','--quiet']
        for split,p in paths.items():command.extend([f'--{split}_npz',str(p)])
        subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
