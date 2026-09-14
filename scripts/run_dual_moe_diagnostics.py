#!/usr/bin/env python3
"""Three sequential, matched V23 diagnostics. Does not change model source code.

1. Frozen best-checkpoint validation interventions (not test-set selection).
2. Train one learned aggregator vs existing three-aggregator controls.
3. Train weak per-scale supervision, changing only its loss coefficient.
All retraining inherits the original data/budget/seed; all stages are predeclared.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

import run_dual_moe_comparison as common

ROOT = common.ROOT


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(policy):
    if policy['reference_variants'] != ['A01','A11']:
        raise ValueError('Keep both predeclared A01/A11 references')
    for key, allowed in [('datasets',common.SPECS),('patterns',('fixed','random'))]:
        values = policy[key]
        if not values or len(values) != len(set(values)) or any(v not in allowed for v in values):
            raise ValueError(f'Invalid {key}')
    weights = policy['expert_loss_weights']
    if not weights or len(weights) != len(set(weights)) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Positive distinct loss weights; weight=0 references are reused')
    seeds = policy['shuffle_seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s,int) or s < 0 for s in seeds):
        raise ValueError('Nonnegative distinct permutation seeds required')
    if policy['cpu_threads'] < 1 or not 0 < policy['baseline_relative_tolerance'] < .01:
        raise ValueError('Invalid threads or baseline tolerance')
    if not math.isfinite(policy['practical_change_percent']) or policy['practical_change_percent'] < 0:
        raise ValueError('Invalid descriptive practical threshold')


def audit_sources(policy):
    """Refuse silently mixing changed code, data, configs or incomplete runs."""
    source = common.resolve(policy['source_suite'])
    protocol = common.load(source/'protocol.json'); original = protocol['policy']
    for name, sha in protocol['code'].items():
        if file_hash(ROOT/name) != sha:
            raise RuntimeError(f'Source code changed since reference training: {name}. Do not mix protocols.')
    for item in protocol['sources']:
        if common.stamp(ROOT/item['path']) != item:
            raise RuntimeError(f'Source data/config changed: {item["path"]}')
    rows = common.load(source/'summary.json'); references = {}
    for dataset in policy['datasets']:
        for pattern in policy['patterns']:
            for variant in policy['reference_variants']:
                candidates = [r for r in rows if (r['dataset'],r['pattern'],r['variant']) == (dataset,pattern,variant)]
                if len(candidates) != 1 or candidates[0]['status'] != 'complete':
                    raise RuntimeError(f'Missing/ambiguous reference: {dataset}/{pattern}/{variant}')
                cfg, paths = common.job_config(original,source,source/'data'/dataset,dataset,pattern,variant)
                completed = common.completed_run(cfg,variant)
                if not completed or Path(completed['run_dir']) != Path(candidates[0]['run_dir']):
                    raise RuntimeError(f'Reference config/completion mismatch: {dataset}/{pattern}/{variant}')
                if cfg['model']['dual_moe']['aggregation_experts'] != 3 or cfg['loss']['dual_moe_expert_weight'] != 0:
                    raise RuntimeError('Expected E=3, auxiliary weight=0 references')
                for path in paths.values():
                    if not path.is_file():
                        raise FileNotFoundError(path)
                key = f'{dataset}_{pattern}_{variant}'
                references[key] = {'dataset':dataset,'pattern':pattern,'variant':variant,
                                   'cfg':cfg,'paths':{k:str(v) for k,v in paths.items()},**completed}
    extra = {str(Path(__file__).relative_to(ROOT)):file_hash(__file__)}
    for key, ref in references.items():
        run = Path(ref['run_dir'])
        for path in [run/'checkpoints/best.pt',run/'config.json',run/'logs/metrics.jsonl',
                     Path(ref['paths']['train']),Path(ref['cfg']['data']['mask']['train_csv'])]:
            extra[str(path)] = file_hash(path)
    return references, {'policy':policy,'source_protocol':protocol,'reference_artifact_hashes':extra}


def candidate_config(reference, suite, stage, name, patch, device):
    cfg = common.merge(reference['cfg'],patch)
    cfg['output_dir'] = str(suite/f'stage{stage}'/'runs')
    cfg['device'] = device
    return cfg


def training_jobs(policy, references, suite, device):
    jobs = {2:[],3:[]}
    for ref in references.values():
        if ref['variant'] == 'A01':
            name='S2_E1'
            cfg=candidate_config(ref,suite,2,name,{'model':{'dual_moe':{'aggregation_experts':1}}},device)
            jobs[2].append((name,ref,cfg))
        for weight in policy['expert_loss_weights']:
            name=f'S3_{ref["variant"]}_w{weight:g}'.replace('.','p')
            cfg=candidate_config(ref,suite,3,name,{'loss':{'dual_moe_expert_weight':weight}},device)
            jobs[3].append((name,ref,cfg))
    return jobs


def intervention_specs(policy):
    return [('baseline',0),('slot_permutation_control',policy['shuffle_seeds'][0]),
            ('zero_mid',0),('zero_coarse',0),('zero_both',0),('zero_fine_context',0)]+[
                ('shuffle_both',seed) for seed in policy['shuffle_seeds']]


@contextmanager
def intervention(model, mode, seed):
    """Perturb only coarse-path condition inputs; keep support and fine head intact.

The posterior completion gate naturally recomputes; this measures end-to-end
sensitivity, NOT fixed-gate direct effects. Permutations are fixed over time.
"""
    import torch
    valid={'baseline','slot_permutation_control','zero_mid','zero_coarse','zero_both','zero_fine_context','shuffle_both'}
    if mode not in valid:
        raise ValueError(mode)
    handles, saved = [], []
    backbone=model.main_branch
    try:
        if mode == 'slot_permutation_control':
            gen=torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for aggregation in backbone.aggregation.values():
                    for expert in aggregation.experts:
                        saved.append((expert.slots,expert.slots.detach().clone()))
                        p=torch.randperm(len(expert.slots),generator=gen).to(expert.slots.device)
                        expert.slots.copy_(expert.slots[p])
        elif mode != 'baseline':
            def make_hook(scale):
                def hook(module,args):
                    z=args[0].clone(); dim=z.shape[1]//2
                    if mode == 'zero_fine_context':
                        z[:,dim:]=0
                    elif mode == 'shuffle_both':
                        n=z.shape[-2]*z.shape[-1]
                        g=torch.Generator().manual_seed(seed+(0 if scale=='mid' else 1009))
                        p=torch.randperm(n,generator=g).to(z.device)
                        z[:,:dim]=z[:,:dim].flatten(-2)[...,p].reshape_as(z[:,:dim])
                    elif mode == 'zero_both' or mode == f'zero_{scale}':
                        z[:,:dim]=0
                    return (z,)
                return hook
            for scale, branch in backbone.scale_experts.items():
                handles.append(branch.condition.register_forward_pre_hook(make_hook(scale)))
        yield
    finally:
        for h in handles:h.remove()
        with torch.no_grad():
            for parameter,value in saved:parameter.copy_(value)


def load_evaluator(cfg, paths, run, device):
    import torch
    from stmoe_imputer.data import build_loader
    from stmoe_imputer.models import DualBranchSTImputer
    from stmoe_imputer.utils.checkpoint import load_checkpoint
    # Build only the validation split; no test metrics are used for interventions.
    from stmoe_imputer.data import FlowNPZDataset
    scale=cfg['data']['scales']
    dataset=FlowNPZDataset(paths['val'],mask_cfg=cfg['data']['mask'],
                          mask_csv=cfg['data']['mask']['val_csv'],fine_to_mid=scale['fine_to_mid'],
                          fine_to_coarse=scale['fine_to_coarse'],pooling_mode=scale.get('pooling_mode','avg'))
    loader=build_loader(dataset,cfg,False)
    model=DualBranchSTImputer.from_config(cfg).to(device).eval()
    checkpoint=load_checkpoint(Path(run)/'checkpoints/best.pt',model,map_location=device)
    if checkpoint['config'] != cfg:
        raise RuntimeError('Checkpoint/config mismatch')
    return model,loader


def measure(model, loader, device, mode='baseline', seed=0):
    import torch
    from stmoe_imputer.metrics import MaskedMetricAccumulator
    from stmoe_imputer.routing_metrics import DualMoEMetricAccumulator
    exact=MaskedMetricAccumulator(); detail=DualMoEMetricAccumulator()
    weighted_abs, final_abs, signed, count = 0.,0.,[0.,0.,0.],0.
    with torch.no_grad(), intervention(model,mode,seed):
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            out=model(batch); pred=out['x_hat_final']; target=batch['x_f_gt']
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f'Nonfinite prediction during {mode}')
            exact.update(pred,target,batch['m_f']);detail.update(out,batch)
            missing=(1-batch['m_f']).expand_as(target)
            count+=float(missing.sum())
            final_abs+=float(((pred-target).abs()*missing).sum())
            for i,scale in enumerate(('fine','mid','coarse')):
                diff=out['scale_predictions'][scale]-target
                weighted_abs+=float((out['completion_gates'][:,i:i+1]*diff.abs()*missing).sum())
                signed[i]+=float((diff*missing).sum())
    result={**exact.compute(),**detail.compute(),
            'weighted_branch_mae':weighted_abs/max(1,count),
            'error_cancellation_fraction':1-final_abs/weighted_abs if weighted_abs>0 else 0.,
            **{f'signed_error_{s}':signed[i]/max(1,count) for i,s in enumerate(('fine','mid','coarse'))},
            'total_params':sum(p.numel() for p in model.parameters()),
            'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError(f'Nonfinite diagnostic metrics: {mode}')
    return result


def close_enough(a,b,tolerance):
    return abs(a-b) <= tolerance*max(abs(b),1e-6)+1e-6


def descriptive_change(delta,threshold):
    # This label is a declared practical threshold, never a significance test.
    return 'lower_error' if delta < -threshold else 'higher_error' if delta > threshold else 'small_change'


def log_line(suite,text):
    print(text,flush=True)
    with (suite/'diagnostics.log').open('a') as f:f.write(text+'\n')


def run_stage1(policy, references, suite, device):
    import torch
    for key,ref in references.items():
        path=suite/'stage1'/f'{key}.json'
        cached=common.load(path) if path.exists() else {}
        if cached.get('status')=='complete':
            log_line(suite,f'[stage1 SKIP] {key}');continue
        model,loader=load_evaluator(ref['cfg'],ref['paths'],ref['run_dir'],device)
        rows=[];baseline=None
        for mode,seed in intervention_specs(policy):
            metrics=measure(model,loader,device,mode,seed)
            if mode=='baseline':
                baseline=metrics
                if not close_enough(metrics['mae'],ref['val_mae'],policy['baseline_relative_tolerance']):
                    raise RuntimeError(f'Baseline validation replay mismatch: {key}')
            if mode=='slot_permutation_control' and not close_enough(metrics['mae'],baseline['mae'],policy['baseline_relative_tolerance']):
                raise RuntimeError(f'Permutation negative control failed: {key}')
            change=100*(metrics['mae']/max(baseline['mae'],1e-12)-1)
            rows.append({'intervention':mode,'permutation_seed':seed,'delta_mae_percent':change,
                         'descriptive_change':descriptive_change(change,policy['practical_change_percent']),'metrics':metrics})
            log_line(suite,f'[stage1] {key} {mode}/{seed}: val_mae={metrics["mae"]:.6f} change={change:+.3f}%')
        common.write_json(path,{'status':'complete','reference':ref['run_dir'],'split':'val','rows':rows})
        del model,loader
        if device.type=='cuda':torch.cuda.empty_cache()


def launch_training(name,ref,cfg,suite,stage):
    existing=common.completed_run(cfg,name)
    key=f'{ref["dataset"]}_{ref["pattern"]}_{name}'
    if existing:
        log_line(suite,f'[stage{stage} SKIP] {key}');return existing
    path=suite/f'stage{stage}'/'configs'/f'{key}.json';common.write_json(path,cfg)
    command=[sys.executable,'-u',str(ROOT/'scripts/train.py'),'-c',str(path),'--name',f'ablation_{name}','--no_plot','--quiet']
    for split,file in ref['paths'].items():command.extend([f'--{split}_npz',file])
    raw=suite/f'stage{stage}'/'logs'/f'{key}.log';raw.parent.mkdir(parents=True,exist_ok=True)
    log_line(suite,f'[stage{stage} RUN] {key}')
    with raw.open('a') as f:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout:
                f.write(line);f.flush()
                if line.strip() and not line.lstrip().startswith(('train epoch ','val epoch ','test best epoch ')):
                    print(line,end='',flush=True)
            code=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    result=common.completed_run(cfg,name)
    if code or not result:
        raise RuntimeError(f'{key} failed, exit={code}; inspect {raw}. Rerun same command to retry.')
    return result


def run_training_stage(stage,jobs,suite,device,policy):
    import torch
    for name,ref,cfg in jobs:
        result=launch_training(name,ref,cfg,suite,stage)
        key=f'{ref["dataset"]}_{ref["pattern"]}_{name}'
        path=suite/f'stage{stage}'/f'{key}.json'
        # Completion of training does not imply completion of follow-up diagnostics.
        cached=common.load(path) if path.exists() else {}
        if cached.get('status')=='complete' and cached.get('run_dir')==result['run_dir']:
            continue
        model,loader=load_evaluator(cfg,ref['paths'],result['run_dir'],device)
        metrics=measure(model,loader,device)
        if not close_enough(metrics['mae'],result['val_mae'],policy['baseline_relative_tolerance']):
            raise RuntimeError(f'Candidate validation replay mismatch: {key}')
        row={'status':'complete','dataset':ref['dataset'],'pattern':ref['pattern'],'name':name,
             'reference_variant':ref['variant'],'reference_run':ref['run_dir'],**result,
             'reference_val_mae':ref['val_mae'],'reference_test_mae':ref['test_mae'],
             'delta_val_mae_percent':100*(result['val_mae']/ref['val_mae']-1),
             'delta_test_mae_percent':100*(result['test_mae']/ref['test_mae']-1),
             'validation_diagnostics':metrics}
        row['validation_signal']=descriptive_change(row['delta_val_mae_percent'],policy['practical_change_percent'])
        common.write_json(path,row)
        log_line(suite,f'[stage{stage} DONE] {key}: val change={row["delta_val_mae_percent"]:+.3f}% test MAE={result["test_mae"]:.6f}')
        del model,loader
        if device.type=='cuda':torch.cuda.empty_cache()


def summarize(policy,references,jobs,suite):
    result={'stage1':[],'stage2':[],'stage3':[],'missing':[],
            'references':[{k:r[k] for k in ('dataset','pattern','variant','run_dir','val_mae','test_mae','test_rmse','best_epoch')} for r in references.values()],
            'interpretation':'Negative MAE change is better. Interventions are sensitivity, not retrained ablations. Single seed; no significance claims.'}
    for key,ref in references.items():
        path=suite/'stage1'/f'{key}.json'
        if path.exists() and common.load(path).get('status')=='complete':
            result['stage1'].append({'key':key,**common.load(path)})
        else:result['missing'].append(f'stage1/{key}')
    for stage in (2,3):
        for name,ref,cfg in jobs[stage]:
            key=f'{ref["dataset"]}_{ref["pattern"]}_{name}';path=suite/f'stage{stage}'/f'{key}.json'
            row=common.load(path) if path.exists() else {}
            if row.get('status')=='complete' and common.completed_run(cfg,name):
                result[f'stage{stage}'].append(row)
            else:result['missing'].append(f'stage{stage}/{key}')
    common.write_json(suite/'summary.json',result)
    columns=['stage','dataset','pattern','name','reference_variant','val_mae','reference_val_mae','delta_val_mae_percent','test_mae','test_rmse','delta_test_mae_percent','best_epoch','total_params','trainable_params','error_cancellation_fraction','run_dir']
    with (suite/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for row in result['references']:
            writer.writerow({**row,'stage':'reference','name':row['variant'],'reference_variant':row['variant']})
        for stage in (2,3):
            for row in result[f'stage{stage}']:
                writer.writerow({**row,'stage':stage,**{k:row['validation_diagnostics'][k] for k in ('total_params','trainable_params','error_cancellation_fraction')}})
    log_line(suite,f'[summary] stage1={len(result["stage1"])}/{len(references)} stage2={len(result["stage2"])}/{len(jobs[2])} stage3={len(result["stage3"])}/{len(jobs[3])}; {suite/"summary.json"}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/presets/dual_moe_diagnostics.json')
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--cpu',action='store_true',help='For small functional tests, not the full queue')
    parser.add_argument('--stages',nargs='+',type=int,choices=(1,2,3),default=[1,2,3])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args()
    if not args.gpu.isdigit():parser.error('One numeric GPU index required')
    policy=common.load(common.resolve(args.config));validate(policy)
    os.environ['CUDA_VISIBLE_DEVICES']='' if args.cpu else args.gpu
    os.environ['PYTHONUNBUFFERED']='1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]=str(policy['cpu_threads'])
    references,record=audit_sources(policy)
    record['device_type']='cpu' if args.cpu else 'cuda'
    suite=common.resolve(policy['output_dir'])/common.digest(record)[:16]
    jobs=training_jobs(policy,references,suite,'cpu' if args.cpu else 'cuda:0')
    print(f'[suite] {suite}',flush=True)
    for stage in sorted(set(args.stages)):
        print(f'[stage{stage}] '+(f'{len(references)} frozen checkpoints × {len(intervention_specs(policy))} validation passes' if stage==1 else f'{len(jobs[stage])} new training jobs; reuse zero-loss E=3 controls'),flush=True)
    if args.dry_run:return
    import fcntl
    suite.mkdir(parents=True,exist_ok=True)
    with (suite/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        common.write_json(suite/'protocol.json',record)
        if args.summary_only:
            summarize(policy,references,jobs,suite);return
        import torch
        sys.path.insert(0,str(ROOT/'src'))
        torch.set_num_threads(policy['cpu_threads'])
        device=torch.device('cpu' if args.cpu else 'cuda:0')
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Activate difftdi: CUDA unavailable')
        try:
            for stage in sorted(set(args.stages)):
                if stage==1:run_stage1(policy,references,suite,device)
                else:run_training_stage(stage,jobs[stage],suite,device,policy)
                summarize(policy,references,jobs,suite)
        finally:
            summarize(policy,references,jobs,suite)


if __name__=='__main__':
    main()
