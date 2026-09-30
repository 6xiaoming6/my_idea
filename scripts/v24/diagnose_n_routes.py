#!/usr/bin/env python3
"""Validation-only candidate-path diagnostic, never a deployable oracle score.

At one round replace native Top-2 by one of five pairs from native Top-4;
subsequent rounds reroute normally. No model fitting/checkpoint selection here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader,Subset
from stmoe_imputer.data import build_datasets
from stmoe_imputer.losses import supervision_mask
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


def sample_errors(prediction,batch):
    target=batch['x_f_gt']
    selected=supervision_mask(target,batch['m_f'],batch.get('target_mask'))
    error=torch.where(selected,(prediction-torch.nan_to_num(target)).abs(),0.).flatten(1).sum(1)
    count=selected.flatten(1).sum(1)
    return error/count.clamp_min(1),count


def diagnose(config,checkpoint,val_npz,samples=96):
    device=torch.device(config.get('device','cuda'))
    model=DualBranchSTImputer.from_config(config).to(device).eval()
    backbone=model.main_branch
    if backbone.top_k!=2 or backbone.pair_mode!='native' or backbone.num_steps!=4:
        raise ValueError('Diagnostic requires four-round native Top-2')
    load_checkpoint(checkpoint,model,map_location=device)
    # Both paths intentionally point to validation NPZ; only the deterministic
    # validation schedule is consumed. The test set is never read.
    _,dataset=build_datasets(config,val_npz,val_npz)
    assignments=dataset.diverse_masks.assignments
    groups=[list((assignments==i).nonzero()[0]) for i in range(len(dataset.diverse_masks.families))]
    chosen=[]
    for offset in range(max(map(len,groups))):
        for group in groups:
            if offset<len(group):chosen.append(int(group[offset]))
    chosen=chosen[:min(samples,len(chosen))]
    loader=DataLoader(Subset(dataset,chosen),batch_size=min(8,config['data']['batch_size']),shuffle=False)
    pairs=((0,1),(0,2),(1,2),(0,3),(2,3))
    rows=[];started=time.monotonic()
    with torch.no_grad():
        cursor=0
        for batch in loader:
            batch=move_batch_to_device(batch,device)
            native=model(batch);native_error,count=sample_errors(native['x_hat_main'],batch)
            for step in range(4):
                top=native['coe']['route_logits'][:,step].topk(4,dim=-1).indices
                final_errors=[native_error]
                immediate_errors=[sample_errors(native['coe']['predictions'][step],batch)[0]]
                for left,right in pairs[1:]:
                    ids=torch.stack((top[:,left],top[:,right]),dim=1)
                    output=model({**batch,'forced_pair_step':step,'forced_pair_indices':ids})
                    final_errors.append(sample_errors(output['x_hat_main'],batch)[0])
                    immediate_errors.append(sample_errors(output['coe']['predictions'][step],batch)[0])
                final=torch.stack(final_errors,1);immediate=torch.stack(immediate_errors,1)
                local_choice=immediate.argmin(1);best=final.argmin(1)
                for index in range(len(native_error)):
                    rows.append({'sample_index':chosen[cursor+index],'family_id':int(batch['mask_family'][index]),
                                 'step':step+1,'target_count':int(count[index]),
                                 'native_final_mae':float(native_error[index]),
                                 'candidate_best_final_mae':float(final[index,best[index]]),
                                 'immediate_choice_final_mae':float(final[index,local_choice[index]]),
                                 'immediate_choice':int(local_choice[index]),'final_choice':int(best[index])})
            cursor+=len(native_error)
    summary={}
    for step in range(1,5):
        selected=[r for r in rows if r['step']==step];den=sum(r['target_count'] for r in selected)
        def average(key):return sum(r[key]*r['target_count'] for r in selected)/max(1,den)
        summary[f'step{step}']={key:average(key) for key in ('native_final_mae','candidate_best_final_mae','immediate_choice_final_mae')}
        summary[f'step{step}']['choice_disagreement']=sum(r['immediate_choice']!=r['final_choice'] for r in selected)/max(1,len(selected))
    return {'status':'finished','split':'validation_only','samples':len(chosen),'seconds':time.monotonic()-started,
            'candidate_pairs_top4_ranks':pairs,'summary':summary,'rows':rows,
            'warning':'Uses hidden validation labels only as a diagnostic. Not test results, not a learned router, not a global oracle. Intermediate readouts have no auxiliary supervision.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','checkpoint','val-npz','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--samples',type=int,default=96)
    args=parser.parse_args()
    if args.samples<1:parser.error('samples must be positive')
    result=diagnose(json.loads(Path(args.config).read_text()),args.checkpoint,args.val_npz,args.samples)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)


if __name__=='__main__':main()
