#!/usr/bin/env python3
"""Evaluate saved-config best checkpoints; reference outputs never modify old runs."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from stmoe_imputer.data import build_test_dataset,build_loader
from stmoe_imputer.engine import evaluate
from stmoe_imputer.models import DualBranchSTImputer
from run_b3_c3 import digest,load,write


def evaluate_sets(checkpoint,protocol,test_npz,output,expected_sha=None,device='cuda'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config']
    if expected_sha and digest(cfg)!=expected_sha:
        raise ValueError('Checkpoint does not match the frozen training config')
    model=DualBranchSTImputer.from_config(cfg)
    model.load_state_dict(saved['model'])
    model.to(torch.device(device)).eval()
    result={'status':'running','config_sha256':digest(cfg),'protocol_sha256':digest(protocol),
            'checkpoint':str(Path(checkpoint).resolve()),'run_dir':str(Path(checkpoint).resolve().parents[1]),
            'best_epoch':saved['epoch'],'selection':'in_distribution_validation_only','sets':{}}
    del saved
    for name,spec in protocol['evaluations'].items():
        evaluation_cfg=copy.deepcopy(cfg)
        evaluation_cfg['data']['eval_mask_diversity']={'families':spec['families'],'rates':[protocol['rate']],
                                                     'seed':spec['seed'],'resample_each_epoch':False}
        dataset=build_test_dataset(evaluation_cfg,test_npz)
        loader=build_loader(dataset,evaluation_cfg,shuffle=False)
        started=time.monotonic()
        with torch.no_grad():
            metrics=evaluate(model,loader,torch.device(device),evaluation_cfg,epoch=result['best_epoch'],show_progress=False)
        result['sets'][name]={'mask_spec':evaluation_cfg['data']['eval_mask_diversity'],
                             'effective_mask_seed':dataset.diverse_masks.seed,'samples':len(dataset),
                             'seconds':time.monotonic()-started,'metrics':metrics}
        write(output,result)
    result['status']='finished';write(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','protocol','test-npz','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--expected-sha');p.add_argument('--device',default='cuda')
    a=p.parse_args()
    evaluate_sets(a.checkpoint,load(a.protocol),a.test_npz,a.output,a.expected_sha,a.device)

if __name__=='__main__':main()
