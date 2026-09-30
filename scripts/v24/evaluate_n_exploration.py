#!/usr/bin/env python3
"""Post-training only: evaluate the ID-validation-best checkpoint on fixed masks."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from stmoe_imputer.data import build_test_dataset,build_loader
from stmoe_imputer.engine import evaluate
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def evaluate_sets(config,checkpoint,test_npz,output,device=None):
    device=torch.device(device or config.get('device','cuda'))
    model=DualBranchSTImputer.from_config(config).to(device)
    saved=load_checkpoint(checkpoint,model,map_location=device)
    if saved.get('config') is not None and digest(saved['config'])!=digest(config):
        raise ValueError('Checkpoint config must exactly match the frozen training config')
    protocol=config['experiment_plan']['protocol']
    result={'status':'running','config_sha256':digest(config),'checkpoint':str(Path(checkpoint).resolve()),
            'run_dir':str(Path(checkpoint).resolve().parents[1]),'best_epoch':saved['epoch'],
            'selection':'in_distribution_validation_only','sets':{}}
    dataset=None
    for name,spec in protocol['evaluations'].items():
        cfg=copy.deepcopy(config)
        cfg['data']['eval_mask_diversity']={'families':spec['families'],'rates':[protocol['rate']],
                                          'seed':spec['seed'],'resample_each_epoch':False}
        # build_test_dataset adds its standard +30000 seed offset for all models.
        dataset=build_test_dataset(cfg,test_npz)
        loader=build_loader(dataset,cfg,shuffle=False)
        start=time.monotonic()
        with torch.no_grad():
            metrics=evaluate(model,loader,device,cfg,epoch=saved['epoch'],show_progress=False)
        result['sets'][name]={'mask_spec':cfg['data']['eval_mask_diversity'],
                             'effective_mask_seed':dataset.diverse_masks.seed,
                             'samples':len(dataset),'seconds':time.monotonic()-start,'metrics':metrics}
        write(output,result)
        del loader,dataset
    result['status']='finished'
    write(output,result)
    write(Path(checkpoint).resolve().parents[1]/'logs/n_evaluation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--test-npz',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    evaluate_sets(cfg,args.checkpoint,args.test_npz,args.output)


if __name__=='__main__':main()
