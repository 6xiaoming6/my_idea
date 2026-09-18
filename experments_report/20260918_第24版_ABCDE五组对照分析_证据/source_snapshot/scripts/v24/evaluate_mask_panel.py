#!/usr/bin/env python3
"""Evaluate saved A/D (or other) checkpoints on identical, fixed geometric masks.

This diagnostic panel never selects checkpoints; use checkpoints selected on the
original common validation CSV. It runs models sequentially on a single device.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from stmoe_imputer.data import FlowNPZDataset, build_loader
from stmoe_imputer.data.diverse_masks import FAMILIES, DiverseMaskSchedule
from stmoe_imputer.engine import evaluate
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--test-npz',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--rates',type=float,nargs='+',default=[.4])
    p.add_argument('--seed',type=int,default=20260918)
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Refusing to overwrite a previous panel')
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    torch.set_num_threads(2);device=torch.device(args.device)
    result={'mask_seed':args.seed,'mask_epoch':1,'families':FAMILIES,'rates':args.rates,
            'test_npz':str(Path(args.test_npz).resolve()),'test_sha256':sha(args.test_npz),
            'checkpoint_selection':'Original shared validation CSV only; this panel is diagnostic.', 'models':[]}
    for checkpoint in args.checkpoints:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
        cfg['data']['batch_size']=args.batch_size;cfg['data']['num_workers']=0
        model=DualBranchSTImputer.from_config(cfg).to(device)
        load_checkpoint(checkpoint,model,map_location=device)
        dataset=FlowNPZDataset(args.test_npz,multiscale=False,
            diverse_mask_config={'families':[FAMILIES[0]],'rates':[args.rates[0]],'seed':args.seed})
        item={'checkpoint':str(Path(checkpoint).resolve()),'sha256':sha(checkpoint),'epoch':saved['epoch'],'panel':{}}
        for family in FAMILIES:
            for rate in args.rates:
                dataset.diverse_masks=DiverseMaskSchedule(len(dataset),{'families':[family],'rates':[rate],'seed':args.seed})
                logs=evaluate(model,build_loader(dataset,cfg,shuffle=False),device,cfg,show_progress=False)
                item['panel'][f'{family}/rate{rate:g}']=logs
        result['models'].append(item)
        del model,saved,dataset
        if device.type=='cuda':torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(output)

if __name__=='__main__':main()
