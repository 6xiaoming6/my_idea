#!/usr/bin/env python3
"""Reuse real-data native and forced TA/ST acceptance on every frozen Z config."""
import argparse
import gc
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import torch
from run_b3_c3 import load,write,digest
from run_history_mechanism import validate
from preflight_history_followup import check_case
from run_team_accept_v4 import check_gpu_idle
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import move_batch_to_device
from stmoe_imputer.utils.deterministic import configure

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('GPU0 only')
    payload=load(a.suite/'plan.json');validate(a.suite,payload);check_gpu_idle([0]);torch.set_num_threads(2)
    result=dict(status='running',source_sha256=digest(payload['source']),configs_sha256=digest({n:j['config'] for n,j in payload['jobs'].items()}),cases=[])
    path=a.suite/'acceptance/preflight.json'
    try:
        for job in payload['jobs'].values():
            configure(job['config']);ds,_=build_datasets(job['config'],job['sources']['train'],job['sources']['val'])
            batch=move_batch_to_device(next(iter(build_loader(ds,job['config'],shuffle=False))),torch.device('cuda'))
            for heavy in (False,True):
                row=check_case(job,batch,heavy);result['cases'].append(row);write(path,result);print(row,flush=True)
                gc.collect();torch.cuda.empty_cache()
            del batch,ds;gc.collect();torch.cuda.empty_cache()
    except BaseException as error:
        result.update(status='failed',error=repr(error));write(path,result);raise
    result['status']='passed';write(path,result)

if __name__=='__main__':main()
