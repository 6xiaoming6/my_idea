#!/usr/bin/env python3
"""Evaluate a checkpoint on the explicitly named validation or test time split."""
import argparse,copy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from stmoe_imputer.data import build_datasets,build_test_dataset,build_loader
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.engine import evaluate
from run_b3_c3 import digest,load,write


def evaluate_sets(checkpoint,protocol,npz,output,expected_sha=None,device='cuda'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];sha=digest(cfg)
    if expected_sha and sha!=expected_sha:raise ValueError('Saved config differs from frozen job')
    from stmoe_imputer.utils.deterministic import configure
    configure(cfg)
    model=DualBranchSTImputer.from_config(cfg);model.load_state_dict(saved['model']);model.to(device).eval()
    split=protocol.get('split','test')
    if split not in ('val','test'):raise ValueError('Unknown evaluation split')
    result={'status':'running','checkpoint':str(Path(checkpoint).resolve()),'config_sha256':sha,'protocol_sha256':digest(protocol),
            'split':split,'npz':str(Path(npz).resolve()),'best_epoch':saved['epoch'],'selection':'ID validation only','sets':{}}
    for name,spec in protocol['evaluations'].items():
        c=copy.deepcopy(cfg);rate=spec.get('rate',protocol['rate'])
        c['data']['mask']['missing_rate']=rate
        c['data']['eval_mask_diversity']={'families':spec['families'],'rates':[rate],'seed':spec['seed'],'resample_each_epoch':False}
        ds=build_datasets(c,npz,npz)[1] if split=='val' else build_test_dataset(c,npz)
        loader=build_loader(ds,c,shuffle=False);started=time.monotonic()
        metrics=evaluate(model,loader,torch.device(device),c,epoch=saved['epoch'],show_progress=False)
        result['sets'][name]={'metrics':metrics,'samples':len(ds),'seconds':time.monotonic()-started,
                             'effective_mask_seed':ds.diverse_masks.seed,'mask_spec':c['data']['eval_mask_diversity']}
        write(output,result)
    result['status']='finished';write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('checkpoint','protocol','npz','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--expected-sha');p.add_argument('--device',default='cuda');a=p.parse_args()
    evaluate_sets(a.checkpoint,load(a.protocol),a.npz,a.output,a.expected_sha,a.device)
