#!/usr/bin/env python3
"""Real TaxiBJ batch32 AMP checks for both FFN sharing modes on GPU0."""
import argparse, copy, gc, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).resolve().parent)]
import torch
from run_post_fusion_ffn import jobs
from run_b3_c3 import write, digest
from run_scale_rate_compare import filehash
from run_team_accept_v4 import check_gpu_idle
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets, build_loader
from stmoe_imputer.engine import build_optimizer, build_grad_scaler, move_batch_to_device
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import configure


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('GPU0 only')
    check_gpu_idle([0]);torch.set_num_threads(2);js=jobs()
    result={'status':'running','model_sha256':filehash(ROOT/'src/stmoe_imputer/models/post_fusion_ffn_coe.py'),
            'configs_sha256':digest({n:j['config'] for n,j in js.items()}),'cases':[]}
    first=next(iter(js.values()));configure(first['config'])
    ds,_=build_datasets(first['config'],first['sources']['train'],first['sources']['val'])
    batch=move_batch_to_device(next(iter(build_loader(ds,first['config'],shuffle=False))),torch.device('cuda'))
    for name,j in js.items():
        for heavy in (False,True):
            c=copy.deepcopy(j['config']);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            model=DualBranchSTImputer.from_config(c).cuda().train()
            if heavy:
                with torch.no_grad():
                    for router in model.main_branch.routers:
                        router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
            opt=build_optimizer(model,c);scaler=build_grad_scaler(torch.device('cuda'),c)
            torch.cuda.reset_peak_memory_stats();started=time.monotonic();updates=0
            try:
                for attempt in range(24):
                    opt.zero_grad(set_to_none=True);calls=[];ffn_calls=[]
                    hooks=[e.register_forward_pre_hook(lambda module,args:calls.append(len(args[0]))) for e in model.main_branch.routed_experts(0)]
                    hooks += [f.register_forward_pre_hook(lambda module,args:ffn_calls.append(tuple(args[0].shape))) for f in model.main_branch.post_ffns]
                    with torch.autocast('cuda',dtype=torch.float16):out=model(batch);loss,_=compute_coe_loss(out,batch,c)
                    for hook in hooks:hook.remove()
                    if sum(calls)!=8*len(batch['m_f']) or ffn_calls!=[(32,64,12,32,32)]*4:
                        raise RuntimeError('Wrong expert/FFN execution')
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite forward')
                    old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                    groups=[list(f.parameters()) for f in model.main_branch.post_ffns]+[list(model.main_branch.history_gate.parameters())]
                    valid=all(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in ps) and
                              all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in ps) for ps in groups)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
                    if scaler.get_scale()>=old:
                        if not valid:raise RuntimeError('Missing finite nonzero FFN/gate gradient')
                        updates+=1
                    del out,loss
                    if updates>=2:break
                if updates<2:raise RuntimeError('Insufficient successful AMP updates')
                torch.cuda.synchronize()
                row={'variant':name,'case':'forced_TA_ST' if heavy else 'native','status':'passed',
                     'expert_calls_per_window':8,'ffn_calls_per_window':4,'ffn_grid':[12,32,32],
                     'attempts':attempt+1,'updates':updates,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
                     'seconds':time.monotonic()-started}
            except BaseException as error:
                result['status']='failed';result['cases'].append({'variant':name,'heavy':heavy,'error':repr(error)});write(a.output,result);raise
            result['cases'].append(row);write(a.output,result);print(row,flush=True)
            del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(a.output,result)

if __name__=='__main__':main()
