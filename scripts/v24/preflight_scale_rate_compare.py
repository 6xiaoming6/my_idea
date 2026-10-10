#!/usr/bin/env python3
"""Real batch32 AMP acceptance for fixed, soft, transition and all-fine paths."""
import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import torch
from run_scale_rate_compare import jobs
from run_b3_c3 import write
from run_team_accept_v4 import check_gpu_idle
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,move_batch_to_device
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import configure


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('GPU0 only')
    check_gpu_idle([0]);torch.set_num_threads(2);js=jobs();result={'status':'running','cases':[],'masks':{}}
    configure(next(iter(js.values()))['config'])
    for dataset in ('taxibj','bikenyc'):
        for rate in (.2,.4,.6,.8):
            j=js[f'{dataset}_r{round(rate*100):02}_cmff']
            ds,val=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
            batch=next(iter(build_loader(ds,j['config'],shuffle=False)))
            actual=float((1-batch['m_f']).mean())
            if abs(actual-rate)>.001:raise RuntimeError('Incorrect mask rate')
            result['masks'][f'{dataset}_{rate}']={'rate':actual,'shape':list(batch['m_f'].shape),'train':len(ds),'val':len(val)}
            del ds,val,batch
        j=js[dataset+'_r80_cmff']
        ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
        batch=move_batch_to_device(next(iter(build_loader(ds,j['config'],shuffle=False))),torch.device('cuda'))
        for label,method,epoch,force,heavy in [('fixed','cmff',1,None,False),('soft','top1',1,None,False),
              ('transition','top1',6,None,False),('all_fine','top1',11,0,False),('soft_attention','top1',1,None,True)]:
            cfg=copy.deepcopy(js[dataset+'_r80_'+method]['config']);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            model=DualBranchSTImputer.from_config(cfg).cuda().train();model.main_branch.set_routing_epoch(epoch)
            if heavy:
                for router in model.main_branch.routers:
                    with torch.no_grad():router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
            if force is not None:
                for head in model.main_branch.scale_heads:
                    with torch.no_grad():head[-1].weight.zero_();head[-1].bias.zero_();head[-1].bias[force]=10
            opt=build_optimizer(model,cfg);scaler=build_grad_scaler(torch.device('cuda'),cfg)
            torch.cuda.reset_peak_memory_stats();start=time.monotonic();applied=0;last_calls=0
            try:
                for attempt in range(18):
                    opt.zero_grad(set_to_none=True);calls=[]
                    hooks=[e.register_forward_pre_hook(lambda e,args:calls.append(len(args[0]))) for e in model.main_branch.routed_experts()]
                    with torch.autocast('cuda',dtype=torch.float16):
                        out=model(batch);loss,_=compute_coe_loss(out,batch,cfg)
                    for h in hooks:h.remove()
                    last_calls=sum(calls)/32
                    expected=24 if method=='top1' and epoch<=10 else 8
                    if last_calls!=expected or not torch.isfinite(loss):raise RuntimeError('Invalid execution/forward')
                    old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                    gs=[p.grad for p in model.main_branch.scale_heads.parameters() if p.grad is not None]
                    finite_grad=bool(gs) and all(torch.isfinite(g).all() for g in gs) and any(g.abs().sum()>0 for g in gs)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
                    if scaler.get_scale()>=old:
                        applied+=1
                        if method=='top1' and not finite_grad:raise RuntimeError('Invalid scale gradients')
                    del out,loss
                    if applied>=2:break
                if applied<2:raise RuntimeError('Fewer than two successful AMP updates')
                torch.cuda.synchronize()
                row={'dataset':dataset,'case':label,'status':'passed','epoch':epoch,'calls_per_window':last_calls,
                     'attempts':attempt+1,'updates':applied,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
                     'seconds':time.monotonic()-start}
            except BaseException as error:
                result['status']='failed';result['cases'].append({'dataset':dataset,'case':label,'error':repr(error)});write(a.output,result);raise
            result['cases'].append(row);write(a.output,result);print(row,flush=True)
            del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
        del batch;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(a.output,result)


if __name__=='__main__':main()
