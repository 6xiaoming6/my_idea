#!/usr/bin/env python3
"""Real batch32 AMP acceptance on GPU0, including expensive expert paths."""
import argparse,copy,gc,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import torch
from run_core_validation import jobs
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
    check_gpu_idle([0]);torch.set_num_threads(2);js=jobs(seeds=(7,));result={'status':'running','cases':[]}
    configure(next(iter(js.values()))['config'])
    for dataset in ('taxibj','bikenyc'):
        j=js[f'{dataset}_K04_seed7'];ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
        batch=move_batch_to_device(next(iter(build_loader(ds,j['config'],shuffle=False))),torch.device('cuda'))
        cases=[('fine_attention','K03',True),('cmff','K04',False)]
        if dataset=='taxibj':cases += [('independent_attention','K06',True),('unconditional','K07',False)]
        for label,method,heavy in cases:
            c=copy.deepcopy(js[f'{dataset}_{method}_seed7']['config']);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            model=DualBranchSTImputer.from_config(c).cuda().train()
            if heavy:
                for r in model.main_branch.routers:
                    with torch.no_grad():r[-1].weight.zero_();r[-1].bias.zero_();r[-1].bias[4:6]=10
            opt=build_optimizer(model,c);scaler=build_grad_scaler(torch.device('cuda'),c)
            torch.cuda.reset_peak_memory_stats();started=time.monotonic();updates=0;calls_per_window=0
            try:
                for attempt in range(18):
                    opt.zero_grad(set_to_none=True);calls=[];unique={id(e):e for step in range(4) for e in model.main_branch.routed_experts(step)}
                    hooks=[e.register_forward_pre_hook(lambda e,args:calls.append(len(args[0]))) for e in unique.values()]
                    with torch.autocast('cuda',dtype=torch.float16):out=model(batch);loss,_=compute_coe_loss(out,batch,c)
                    for hook in hooks:hook.remove()
                    calls_per_window=sum(calls)/len(batch['m_f'])
                    if calls_per_window!=8 or not torch.isfinite(loss):raise RuntimeError('Invalid sparse execution/forward')
                    old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                    grads=[p.grad for p in model.main_branch.history_gate.parameters() if p.grad is not None]
                    valid=bool(grads) and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
                    if scaler.get_scale()>=old:
                        if not valid:raise RuntimeError('No finite nonzero gate gradient')
                        updates+=1
                    del out,loss
                    if updates>=2:break
                if updates<2:raise RuntimeError('Insufficient successful AMP updates')
                torch.cuda.synchronize()
                row={'dataset':dataset,'case':label,'method':method,'status':'passed','calls_per_window':calls_per_window,
                     'attempts':attempt+1,'updates':updates,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-started}
            except BaseException as error:
                result['status']='failed';result['cases'].append({'dataset':dataset,'case':label,'error':repr(error)});write(a.output,result);raise
            result['cases'].append(row);write(a.output,result);print(row,flush=True)
            del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
        del batch,ds;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(a.output,result)
if __name__=='__main__':main()
