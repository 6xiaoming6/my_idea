#!/usr/bin/env python3
"""Real batch32 GPU0 AMP checks, native and expensive TA/ST paths for all cells."""
import argparse,copy,gc,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
import torch
from run_b3_c3 import load,write
from run_first_validation import check_gpu0
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,move_batch_to_device
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import configure,common_initialization
from stmoe_imputer.utils.gpu_monitor import GPUMonitor


def run(suite, gpu=0, check_gpu=check_gpu0):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise RuntimeError('Incorrect physical GPU visibility')
    check_gpu([gpu]);torch.set_num_threads(2)
    js=load(suite/'resolved_jobs.json');result={'status':'running','cases':[]}
    for name,j in js.items():
        ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
        batch=move_batch_to_device(next(iter(build_loader(ds,j['config'],shuffle=False))),torch.device('cuda'))
        del ds
        for heavy in (False,True):
            label=name+('_TA_ST' if heavy else '_native')
            with GPUMonitor(suite/'acceptance'/label):
                c=copy.deepcopy(j['config']);configure(c);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
                model=DualBranchSTImputer.from_config(c).cuda().train();common_initialization(model,j)
                if heavy:
                    for router in model.main_branch.routers:
                        with torch.no_grad():
                            router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
                opt=build_optimizer(model,c);scaler=build_grad_scaler(torch.device('cuda'),c)
                torch.cuda.reset_peak_memory_stats();start=time.monotonic();updates=0
                try:
                    for attempt in range(24):
                        opt.zero_grad(set_to_none=True);calls=[]
                        hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in model.main_branch.routed_experts()]
                        try:
                            with torch.autocast('cuda',dtype=torch.float16):
                                out=model(batch);loss,_=compute_coe_loss(out,batch,c)
                        finally:
                            for hook in hooks:hook.remove()
                        count=sum(calls)/len(batch['m_f'])
                        if count!=8 or not bool(torch.isfinite(loss)):raise RuntimeError('Invalid native Top2 execution/forward')
                        old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                        grads=[p.grad for p in model.parameters() if p.grad is not None]
                        finite=all(bool(torch.isfinite(g).all()) for g in grads)
                        gate=[p.grad for p in model.main_branch.history_gate.parameters() if p.grad is not None]
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        scaler.step(opt);scaler.update()
                        if scaler.get_scale()>=old:
                            if not finite:raise RuntimeError('Nonfinite accepted gradient')
                            if c['model']['coe']['core_validation']['communication']=='conditional':
                                if not gate or not any(bool(g.abs().sum()>0) for g in gate):raise RuntimeError('Delta gate has no gradient')
                            elif gate:raise RuntimeError('Disabled gate received gradients')
                            fusion=model.main_branch.observation_fusion.weight.grad
                            if fusion is None or not bool(fusion.abs().sum()>0):raise RuntimeError('Input encoder unused')
                            if c['data']['input_scales']=='FMC':
                                d=model.main_branch.dim
                                if not all(bool(fusion[:,i*d:(i+1)*d].abs().sum()>0) for i in (1,2)):
                                    raise RuntimeError('Coarse input paths have no gradients')
                            updates+=1
                        del out,loss
                        if updates>=3:break
                    if updates<3:raise RuntimeError('Insufficient successful AMP updates')
                    torch.cuda.synchronize()
                    row={'case':label,'status':'passed','calls_per_window':count,'attempts':attempt+1,'updates':updates,
                         'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-start}
                    result['cases'].append(row);write(suite/'acceptance/preflight.json',result);print(row,flush=True)
                except BaseException as error:
                    result['status']='failed';result['cases'].append({'case':label,'error':repr(error)})
                    write(suite/'acceptance/preflight.json',result);raise
                finally:
                    del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
        del batch;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(suite/'acceptance/preflight.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();run(a.suite)
