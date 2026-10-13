#!/usr/bin/env python3
"""Physical GPU1, batch32 AMP and real worst-cost sparse execution checks."""
import argparse,copy,gc,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
import torch
from run_b3_c3 import load,write
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,move_batch_to_device
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import configure,common_initialization
from stmoe_imputer.utils.gpu_monitor import GPUMonitor


def run(suite):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Physical GPU1 only')
    torch.set_num_threads(2);js=load(suite/'resolved_jobs.json')
    result={'status':'running','gpu':1,'cases':[]}
    for name,j in js.items():
        ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
        batch=move_batch_to_device(next(iter(build_loader(ds,j['config'],shuffle=False))),torch.device('cuda'));del ds
        for heavy in (False,True):
            label=name+('_heavy' if heavy else '_native')
            print('Preflight '+label,flush=True)
            with GPUMonitor(suite/'acceptance'/label):
                c=copy.deepcopy(j['config']);configure(c);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
                model=DualBranchSTImputer.from_config(c).cuda().train();common_initialization(model,j)
                m=model.main_branch
                if heavy:
                    with torch.no_grad():
                        for router in m.routers:
                            router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
                        if m.exploration['path']=='free':
                            for h in m.scale_heads:h[-1].weight.zero_();h[-1].bias.zero_();h[-1].bias[0]=10
                opt=build_optimizer(model,c);scaler=build_grad_scaler(torch.device('cuda'),c)
                torch.cuda.reset_peak_memory_stats();started=time.monotonic();updates=0
                try:
                    for attempt in range(32):
                        opt.zero_grad(set_to_none=True);calls=[]
                        hooks=[e.register_forward_pre_hook(lambda e,a:calls.append((len(a[0]),a[0].shape[-1]))) for e in m.routed_experts()]
                        try:
                            with torch.autocast('cuda',dtype=torch.float16):out=model(batch);loss,_=compute_coe_loss(out,batch,c)
                        finally:
                            for h in hooks:h.remove()
                        count=sum(n for n,w in calls)/len(batch['m_f'])
                        if count!=sum(m.step_top_k) or not bool(torch.isfinite(loss)):raise RuntimeError('Invalid sparse forward')
                        if not bool((out['coe']['triscale_executed'].sum(2)==1).all()):raise RuntimeError('Multiple scales executed')
                        old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                        finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
                        if scaler.get_scale()>=old:
                            if not finite:raise RuntimeError('Nonfinite accepted gradient')
                            gate=[p.grad for p in m.history_gate.parameters() if p.grad is not None]
                            if m.delta_steps:
                                if not gate or not any(bool(g.abs().sum()>0) for g in gate):raise RuntimeError('Enabled gate has no gradient')
                            elif gate:raise RuntimeError('Disabled gate received gradient')
                            if m.exploration['path']=='free' and not any(p.grad is not None and bool(p.grad.abs().sum()>0) for h in m.scale_heads for p in h.parameters()):
                                raise RuntimeError('Scale router has no gradient')
                            fusion=m.observation_fusion.weight.grad;d=m.dim
                            if fusion is None or not all(bool(fusion[:,i*d:(i+1)*d].abs().sum()>0) for i in range(3)):
                                raise RuntimeError('A real observation branch has no gradient')
                            updates+=1
                        del out,loss
                        if updates>=3:break
                    if updates<3:raise RuntimeError('Insufficient successful AMP updates')
                    torch.cuda.synchronize()
                    row={'case':label,'status':'passed','calls_per_window':count,'attempts':attempt+1,'updates':updates,
                         'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-started,'native_widths':sorted(set(w for n,w in calls))}
                    result['cases'].append(row);write(suite/'acceptance/preflight.json',result);print(row,flush=True)
                except BaseException as error:
                    result['status']='failed';result['cases'].append({'case':label,'error':repr(error)})
                    write(suite/'acceptance/preflight.json',result);raise
                finally:
                    del model,m,opt,scaler;gc.collect();torch.cuda.empty_cache()
        del batch;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(suite/'acceptance/preflight.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();run(a.suite)
