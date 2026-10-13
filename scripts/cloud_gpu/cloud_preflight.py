#!/usr/bin/env python3
"""Full-batch cloud preflight; native and forced heavy experts, without saving weights."""
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
    device=torch.device(load(suite/'plan.json')['device'])
    torch.set_num_threads(2);result={'status':'running','cases':[]}
    for n,j in load(suite/'resolved_jobs.json').items():
        cfg=j['config'];ds,_=build_datasets(cfg,j['sources']['train'],j['sources']['val'])
        batch=move_batch_to_device(next(iter(build_loader(ds,cfg,shuffle=False))),device);del ds
        for name in ['m','c']:batch['x_'+name+'_obs'].requires_grad_()
        for heavy in [False,True]:
            label=n+('_TA_ST' if heavy else '_native')
            with GPUMonitor(suite/'acceptance'/label,enabled=device.type=='cuda'):
                configure(cfg);torch.manual_seed(cfg['seed'])
                if device.type=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
                model=DualBranchSTImputer.from_config(cfg).to(device).train();common_initialization(model,j)
                if heavy:
                    for router in model.main_branch.routers:
                        with torch.no_grad():router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
                optimizer=build_optimizer(model,cfg);scaler=build_grad_scaler(device,cfg)
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                start=time.monotonic();updates=0
                coe=cfg['model']['coe'];spec=coe.get('scale_delta_exploration',{});expected=sum(spec.get('top_k_per_step',[2]*4))
                try:
                    for attempt in range(24):
                        optimizer.zero_grad(set_to_none=True)
                        for name in ['m','c']:batch['x_'+name+'_obs'].grad=None
                        calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in model.main_branch.routed_experts()]
                        try:
                            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):out=model(batch);loss,_=compute_coe_loss(out,batch,cfg)
                        finally:
                            for hook in hooks:hook.remove()
                        if sum(calls)!=expected*len(batch['m_f']) or not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss / incorrect native expert calls')
                        old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
                        if scaler.get_scale()>=old:
                            if not finite:raise RuntimeError('Nonfinite accepted gradient')
                            gate=[p.grad for p in model.main_branch.history_gate.parameters() if p.grad is not None]
                            if coe['core_validation']['communication']!='none' and (not gate or not any(bool(g.abs().sum()>0) for g in gate)):raise RuntimeError('Difference gate unused')
                            fusion=model.main_branch.observation_fusion.weight.grad;dim=model.main_branch.dim
                            if fusion is None or not bool(fusion[:,:dim].abs().sum()>0):raise RuntimeError('F initialization unused')
                            if coe['initial_observation_scales']=='F' or cfg['data']['input_scales']=='F':
                                if bool(fusion[:,dim:].any()):raise RuntimeError('M/C leaked into H0')
                            elif not all(bool(fusion[:,i*dim:(i+1)*dim].abs().sum()>0) for i in [1,2]):raise RuntimeError('FMC initialization unused')
                            for name in ['m','c']:
                                grad=batch['x_'+name+'_obs'].grad
                                if cfg['data']['input_scales']!='F' and (grad is None or not bool(grad[batch['m_'+name].bool()].abs().sum()>0)):raise RuntimeError('Real scale unused '+name)
                                if grad is not None and bool(grad[~batch['m_'+name].bool()].any()):raise RuntimeError('Hidden input leaked '+name)
                            if hasattr(model.main_branch,'x_history'):
                                d=out['coe']['diagnostics']
                                if [float(d[f'x_step{i}_pool_size']) for i in range(1,5)]!=[0,1,2,3]:raise RuntimeError('Noncausal history pool')
                                if not bool(model.main_branch.x_history.inject[-1].weight.grad.abs().sum()>0):raise RuntimeError('History injection unused')
                            if spec.get('path')=='free':
                                if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.main_branch.scale_heads.parameters()):raise RuntimeError('Free scale heads unused')
                            updates+=1
                        del out,loss
                        if updates>=3:break
                    if updates<3:raise RuntimeError('Insufficient AMP updates')
                    if device.type=='cuda':torch.cuda.synchronize()
                    row={'case':label,'status':'passed','attempts':attempt+1,'updates':updates,'calls_per_window':expected,'peak_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,'seconds':time.monotonic()-start};result['cases'].append(row);print(row,flush=True)
                except BaseException as error:
                    result['status']='failed';result['cases'].append({'case':label,'error':repr(error)});write(suite/'acceptance/preflight.json',result);raise
                finally:
                    del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
            write(suite/'acceptance/preflight.json',result)
        del batch;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(suite/'acceptance/preflight.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();run(a.suite)
