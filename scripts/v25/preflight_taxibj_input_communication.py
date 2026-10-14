#!/usr/bin/env python3
"""Full-size GPU1 AMP checks for selected I groups, including consistency safety."""
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


def check_consistency(out,batch,cfg):
    spec=cfg['model']['coe'].get('observation_consistency',{})
    if not spec.get('enabled'):return
    if spec['mode']=='observed_loss':
        if any('x_'+scale+'_gt' in batch for scale in ('m','c')):
            raise RuntimeError('Observed-only control received full coarse labels')
        return
    if spec['mode']=='full_target_loss':
        for scale in ('m','c'):
            target=batch.get('x_'+scale+'_gt')
            if target is None or target.shape!=batch['m_'+scale].shape:
                raise RuntimeError('Full coarse supervision missing '+scale)
        return
    # Final completion is constrained only by visible observations. FP32 checks
    # tolerate accumulation roundoff under AMP, never a percent-level residual.
    final=out['x_hat_main'].float();visible=batch['m_f'].bool()
    if not bool(torch.isfinite(final).all()):raise RuntimeError('Nonfinite corrected completion')
    if not torch.equal(final[visible],batch['x_f_obs'].float()[visible]):
        raise RuntimeError('Consistency changed visible F')
    h,w=final.shape[-2:];constrained=torch.zeros_like(visible)
    for scale in ('m','c'):
        target=batch['x_'+scale+'_obs'].float();mask=batch['m_'+scale].bool()
        sh,sw=target.shape[-2:];rh,rw=h//sh,w//sw
        constrained|=mask.repeat_interleave(rh,-2).repeat_interleave(rw,-1)
        aggregate=final.reshape(*final.shape[:-2],sh,rh,sw,rw).sum(dim=(-3,-1))
        if cfg['data']['scales']['pooling_mode']=='mean':aggregate=aggregate/(rh*rw)
        if mask.any() and not torch.allclose(aggregate[mask],target[mask],atol=2e-3,rtol=2e-5):
            raise RuntimeError('Visible scale constraint violated '+scale)
    free=constrained&~visible
    if free.any() and float(final[free].min()) < -1e-6:
        raise RuntimeError('Negative missing-region projection')


def run(suite):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Physical GPU1 required')
    torch.set_num_threads(2);result={'status':'running','cases':[]}
    for n,j in load(suite/'resolved_jobs.json').items():
        cfg=j['config'];ds,_=build_datasets(cfg,j['sources']['train'],j['sources']['val'])
        batch=move_batch_to_device(next(iter(build_loader(ds,cfg,shuffle=False))),torch.device('cuda'));del ds
        for name in ['m','c']:batch['x_'+name+'_obs'].requires_grad_()
        for heavy in [False,True]:
            label=n+('_TA_ST' if heavy else '_native')
            with GPUMonitor(suite/'acceptance'/label):
                configure(cfg);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
                model=DualBranchSTImputer.from_config(cfg).cuda().train();common_initialization(model,j)
                if heavy:
                    for router in model.main_branch.routers:
                        with torch.no_grad():router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
                optimizer=build_optimizer(model,cfg);scaler=build_grad_scaler(torch.device('cuda'),cfg)
                torch.cuda.reset_peak_memory_stats();start=time.monotonic();updates=0
                try:
                    for attempt in range(24):
                        optimizer.zero_grad(set_to_none=True)
                        for name in ['m','c']:batch['x_'+name+'_obs'].grad=None
                        calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in model.main_branch.routed_experts()]
                        try:
                            with torch.autocast('cuda',dtype=torch.float16):out=model(batch);loss,_=compute_coe_loss(out,batch,cfg)
                        finally:
                            for hook in hooks:hook.remove()
                        if sum(calls)!=8*len(batch['m_f']) or not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss / non-native Top2')
                        exploration=cfg['model']['coe'].get('scale_delta_exploration',{})
                        if exploration.get('last_round_fine'):
                            choices=out['coe']['triscale_choices']
                            if not bool((choices[:,3]==0).all()):raise RuntimeError('Last-round F constraint violated')
                            if exploration.get('first_three_unique') and not bool((choices[:,:3].sort(1).values==choices.new_tensor([0,1,2])).all()):
                                raise RuntimeError('First three scales must be a permutation of F/M/C')
                        check_consistency(out,batch,cfg)
                        old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
                        if scaler.get_scale()>=old:
                            if not finite:raise RuntimeError('Nonfinite accepted gradient')
                            gate=[p.grad for p in model.main_branch.history_gate.parameters() if p.grad is not None]
                            if not gate or not any(bool(g.abs().sum()>0) for g in gate):raise RuntimeError('Difference gate unused')
                            fusion=model.main_branch.observation_fusion.weight.grad;dim=model.main_branch.dim
                            if fusion is None or not bool(fusion[:,:dim].abs().sum()>0):raise RuntimeError('F initialization unused')
                            if cfg['model']['coe']['initial_observation_scales']=='F':
                                if bool(fusion[:,dim:].any()):raise RuntimeError('M/C leaked into H0')
                            elif not all(bool(fusion[:,i*dim:(i+1)*dim].abs().sum()>0) for i in [1,2]):raise RuntimeError('FMC initialization unused')
                            for name in ['m','c']:
                                grad=batch['x_'+name+'_obs'].grad
                                if grad is None or not bool(grad[batch['m_'+name].bool()].abs().sum()>0):raise RuntimeError('Real scale unused '+name)
                                if bool(grad[~batch['m_'+name].bool()].any()):raise RuntimeError('Hidden input leaked '+name)
                            if hasattr(model.main_branch,'x_history'):
                                d=out['coe']['diagnostics']
                                if [float(d[f'x_step{i}_pool_size']) for i in range(1,5)]!=[0,1,2,3]:raise RuntimeError('Noncausal history pool')
                                if not bool(model.main_branch.x_history.inject[-1].weight.grad.abs().sum()>0):raise RuntimeError('History injection unused')
                            if hasattr(model.main_branch,'exploration'):
                                if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.main_branch.scale_heads.parameters()):raise RuntimeError('Free scale heads unused')
                            if hasattr(model.main_branch,'x_consistency'):
                                grads=[p.grad for p in model.main_branch.x_consistency.parameters() if p.requires_grad and p.grad is not None]
                                if not grads:raise RuntimeError('Consistency weights detached from task loss')
                            updates+=1
                        del out,loss
                        if updates>=3:break
                    if updates<3:raise RuntimeError('Insufficient AMP updates')
                    torch.cuda.synchronize();row={'case':label,'status':'passed','attempts':attempt+1,'updates':updates,'calls_per_window':8,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-start};result['cases'].append(row);print(row,flush=True)
                except BaseException as error:
                    result['status']='failed';result['cases'].append({'case':label,'error':repr(error)});write(suite/'acceptance/preflight.json',result);raise
                finally:
                    del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
            write(suite/'acceptance/preflight.json',result)
        del batch;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(suite/'acceptance/preflight.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();run(a.suite)
