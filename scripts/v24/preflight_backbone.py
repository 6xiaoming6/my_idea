#!/usr/bin/env python3
"""W GPU0 real batch32 deterministic AMP acceptance, independent of queue receipts."""
import copy,gc,os,sys,time,argparse,tempfile
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import torch
from run_backbone_exploration import jobs
from run_b3_c3 import write
from run_team_accept_v4 import check_gpu_idle
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,move_batch_to_device
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.backbone_training import TrainingContext
from stmoe_imputer.utils.deterministic import configure,state_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Only GPU0 accepted')
    check_gpu_idle([0]);torch.set_num_threads(2);js=jobs();configure(js['W01']['config'])
    ds,_=build_datasets(js['W01']['config'],js['W01']['sources']['train'],js['W01']['sources']['val'])
    batch=move_batch_to_device(next(iter(build_loader(ds,js['W01']['config'],shuffle=False))),torch.device('cuda'))
    if len(batch['m_f'])!=32:raise RuntimeError('Expected real batch32')
    names=('W01','W01_REPLAY','W10','W15','W17','W21','COMBINATION','HEAVY_TA_ST')
    result={'status':'running','gpu':0,'device':torch.cuda.get_device_name(),'batch_shape':list(batch['x_f_obs'].shape),'cases':[]};replay=None
    for n in names:
        cfg=copy.deepcopy(js[n if n in js else 'W01']['config'])
        if n in ('COMBINATION','HEAVY_TA_ST'):cfg['model']['coe']['backbone_exploration'].update(adapter='coverage_rank',feedback='both',observed_loss=True,view='ema',structure='region')
        torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        m=DualBranchSTImputer.from_config(cfg).cuda().train()
        if n=='HEAVY_TA_ST':
            # Force the two attention-heavy experts in preflight only.
            for router in m.main_branch.routers:
                with torch.no_grad():router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
        opt=build_optimizer(m,cfg);scaler=build_grad_scaler(torch.device('cuda'),cfg);context=TrainingContext(m,cfg)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();applied=0;gradients={};trajectory=[]
        try:
            for attempt in range(16):
                opt.zero_grad(set_to_none=True);m.main_branch.prepare_training_batch(batch)
                with torch.autocast('cuda',dtype=torch.float16):
                    out=m(batch);loss,_=compute_coe_loss(out,batch,cfg);extra,_=context.extra_loss(m,batch,out);loss=loss+extra
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
                old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                for label in ('adapter','feedback','history_gate'):
                    gs=[p.grad for name,p in m.main_branch.named_parameters() if name.startswith(label) and p.grad is not None]
                    if gs:gradients[label]=sum(float(g.detach().float().square().sum()) for g in gs)**.5
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.);scaler.step(opt);scaler.update();success=scaler.get_scale()>=old
                if success:applied+=1;context.after_update(m)
                trajectory.append({'loss':float(loss.detach()),'model':state_hash(m.state_dict()),'optimizer':state_hash(opt.state_dict()),'scaler':state_hash(scaler.state_dict())})
                del loss,extra,out
                if applied>=2:break
            if applied<2 or not all(torch.isfinite(p).all() for p in m.parameters()):raise RuntimeError('No two finite successful updates')
            if not all(0<v<float('inf') for v in gradients.values()):raise RuntimeError('Nonfinite or zero module gradient')
            if n=='W01':replay=trajectory
            if n=='W01_REPLAY' and trajectory!=replay:raise RuntimeError('GPU trajectories are not exactly reproducible')
            torch.cuda.synchronize();entry={'variant':n,'status':'passed','attempts':attempt+1,'optimizer_updates':applied,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.monotonic()-start,'gradient_norms':gradients,'trajectory':trajectory}
        except Exception as error:
            result['status']='failed';result['cases'].append({'variant':n,'error':repr(error)});write(a.output,result);raise
        result['cases'].append(entry);write(a.output,result);print({k:v for k,v in entry.items() if k!='trajectory'},flush=True)
        del m,opt,scaler,context;gc.collect();torch.cuda.empty_cache()
    # Exercise real CUDA optimizer/AMP/RNG continuation, not just identical forward calls.
    from train_four_direction import train
    with tempfile.TemporaryDirectory(prefix='w_cuda_resume_') as td:
        root=Path(td);x=batch['x_f_gt'].detach().cpu().numpy();source=root/'small.npz'
        np.savez(source,x_f_gt=np.concatenate((x,x),axis=0))
        saved_runs={}
        for label,variant,pause in (('base','W01',False),('base_resume','W02',True),('ema','W21',False),('ema_resume','W21',True)):
            job=copy.deepcopy(js[variant]);job['sources']={k:str(source) for k in ('train','val','test')}
            job['common_initialization']=str(root/'common_initialization.pth')
            job['config']['train'].update(epochs=2,val_epoch=1)
            job['config']['train']['scheduler']['total_epochs']=2
            run=root/label;receipt=root/(label+'.json')
            if pause:train(job,run,receipt,'cuda',stop_after=1)
            train(job,run,receipt,'cuda')
            cp=torch.load(run/'checkpoints/last.pth',map_location='cpu',weights_only=False)
            saved_runs[label]={k:state_hash(cp[k]) for k in ('model','optimizer','scheduler','scaler','rng_states')}
            saved_runs[label]['w_training']=state_hash(cp['training_state']['w_training'])
            del cp;gc.collect();torch.cuda.empty_cache()
        if saved_runs['base']!=saved_runs['base_resume'] or saved_runs['ema']!=saved_runs['ema_resume']:
            result['status']='failed';result['resume_audit']=saved_runs;write(a.output,result)
            raise RuntimeError('GPU full training and interrupted/resumed training differ')
        result['resume_audit']=saved_runs
    result['status']='passed';result['exact_gpu_replay']=True;result['exact_gpu_resume']=True;write(a.output,result)
if __name__=='__main__':main()
