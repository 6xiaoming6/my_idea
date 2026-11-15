#!/usr/bin/env python3
"""GPU1-only batch32 acceptance checks; no training-budget run is created."""
import argparse,copy,gc,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_scheduler,build_grad_scaler
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import configure,state_hash
from stmoe_imputer.utils.gpu_monitor import GPUMonitor
from release_utils import load,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
 cfg=load(ROOT/'configs/latent_size/GS01.json');configure(cfg)
 ds,_=build_datasets(cfg,a.data/'taxibj_train.npz',a.data/'taxibj_val.npz')
 batch={k:v.to('cuda') for k,v in next(iter(build_loader(ds,cfg,False))).items()}
 rows=[]
 with GPUMonitor(a.output,enabled=True):
  for i in range(1,11):
   name=f'GS{i:02}';cfg=load(ROOT/f'configs/latent_size/{name}.json');torch.manual_seed(7)
   m=DualBranchSTImputer.from_config(cfg).cuda().train();op=build_optimizer(m,cfg);sc=build_grad_scaler(torch.device('cuda'),cfg)
   scheduler=build_scheduler(op,cfg)
   initial_hash=state_hash(m.state_dict());calls=[]
   hooks=[e.register_forward_hook(lambda mod,inp,out:calls.append(len(inp[0]))) for e in m.main_branch.pattern_experts.values()]
   def step(forced=False):
    op.zero_grad(set_to_none=True);calls.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    b=dict(batch)
    if forced:
     b['forced_pair_step']=3;b['forced_pair_indices']=torch.tensor([[4,5]],device='cuda').expand(32,2) # TA and ST at full F
    with torch.autocast('cuda',dtype=torch.float16):
     out=m(b,collect_diagnostics=True);loss,logs=compute_main_stage_loss(out,b,cfg)
    sc.scale(loss).backward();sc.unscale_(op)
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert sum(calls)==32*8
    assert out['x_hat_main'].shape==batch['x_f_obs'].shape
    assert m.main_branch.decoder[-1].weight.grad is not None
    if i>=7:assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if '.to_latent.' in n or '.to_native.' in n)
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);sc.step(op);sc.update();scheduler.step();torch.cuda.synchronize()
    return dict(loss=float(loss),seconds=time.perf_counter()-start,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
   native=step();forced=step(True)
   for h in hooks:h.remove()
   # Exact continuation with full model/optimizer/scheduler/scaler and CUDA/CPU RNG.
   if i in (4,6,10):
    snapshot=copy.deepcopy(dict(model=m.state_dict(),optimizer=op.state_dict(),scheduler=scheduler.state_dict(),scaler=sc.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
    def update():
     op.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.float16):
      out=m(batch,collect_diagnostics=False);loss,_=compute_main_stage_loss(out,batch,cfg)
     sc.scale(loss).backward();sc.unscale_(op);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);sc.step(op);sc.update();scheduler.step()
     return state_hash(dict(model=m.state_dict(),optimizer=op.state_dict(),scheduler=scheduler.state_dict(),scaler=sc.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
    continuous=update()
    torch.save(snapshot,a.output/f'{name}_resume.pth')
    saved=torch.load(a.output/f'{name}_resume.pth',weights_only=False)
    m.load_state_dict(saved['model']);op.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);sc.load_state_dict(saved['scaler']);torch.set_rng_state(saved['cpu'].cpu());torch.cuda.set_rng_state(saved['cuda'].cpu())
    assert update()==continuous
    del snapshot,saved
   rows.append(dict(name=name,status='passed',native=native,forced_TA_ST=forced,initial_hash=initial_hash,params=sum(p.numel() for p in m.parameters()),trainable_params=sum(p.numel() for p in m.parameters() if p.requires_grad)))
   write(a.output/'results.json',dict(status='running',physical_gpu=1,rows=rows));print(json.dumps(rows[-1]),flush=True)
   del step,m,op,sc,scheduler;gc.collect();torch.cuda.empty_cache()
 write(a.output/'results.json',dict(status='passed',physical_gpu=1,batch_size=32,rows=rows))
if __name__=='__main__':main()
