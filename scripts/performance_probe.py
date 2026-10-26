#!/usr/bin/env python3
"""Isolated, strict CUDA AMP trace of an explicitly selected source tree."""
import argparse,copy,gc,json,os,random,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--features',default='none');p.add_argument('--compact',action='store_true');a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
sys.path[:0]=[str(a.source/'src'),str(a.source/'scripts')]
import numpy as np,torch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,build_scheduler
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import configure,state_hash,pool
from stmoe_imputer.utils.gpu_monitor import GPUMonitor
cfg=json.loads((a.source/'configs/i37_taxibj.json').read_text());features=('skip_redundant_init','reuse_fine_support','no_grad_intermediate','compact_outputs','cache_constants','pack_scalar_logs','skip_last_history')
selected=set(features) if a.features=='all' else set(a.features.split(',')) if a.features not in ('none','diagnostics') else set()
if a.features!='none':cfg['train']['performance']=dict(enabled=True,**{k:k in selected for k in features})
configure(cfg);torch.set_num_threads(2);ds,_=build_datasets(cfg,a.data_dir/'taxibj_train.npz',a.data_dir/'taxibj_val.npz')
cases=['native','partial20','FFFF','MMMF','CCCF','TA_ST','empty','full'];results={};a.output.parent.mkdir(exist_ok=True,parents=True)
with GPUMonitor(a.output.parent):
 for case in cases:
  random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
  n=20 if case=='partial20' else 32;items=[ds[i] for i in range(n)];batch={k:torch.stack([s[k] for s in items]).cuda() for k in items[0]}
  if case in ('empty','full'):
   for key,f in zip('fmc',(1,2,4)):
    batch['m_'+key].fill_(float(case=='full'))
    batch['x_'+key+'_obs']=pool(batch['x_f_gt'],f)*f*f if case=='full' else torch.full_like(batch['x_'+key+'_obs'],float('nan'))
  m=DualBranchSTImputer.from_config(cfg).cuda().train();m.main_branch.set_routing_epoch(1)
  if case in ('FFFF','MMMF','CCCF','TA_ST'):
   sid={'FFFF':0,'MMMF':1,'CCCF':2,'TA_ST':0}[case]
   with torch.no_grad():
    for h in m.main_branch.scale_heads:h[-1].weight.zero_();h[-1].bias.zero_();h[-1].bias[sid]=10
    if case=='TA_ST':
     for h in m.main_branch.routers:h[-1].weight.zero_();h[-1].bias.zero_();h[-1].bias[4]=10;h[-1].bias[5]=9
  optimizer=build_optimizer(m,cfg);scaler=build_grad_scaler(torch.device('cuda'),cfg);scheduler=build_scheduler(optimizer,cfg);rows=[];accepted=0
  def update():
   optimizer.zero_grad(set_to_none=True);calls=[];hooks=[e.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0]))) for e in m.main_branch.routed_experts()]
   try:
    with torch.autocast('cuda',dtype=torch.float16):
     out=m(batch,collect_diagnostics=not a.compact) if a.features!='none' else m(batch)
     loss,_=compute_main_stage_loss(out,batch,cfg);loss=loss+out['x_hat_main'].sum()*0.
   finally:
    for hook in hooks:hook.remove()
   assert sum(calls)==8*n;assert bool((out['coe']['triscale_choices'][:,-1]==0).all());assert bool(torch.isfinite(loss))
   if not torch.equal(out['x_hat_main'][batch['m_f'].bool()],batch['x_f_obs'][batch['m_f'].bool()]):raise RuntimeError('Visible F changed')
   prediction=state_hash({k:out[k] for k in ['x_hat_main','x_hat_raw_consistency']})
   route=state_hash({k:out['coe'][k] for k in ['selected_experts','triscale_choices','route_weights']})
   scaler.scale(loss).backward();scaler.unscale_(optimizer)
   grad=state_hash({n:p.grad for n,p in m.named_parameters()})
   torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['train']['grad_clip_norm'])
   clipped=state_hash({n:p.grad for n,p in m.named_parameters()})
   previous=scaler.get_scale();scaler.step(optimizer);scaler.update();ok=scaler.get_scale()>=previous
   row=dict(prediction=prediction,route=route,loss=float(loss.detach()),gradients=grad,clipped_gradients=clipped,model=state_hash(m.state_dict()),optimizer=state_hash(optimizer.state_dict()),scheduler=state_hash(scheduler.state_dict()),scaler=state_hash(scaler.state_dict()),rng=state_hash(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())),accepted=ok,expert_calls=sum(calls))
   return row
  checkpoint_path=a.output.parent/('resume_'+case+'.pth')
  checkpoint_index=None
  for attempt in range(16):
   row=update();rows.append(row);accepted+=int(row['accepted'])
   if accepted==1 and checkpoint_index is None:
    torch.save(dict(model=m.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state()),checkpoint_path)
    checkpoint_index=len(rows)
   if accepted>=3:break
  if accepted<3:raise RuntimeError('Fewer than 3 accepted AMP updates '+case)
  saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
  m.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler']);torch.set_rng_state(saved['cpu']);torch.cuda.set_rng_state(saved['cuda']);random.setstate(saved['python']);np.random.set_state(saved['numpy'])
  for expected in rows[checkpoint_index:]:
   actual=update()
   if actual!=expected:raise RuntimeError('AMP full-checkpoint continuation differs '+case)
  checkpoint_path.unlink()
  results[case]=rows;print(case,'passed',len(rows),'attempts',flush=True)
  del m,optimizer,scaler,scheduler,batch;gc.collect();torch.cuda.empty_cache()
a.output.write_text(json.dumps(dict(status='passed',source=str(a.source),features=sorted(selected),compact=a.compact,cases=results),indent=2)+'\n')
