#!/usr/bin/env python3
"""GPU0-only real batch32 AMP acceptance; never creates training receipts."""
import argparse,copy,gc,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import torch
from run_id_priority_exploration import static_jobs
from run_b3_c3 import write
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,move_batch_to_device
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--dataset',default='taxibj');a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Set CUDA_VISIBLE_DEVICES=0; GPU1 is not authorized')
 torch.set_num_threads(2);jobs=static_jobs(a.dataset);j=jobs['U01'];ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
 batch=move_batch_to_device(next(iter(build_loader(ds,j['config'],shuffle=False))),torch.device('cuda'))
 if len(batch['m_f'])!=32:raise RuntimeError('Acceptance requires real batch32')
 cases=[('U01','U01',None,False),('U05','U05',None,False),('U07','U07',None,False),('U09_all_fine','U09',None,True),('U13','U13',None,False),('U13_all_FM','U13',None,True),('U11','U11',None,False),('U15','U15',None,False),('U16','U16',None,False),('U17_innovation_teacher','U11','innovation',True),('U17_router_identity','U16','innovation_router',True)]
 result={'status':'running','device':torch.cuda.get_device_name(),'physical_gpu':0,'batch_size':32,'cases':[]}
 for label,key,communication,fine in cases:
  cfg=copy.deepcopy(jobs[key]['config'])
  if communication:cfg['model']['coe']['id_priority'].update(communication=communication,bound=.1)
  torch.manual_seed(7);model=DualBranchSTImputer.from_config(cfg).cuda().train();back=model.main_branch;back.set_routing_epoch(21)
  if fine:
   for h in back.scale_heads:h[-1].bias.data.copy_(torch.tensor([.001,0.,-10.] if back.scale_policy=='top2' else [10.,-10.,-10.],device='cuda'))
  opt=build_optimizer(model,cfg);scaler=build_grad_scaler(torch.device('cuda'),cfg);torch.cuda.reset_peak_memory_stats();start=time.monotonic();applied=0;probes=[]
  try:
   for attempt in range(16):
    opt.zero_grad(set_to_none=True);back.batch_clock.fill_(20);back.teacher_clock.fill_(attempt%len(back.learnable_steps()));back.prepare_training_batch(batch)
    with torch.autocast('cuda'):
     out=model(batch);loss,_=compute_coe_loss(out,batch,cfg)
     if 'four_probe' in out:
      aux,probe=back.candidate_loss(batch,out);loss=loss+.1*aux;probes.append(probe)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
    before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
    applied+=int(scaler.get_scale()>=before)
    del out,loss
    if applied>=2 and (back.teacher=='none' or attempt>=3):break
   if applied<2:raise RuntimeError('No two finite AMP updates within 16 attempts')
   if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite trained parameters')
   torch.cuda.synchronize();case={'name':label,'status':'passed','peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.monotonic()-start,'attempts':attempt+1,'optimizer_updates':applied,'probes':probes}
  except Exception as error:
   result['status']='failed';result['cases'].append({'name':label,'error':repr(error),'peak_gib':torch.cuda.max_memory_allocated()/2**30});write(a.output,result);raise
  result['cases'].append(case);write(a.output,result);print(json.dumps(case),flush=True)
  del model,back,opt,scaler;gc.collect();torch.cuda.empty_cache()
 result['status']='passed';write(a.output,result)
if __name__=='__main__':main()
