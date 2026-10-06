#!/usr/bin/env python3
"""Single GPU0 acceptance only. Does not write experiment training receipts."""
import argparse,copy,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import torch
from run_four_direction_exploration import static_jobs
from run_b3_c3 import write
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,move_batch_to_device
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--dataset',default='taxibj');a=p.parse_args()
 torch.set_num_threads(2);jobs=static_jobs(a.dataset);ds,_=build_datasets(jobs['M01']['config'],jobs['M01']['sources']['train'],jobs['M01']['sources']['val'])
 batch=move_batch_to_device(next(iter(build_loader(ds,jobs['M01']['config'],shuffle=False))),torch.device('cuda'))
 cases=[('M01',None),('M11',None),('M12',None),('S02','all_fine'),('S02',None),('S06','all_fine'),('S09',None),('S04',None),('P03',None),('P04',None),('P05',None),('P06',None),('G12_gru_bank','combine_bank'),('G12_message_teacher','combine_teacher')]
 result={'status':'running','device':torch.cuda.get_device_name(),'batch_size':len(batch['m_f']),'cases':[]}
 for label,override in cases:
  key='S06' if override=='combine_bank' else 'S04' if override=='combine_teacher' else label
  cfg=copy.deepcopy(jobs[key]['config'])
  if override=='combine_bank':cfg['model']['coe']['four_direction']['memory']='gru'
  if override=='combine_teacher':cfg['model']['coe']['four_direction']['memory']='message'
  torch.manual_seed(7);model=DualBranchSTImputer.from_config(cfg).cuda().train();back=model.main_branch
  if override in ('all_fine','combine_bank','combine_teacher'):
   for head in back.scale_heads:head[-1].bias.data.copy_(torch.tensor([10.,-10.,-10.],device='cuda'))
  opt=build_optimizer(model,cfg);scaler=build_grad_scaler(torch.device('cuda'),cfg);torch.cuda.reset_peak_memory_stats();start=time.monotonic();applied=0
  try:
   for attempt in range(16):
    opt.zero_grad(set_to_none=True);back.set_routing_epoch(1)
    # Exercise a distinct intervention round at successive AMP attempts.
    back.batch_clock.fill_(20*(attempt%4));back.prepare_training_batch(batch)
    with torch.autocast('cuda'):
     out=model(batch);loss,_=compute_coe_loss(out,batch,cfg)
     if 'four_probe' in out:
      aux,probe=back.candidate_loss(batch,out);loss=loss+.1*aux
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
    before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
    applied+=int(scaler.get_scale()>=before)
    del out,loss
    if applied>=2:break
   if applied<2:raise RuntimeError('No two finite AMP optimizer updates within 16 attempts')
   torch.cuda.synchronize();case={'name':label,'status':'passed','peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-start,'attempts':attempt+1,'optimizer_updates':applied}
  except Exception as error:
   result['status']='failed';result['cases'].append({'name':label,'error':repr(error),'peak_gib':torch.cuda.max_memory_allocated()/2**30});write(a.output,result);raise
  result['cases'].append(case);write(a.output,result);print(json.dumps(case),flush=True)
  del model,back,opt,scaler;gc.collect();torch.cuda.empty_cache()
 result['status']='passed';write(a.output,result)
if __name__=='__main__':main()
