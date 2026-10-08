#!/usr/bin/env python3
"""Real TaxiBJ batch32 AMP/gradient/VRAM check, GPU0 only, no training receipts."""
import gc,os,sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import torch
from run_coordination_exploration import jobs
from run_b3_c3 import write
from run_team_accept_v4 import check_gpu_idle
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler,move_batch_to_device
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Set CUDA_VISIBLE_DEVICES=0')
    check_gpu_idle([0]);torch.set_num_threads(2);js=jobs();j=js['V1'];ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
    batch=move_batch_to_device(next(iter(build_loader(ds,j['config'],shuffle=False))),torch.device('cuda'))
    if len(batch['m_f'])!=32:raise RuntimeError('Expected batch32')
    result={'status':'running','gpu':0,'device':torch.cuda.get_device_name(),'batch_shape':list(batch['x_f_obs'].shape),'cases':[]}
    for n,j in js.items():
        torch.manual_seed(7);m=DualBranchSTImputer.from_config(j['config']).cuda().train();opt=build_optimizer(m,j['config']);scaler=build_grad_scaler(torch.device('cuda'),j['config'])
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();applied=0;gradients={}
        try:
            for attempt in range(16):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):out=m(batch);loss,_=compute_coe_loss(out,batch,j['config'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
                old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
                for label in ('region_heads','allocation_head','history_gate'):
                    ps=[p.grad for name,p in m.main_branch.named_parameters() if name.startswith(label) and p.grad is not None]
                    if ps:gradients[label]=sum(float(g.detach().float().square().sum()) for g in ps)**.5
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.);scaler.step(opt);scaler.update();applied+=int(scaler.get_scale()>=old)
                del loss,out
                if applied>=2:break
            if applied<2 or not all(torch.isfinite(p).all() for p in m.parameters()):raise RuntimeError('No two successful finite updates')
            if n!='V1' and not all(0<v<float('inf') for k,v in gradients.items() if k in ('region_heads','allocation_head')):raise RuntimeError('Invalid head gradient')
            torch.cuda.synchronize()
            entry={'variant':n,'status':'passed','attempts':attempt+1,'optimizer_updates':applied,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.monotonic()-start,'gradient_norms':gradients}
        except Exception as e:
            result['status']='failed';result['cases'].append({'variant':n,'error':repr(e)});write(a.output,result);raise
        result['cases'].append(entry);write(a.output,result);print(entry,flush=True)
        del m,opt,scaler;gc.collect();torch.cuda.empty_cache()
    result['status']='passed';write(a.output,result)
if __name__=='__main__':main()
