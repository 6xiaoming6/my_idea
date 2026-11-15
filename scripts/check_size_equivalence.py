#!/usr/bin/env python3
"""Check full-size AMP compatibility before reusing the interrupted 16x16 run."""
import os,sys,copy,json,argparse,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from release_utils import load,write
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import build_optimizer,build_grad_scaler
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import configure,state_hash
from stmoe_imputer.utils.gpu_monitor import GPUMonitor

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 data=a.project/'data/TaxiBJ/v25_observed_multiscale/73fe84df49ca05de';old=a.project/'outputs/v25/experiments/latent_space/20261114_seed7/runs/L01'
 base=load(ROOT/'configs/i37_performance.json');new32=copy.deepcopy(base);new32['model']['architecture']='v25_spatial_size_coe';new32['model']['coe']['latent_size']={'lh':32,'lw':32,'projection':'fixed'}
 old16=load(old/'config.json');new16=load(ROOT/'configs/latent_size/GS03.json')
 ca,cb=copy.deepcopy(old16),copy.deepcopy(new16)
 ca['model'].pop('architecture');cb['model'].pop('architecture');ca['model']['coe'].pop('latent_space');cb['model']['coe'].pop('latent_size');assert ca==cb
 saved=torch.load(old/'checkpoints/last.pth',map_location='cpu',weights_only=False)
 configure(base);ds,_=build_datasets(base,data/'taxibj_train.npz',data/'taxibj_val.npz');batch={k:v.cuda() for k,v in next(iter(build_loader(ds,base,False))).items()}
 rows=[]
 a.output.mkdir(parents=True,exist_ok=True)
 with GPUMonitor(a.output,enabled=True):
  for label,left,right,state in [('32x32_vs_I37',base,new32,None),('16x16_vs_original_L01_epoch80',old16,new16,saved)]:
   records=[]
   for cfg in (left,right):
    torch.manual_seed(7);m=DualBranchSTImputer.from_config(cfg).cuda().train();op=build_optimizer(m,cfg);sc=build_grad_scaler(torch.device('cuda'),cfg)
    if state:
     m.load_state_dict(state['model']);op.load_state_dict(copy.deepcopy(state['optimizer']));sc.load_state_dict(state['scaler'])
    with torch.autocast('cuda',dtype=torch.float16):
     out=m(batch);loss,_=compute_main_stage_loss(out,batch,cfg)
    sc.scale(loss).backward();sc.unscale_(op);grad=state_hash({n:p.grad for n,p in m.named_parameters()});torch.nn.utils.clip_grad_norm_(m.parameters(),1.);sc.step(op);sc.update()
    records.append(dict(raw=out['x_hat_raw_consistency'].detach().cpu(),pred=out['x_hat_main'].detach().cpu(),loss=loss.detach().cpu(),grad=grad,model=state_hash(m.state_dict()),model_values={n:v.detach().cpu().clone() for n,v in m.state_dict().items()},optimizer=state_hash(op.state_dict()),scaler=state_hash(sc.state_dict()),cpu=state_hash(torch.get_rng_state()),cuda=state_hash(torch.cuda.get_rng_state())))
    del m,op,sc,out,loss;gc.collect();torch.cuda.empty_cache()
   for n,v in records[0]['model_values'].items():
    vv=records[1]['model_values'][n]
    if not torch.equal(v,vv):print('DIFF',label,n,'max',float((v.float()-vv.float()).abs().max()),flush=True)
   for key in records[0]:
    if key=='model_values':continue
    x,y=records[0][key],records[1][key]
    if torch.is_tensor(x):torch.testing.assert_close(x,y,atol=0,rtol=0)
    else:assert x==y,(label,key)
   rows.append(dict(comparison=label,status='bitwise_equal'))
 write(a.output/'result.json',dict(status='passed',batch_size=32,physical_gpu=1,reused_epoch=saved['epoch'],comparisons=rows))
 print(rows)
if __name__=='__main__':main()
