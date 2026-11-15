import os,sys,json,types
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='2'
p=Path(__file__).resolve().parents[1];sys.path[:0]=[str(p/'src'),str(p/'scripts')]
import torch
from release_utils import load
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import configure
from stmoe_imputer.utils.gpu_monitor import GPUMonitor
torch.set_num_threads(2)
cfg=None
import argparse
a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--config',type=Path,required=True);args=a.parse_args()
cfg=load(args.config);configure(cfg)
data=args.data
ds,_=build_datasets(cfg,data/'taxibj_train.npz',data/'taxibj_val.npz');batch={k:v.cuda() for k,v in next(iter(build_loader(ds,cfg,False))).items()}
torch.manual_seed(7);m=DualBranchSTImputer.from_config(cfg).cuda().train();main=m.main_branch
original_scales=main._scales;original_pairs=main._pairs
main._scales=lambda c,f,s,force=None,probe=False:original_scales(c,f,s,torch.zeros(len(f),device=f.device,dtype=torch.long),probe)
main._pairs=lambda c,f,l,s,force=None,probe=False:original_pairs(c,f,l,s,torch.tensor([[4,5]],device=f.device).expand(len(f),2),probe)
folder=args.output
folder.mkdir(parents=True,exist_ok=True)
with GPUMonitor(folder,enabled=True):
 with torch.autocast('cuda',dtype=torch.float16):
  out=m(batch);loss,_=compute_main_stage_loss(out,batch,cfg)
 loss.backward();torch.cuda.synchronize()
 assert torch.isfinite(loss)
 result=dict(status='passed',batch32=True,paths=out['coe']['triscale_choices'][0].tolist(),experts=out['coe']['selected_experts'][0].tolist(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
 (folder/'result.json').write_text(json.dumps(result,indent=2));print(result)
