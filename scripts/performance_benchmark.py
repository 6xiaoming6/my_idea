#!/usr/bin/env python3
"""Real-data train_one_epoch benchmark in an isolated source process, GPU1 only."""
import argparse,json,os,sys,time,gc,random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--features',default='none');p.add_argument('--weights',type=Path,required=True);a=p.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');sys.path[:0]=[str(a.source/'src'),str(a.source/'scripts')]
import torch,numpy as np
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.engine import train_one_epoch,build_optimizer,build_grad_scaler
from stmoe_imputer.backbone_training import TrainingContext
from stmoe_imputer.utils.deterministic import configure,state_hash
from stmoe_imputer.utils.gpu_monitor import GPUMonitor
cfg=json.loads((a.source/'configs/i37_taxibj.json').read_text());features=('skip_redundant_init','reuse_fine_support','no_grad_intermediate','compact_outputs','cache_constants','pack_scalar_logs','skip_last_history');chosen=set(features) if a.features=='all' else set(a.features.split(',')) if a.features not in ('none','diagnostics') else set()
if a.features!='none':cfg['train']['performance']=dict(enabled=True,**{k:k in chosen for k in features})
configure(cfg);torch.set_num_threads(2);rows=[];a.output.parent.mkdir(exist_ok=True,parents=True)
with GPUMonitor(a.output.parent):
 for trial in range(3):
  random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
  ds,_=build_datasets(cfg,a.data_dir/'taxibj_train.npz',a.data_dir/'taxibj_val.npz')
  m=DualBranchSTImputer.from_config(cfg).cuda().train();m.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['model'])
  optimizer=build_optimizer(m,cfg);scaler=build_grad_scaler(torch.device('cuda'),cfg);context=TrainingContext(m,cfg)
  # Warm and measure whole batches with the exact legacy engine, including
  # observation mask generation, host diagnostics and TrainingContext zero loss.
  warm=build_loader(torch.utils.data.Subset(ds,range(5*32)),cfg,shuffle=False)
  train_one_epoch(m,warm,optimizer,torch.device('cuda'),cfg,1,scaler=scaler,show_progress=False,training_context=context)
  loader=build_loader(torch.utils.data.Subset(ds,range(20*32)),cfg,shuffle=False)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  metrics=train_one_epoch(m,loader,optimizer,torch.device('cuda'),cfg,1,scaler=scaler,show_progress=False,training_context=context)
  torch.cuda.synchronize();seconds=time.perf_counter()-start
  row=dict(trial=trial,steps=20,samples=640,seconds=seconds,seconds_per_step=seconds/20,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,model_sha256=state_hash(m.state_dict()),optimizer_sha256=state_hash(optimizer.state_dict()),rng_sha256=state_hash(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())),primary={k:metrics[k] for k in ['loss','mae','rmse','wape','metric_missing_count']})
  rows.append(row);print(row,flush=True);del m,optimizer,scaler,context,loader,warm,ds;gc.collect();torch.cuda.empty_cache()
a.output.write_text(json.dumps(dict(status='passed',source=str(a.source),features=sorted(chosen),trials=rows),indent=2)+'\n')
