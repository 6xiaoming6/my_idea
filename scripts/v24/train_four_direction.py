#!/usr/bin/env python3
"""Single-device, epoch-atomic training with complete RNG/optimizer continuation."""
from __future__ import annotations
import argparse, json, os, random, signal, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from stmoe_imputer.data import build_datasets,build_test_dataset,build_loader
from stmoe_imputer.engine import build_optimizer,build_scheduler,build_grad_scaler,train_one_epoch,evaluate
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import save_checkpoint,snapshot_model_state
from run_b3_c3 import load,write,digest


def rng_snapshot(loaders,device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
            'loaders':{n:l.generator.get_state() if l.generator is not None else None for n,l in loaders.items()},
            'datasets':{n:l.dataset.generator.get_state() for n,l in loaders.items() if hasattr(l.dataset,'generator')}}


def restore_rng(state,loaders,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'].cpu(),device)
    for n,v in state['loaders'].items():
        if v is not None:loaders[n].generator.set_state(v.cpu())
    for n,v in state['datasets'].items():loaders[n].dataset.generator.set_state(v.cpu())


def train(job,run_dir,result_file,device='cuda',stop_after=None):
    cfg=job['config'];run=Path(run_dir);run.mkdir(parents=True,exist_ok=True);log=run/'logs';log.mkdir(exist_ok=True)
    cp=run/'checkpoints';cp.mkdir(exist_ok=True)
    if (run/'config.json').exists() and digest(load(run/'config.json'))!=digest(cfg):raise ValueError('Resume config changed')
    if Path(result_file).exists():
        r=load(result_file)
        if r.get('status')=='finished' and r['config_sha256']==digest(cfg):return r
        raise ValueError('Invalid completion receipt')
    write(run/'config.json',cfg)
    d=torch.device(device);seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if d.type=='cuda':torch.cuda.manual_seed_all(seed)
    train_ds,val_ds=build_datasets(cfg,job['sources']['train'],job['sources']['val'])
    test_ds=build_test_dataset(cfg,job['sources']['test'])
    loaders={k:build_loader(ds,cfg,shuffle=k=='train') for k,ds in [('train',train_ds),('val',val_ds),('test',test_ds)]}
    model=DualBranchSTImputer.from_config(cfg).to(d);optimizer=build_optimizer(model,cfg)
    scheduler=build_scheduler(optimizer,cfg);scaler=build_grad_scaler(d,cfg)
    start_epoch=1;best=float('inf');best_epoch=0;best_state=None;elapsed=0.;history=[]
    if (cp/'last.pth').exists():
        saved=torch.load(cp/'last.pth',map_location='cpu',weights_only=False)
        if digest(saved['config'])!=digest(cfg):raise ValueError('Checkpoint/config mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        state=saved['training_state'];start_epoch=saved['epoch']+1;best=state['best_val_mae'];best_epoch=state['best_epoch']
        best_state=state['best_model'];elapsed=state['elapsed'];history=state['history']
        restore_rng(saved['rng_states'][0],loaders,d);del saved
        # The checkpoint is the commit record, so interrupted log appends are rebuilt.
        (log/'metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
        for split in ('train','val'):
            (log/f'{split}.log').write_text(''.join(json.dumps({'epoch':r['epoch'],'metrics':r[split]})+'\n' for r in history if r.get(split)))
    write(run/'training_metadata.json',{'config_sha256':digest(cfg),'samples':{k:len(v.dataset) for k,v in loaders.items()},
        'steps':{k:len(v) for k,v in loaders.items()},'total_params':sum(p.numel() for p in model.parameters()),
        'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'device':device})
    for epoch in range(start_epoch,cfg['train']['epochs']+1):
        clock=time.perf_counter()
        if d.type=='cuda':torch.cuda.reset_peak_memory_stats(d)
        tl=train_one_epoch(model,loaders['train'],optimizer,d,cfg,epoch,scaler=scaler)
        training_seconds=time.perf_counter()-clock;vl=None
        if epoch%cfg['train']['val_epoch']==0 or epoch==cfg['train']['epochs']:
            vl=evaluate(model,loaders['val'],d,cfg,epoch=epoch,show_progress=False)
        improved=vl is not None and vl['mae']<best
        if improved:best=vl['mae'];best_epoch=epoch;best_state=snapshot_model_state(model)
        lr=optimizer.param_groups[0]['lr'];scheduler.step();tl['lr']=lr
        seconds=time.perf_counter()-clock;elapsed+=seconds
        row={'epoch':epoch,'train':tl,'val':vl,'is_best':improved,'perf':{'epoch_time_sec':seconds,'train_time_sec':training_seconds,
             'peak_memory_gb':torch.cuda.max_memory_allocated(d)/2**30 if d.type=='cuda' else 0.}}
        # Nonfinite diagnostic values remain explicit strings; main metrics must be finite.
        for split in (tl,vl):
            if split:
                if not all(np.isfinite(split[k]) for k in ('loss','mae','rmse')):raise RuntimeError('Nonfinite primary metric')
                for k,v in split.items():
                    if isinstance(v,float) and not np.isfinite(v):split[k]=str(v)
        history.append(row)
        state={'next_epoch':epoch+1,'best_epoch':best_epoch,'best_val_mae':best,'best_model':best_state,'elapsed':elapsed,'history':history}
        rng=[rng_snapshot(loaders,d)]
        save_checkpoint(cp/'last.pth',model,optimizer,epoch,tl,cfg,scheduler=scheduler,scaler=scaler,rng_states=rng,training_state=state)
        if improved:save_checkpoint(cp/'best.pth',model,optimizer,epoch,vl,cfg,scheduler=scheduler,scaler=scaler,rng_states=rng)
        with (log/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        for split in ('train','val'):
            if row.get(split):
                with (log/f'{split}.log').open('a') as f:f.write(json.dumps({'epoch':epoch,'metrics':row[split]})+'\n')
        if stop_after and epoch>=stop_after and epoch<cfg['train']['epochs']:return {'status':'paused_for_test','epoch':epoch}
    if best_state is None:raise RuntimeError('No validation-best state')
    model.load_state_dict(best_state)
    # Repair a checkpoint/log write interrupted immediately after last.pth committed.
    best_file=cp/'best.pth'
    if not best_file.exists() or torch.load(best_file,map_location='cpu',weights_only=False)['epoch']!=best_epoch:
        save_checkpoint(best_file,model,None,best_epoch,{'mae':best},cfg)
    test=evaluate(model,loaders['test'],d,cfg,epoch=best_epoch,show_progress=False)
    write(log/'test.json',test)
    receipt={'status':'finished','variant':job['variant'],'run_dir':str(run.resolve()),'config_sha256':digest(cfg),
             'completed_epochs':cfg['train']['epochs'],'best_epoch':best_epoch,'best_val_mae':best,'total_time_sec':elapsed,'test':test}
    write(result_file,receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--result',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *args:sys.exit(143))
    train(load(a.job),a.run_dir,a.result,a.device,a.stop_after)
if __name__=='__main__':main()
