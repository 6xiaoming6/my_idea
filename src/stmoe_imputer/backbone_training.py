"""W-series label-side losses, nested-mask views, EMA and replay audit state."""
import copy,hashlib
import numpy as np
import torch
from .losses import supervision_mask
from .coordination import block_mean
from .models.backbone_exploration_coe import box_sum
from .data.diverse_masks import _field
from .utils.deterministic import state_hash


def average(values,selected):
    return values[selected].sum()/selected.sum().clamp_min(1)


def structural_loss(outputs,batch):
    spec=outputs['w_spec'];pred=outputs['x_hat_main'].float();target=batch['x_f_gt'].float()
    q=supervision_mask(target,batch['m_f'],batch.get('target_mask'));zero=pred.sum()*0.;total=zero;logs={}
    if spec.get('observed_loss',False):
        obs=batch['m_f'].bool().expand_as(target)&torch.isfinite(batch['x_f_obs'])
        loss=torch.stack([average((p.float()-batch['x_f_obs'].float()).abs(),obs) for p in outputs['coe']['predictions']]).mean()
        total=total+.02*loss;logs['l_w_observed']=loss.detach()
    kind=spec.get('structure','none')
    if kind in ('region','region_count'):
        for label,factor in (('c',4),('m',2)):
            a,count=block_mean(pred,q,factor);b,_=block_mean(target,q,factor);error=(a-b).abs();valid=count>0
            loss=average(error,valid) if kind=='region' else (error*count).sum()/count.sum().clamp_min(1)
            total=total+.025*loss;logs['l_w_region_'+label]=loss.detach()
    if kind in ('temporal','spatial'):
        comp=torch.where(batch['m_f'].bool(),batch['x_f_obs'].float(),pred)
        valid=torch.isfinite(target)
        if 'target_mask' in batch:valid=valid&batch['target_mask'].bool().expand_as(target)
        numerator=zero;count=zero.detach()
        for axis in ((2,) if kind=='temporal' else (3,4)):
            a=[slice(None)]*5;b=list(a);a[axis]=slice(1,None);b[axis]=slice(None,-1);a=tuple(a);b=tuple(b)
            selected=valid[a]&valid[b]&(q[a]|q[b])
            diff=(comp[a][selected]-comp[b][selected])-(target[a][selected]-target[b][selected])
            numerator=numerator+diff.abs().sum();count=count+selected.sum()
        loss=numerator/count.clamp_min(1);total=total+.05*loss;logs['l_w_'+kind]=loss.detach()
    logs['l_w_structure_weighted']=total.detach()
    return total,logs


def nested_batch(batch,rng):
    n=min(8,len(batch['m_f']));small={k:v[:n] for k,v in batch.items()}
    masks=small['m_f'].detach().cpu().numpy().copy();families=('random_point','node_outage','temporal_gap','spatial_region')
    for i in range(n):
        shape=tuple(masks.shape[-3:]);family=families[int(rng.integers(4))];raw=_field(shape,family,rng).ravel()
        order=np.lexsort((rng.random(raw.size),raw))
        for channel in range(masks.shape[1]):
            old=masks[i,channel].reshape(-1);missing=int((old==0).sum());need=max(0,round(old.size*.5)-missing)
            candidates=order[old[order]>0];old[candidates[:need]]=0
    mask=torch.as_tensor(masks,device=batch['m_f'].device,dtype=batch['m_f'].dtype)
    small['m_f']=mask;small['x_f_obs']=torch.where(mask.bool(),small['x_f_obs'],0.)
    return small


def reliability(teacher,batch,selected):
    mask=batch['m_f'].float().expand_as(teacher);obs=mask.bool();x=batch['x_f_obs'].float()
    error=torch.where(obs,(teacher.float()-x).abs(),0.).detach()
    count=box_sum(mask);window=error.sum((1,2,3,4),keepdim=True)/mask.sum((1,2,3,4),keepdim=True).clamp_min(1)
    local=torch.where(count>0,box_sum(error)/count.clamp_min(1),window)
    weight=torch.exp(-local/window.clamp_min(1e-6))*selected
    denom=weight.sum((1,2,3,4),keepdim=True)
    # Underflow -> uniform on the eligible cells, never NaN or no supervision.
    weight=torch.where(denom>1e-30,weight/denom.clamp_min(1e-30)*selected.sum((1,2,3,4),keepdim=True),selected.float())
    return weight.detach()


class TrainingContext:
    def __init__(self,model,cfg):
        self.spec=cfg['model']['coe']['backbone_exploration'];self.view=self.spec.get('view','none')
        self.rng=np.random.default_rng(cfg.get('seed',7)+83003)
        self.teacher=copy.deepcopy(model).eval().requires_grad_(False) if self.view=='ema' else None
        self.successful_updates=0;self.audit=[];self.begin_epoch()
    def begin_epoch(self):
        self.order=hashlib.sha256();self.extra_order=hashlib.sha256();self.student_windows=0;self.teacher_windows=0;self.extra_batches=0
    def record_batch(self,batch):
        self.order.update(state_hash({k:batch[k] for k in ('x_f_gt','m_f')}).encode())
    def extra_loss(self,model,batch,outputs):
        zero=outputs['x_hat_main'].sum()*0.
        if self.view=='none':return zero,{}
        small=nested_batch(batch,self.rng);n=len(small['m_f']);self.extra_order.update(state_hash(small['m_f']).encode())
        other=model(small);q=supervision_mask(small['x_f_gt'],small['m_f'],small.get('target_mask'))
        auxiliary=average((other['x_hat_main'].float()-torch.nan_to_num(small['x_f_gt'].float())).abs(),q)
        total=.25*auxiliary;logs={'l_w_second_view':auxiliary.detach()}
        self.student_windows+=n;self.extra_batches+=1
        if self.view!='augment':
            original={k:v[:n] for k,v in batch.items()}
            if self.teacher is not None:
                with torch.no_grad():teacher=self.teacher(original)['x_hat_main']
                self.teacher_windows+=n
            else:teacher=outputs['x_hat_main'][:n].detach()
            common=q&supervision_mask(original['x_f_gt'],original['m_f'],original.get('target_mask'))
            weight=reliability(teacher,original,common) if self.view=='reliability' else torch.ones_like(teacher)
            consistency=average((other['x_hat_main'].float()-teacher.float()).abs()*weight,common)
            total=total+.05*consistency;logs['l_w_consistency']=consistency.detach()
            logs['w_teacher_student_gap']=average((other['x_hat_main'].float()-teacher.float()).abs(),common).detach()
            logs['w_consistency_weight_max']=weight.max().detach()
        logs['l_w_view_weighted']=total.detach()
        return total,logs
    @torch.no_grad()
    def after_update(self,model):
        self.successful_updates+=1
        if self.teacher is not None:
            for dest,src in zip(self.teacher.parameters(),model.parameters()):dest.mul_(.99).add_(src,alpha=.01)
            for dest,src in zip(self.teacher.buffers(),model.buffers()):dest.copy_(src)
    def state_dict(self):
        return {'rng':self.rng.bit_generator.state,'teacher':self.teacher.state_dict() if self.teacher is not None else None,
                'successful_updates':self.successful_updates,'audit':self.audit}
    def load_state_dict(self,state):
        self.rng.bit_generator.state=state['rng'];self.successful_updates=state['successful_updates'];self.audit=state['audit']
        if self.teacher is not None:self.teacher.load_state_dict(state['teacher'])
    def finish_epoch(self,epoch,model,optimizer,scaler,scheduler,logs):
        metrics={sp:{k:v[k] for k in ('loss','mae','rmse') if k in v} if v else None for sp,v in logs.items()}
        entry={'epoch':epoch,'data_order_sha256':self.order.hexdigest(),'extra_order_sha256':self.extra_order.hexdigest(),
               'model_sha256':state_hash(model.state_dict()),'optimizer_sha256':state_hash(optimizer.state_dict()),
               'scaler_sha256':state_hash(scaler.state_dict()),'scheduler_sha256':state_hash(scheduler.state_dict()),
               'metrics':metrics,'extra_student_windows':self.student_windows,'extra_teacher_windows':self.teacher_windows,'extra_batches':self.extra_batches}
        self.audit.append(entry);return entry
