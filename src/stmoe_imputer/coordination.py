"""Observed-value preserving, spatial-only coordination and label-side diagnostics."""
from collections import defaultdict
import torch


def blocks(x, factor):
    b,c,t,h,w=x.shape
    if h%factor or w%factor:
        raise ValueError('Spatial dimensions must be divisible by block factor')
    return x.reshape(b,c,t,h//factor,factor,w//factor,factor).permute(0,1,2,3,5,4,6).reshape(b,c,t,h//factor,w//factor,factor*factor)


def unblocks(x, factor):
    b,c,t,h,w,_=x.shape
    return x.reshape(b,c,t,h,w,factor,factor).permute(0,1,2,3,5,4,6).reshape(b,c,t,h*factor,w*factor)


def block_mean(x, selected, factor):
    q=blocks(selected.bool(),factor)
    count=q.sum(-1)
    mean=torch.where(q,blocks(x.float(),factor),0.).sum(-1)/count.clamp_min(1)
    return mean,count


def coordinate(pred, observed, mean, factor, strength=.1, scores=None):
    """FP32 correction; a region without missing points is an exact no-op."""
    missing=~observed.bool().expand_as(pred)
    q=blocks(missing,factor);count=q.sum(-1)
    p=blocks(pred.float(),factor)
    residual=count*mean.float()-torch.where(q,p,0.).sum(-1)
    if scores is None:
        weights=q.float()/count.clamp_min(1).unsqueeze(-1)
    else:
        logits=blocks(scores.float(),factor).masked_fill(~q,-torch.inf)
        logits=torch.where((count>0).unsqueeze(-1),logits,torch.zeros_like(logits))
        weights=logits.softmax(-1)*q
    delta=strength*weights*residual.unsqueeze(-1)
    # Do not round through FP16 here, and do not alter observed entries.
    result=torch.where(missing,pred.float()+unblocks(delta,factor),pred.float())
    entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
    return result,{'entropy':entropy,'valid':count>0,'weights':weights,'residual':residual}


def auxiliary_loss(outputs,batch):
    from .losses import supervision_mask
    info=outputs['coordination'];target=batch['x_f_gt']
    selected=supervision_mask(target,batch['m_f'],batch.get('target_mask'))
    total=outputs['x_hat_main'].sum()*0.;logs={}
    for label,factor in (('c',4),('m',2)):
        if label not in info['means']:continue
        target_mean,count=block_mean(target,selected,factor)
        # A prediction estimates ALL missing positions in a bin. Partially labelled
        # bins cannot supervise that quantity: require complete missing coverage.
        all_count=blocks(~batch['m_f'].bool().expand_as(target),factor).sum(-1)
        valid=(count>0)&(count==all_count)
        error=(info['means'][label].float()[valid]-target_mean[valid]).abs()
        loss=error.sum()/valid.sum().clamp_min(1)
        total=total+info['aux_weight']*loss
        logs['l_coord_'+label]=loss.detach()
        logs['coord_'+label+'_supervised_regions']=valid.sum().detach()
    logs['l_coord_weighted']=total.detach()
    return total,logs


def _pair():return [0.,0.]


class CoordinationMetrics:
    """Exact split-level sums/counts; labels enter only after model forward."""
    def __init__(self):self.totals=defaultdict(_pair)
    def add(self,key,values):
        self.totals[key][0]+=float(values.float().sum().cpu())
        self.totals[key][1]+=values.numel()
    @torch.no_grad()
    def update(self,outputs,batch):
        from .losses import supervision_mask
        from .data.diverse_masks import ALL_FAMILIES
        info=outputs.get('coordination')
        if info is None:return
        target=batch['x_f_gt'];q=supervision_mask(target,batch['m_f'],batch.get('target_mask'))
        for label,pred in info['stages'].items():
            diff=(pred.float()-torch.where(q,target.float(),0.))[q]
            self.add('coord_'+label+'_mae',diff.abs());self.add('coord_'+label+'_mse',diff.square())
            if 'mask_family' in batch:
                for fid in batch['mask_family'].unique().tolist():
                    sq=q&(batch['mask_family']==fid)[:,None,None,None,None]
                    d=(pred[sq].float()-target[sq].float())
                    prefix='coord_family_'+ALL_FAMILIES[int(fid)]+'_'+label
                    self.add(prefix+'_mae',d.abs());self.add(prefix+'_mse',d.square())
        for label,factor in (('c',4),('m',2)):
            truth,count=block_mean(target,q,factor);valid=count>0
            all_count=blocks(~batch['m_f'].bool().expand_as(target),factor).sum(-1)
            full=valid&(count==all_count)
            if label in info['means']:
                self.add('coord_'+label+'_head_region_mae',(info['means'][label].float()-truth).abs()[full])
            for stage,pred in info['stages'].items():
                mean,_=block_mean(pred,q,factor)
                self.add('coord_'+label+'_'+stage+'_region_mae',(mean-truth).abs()[full])
                residual=blocks(pred.float()-torch.where(q,target.float(),0.),factor)-(mean-truth).unsqueeze(-1)
                self.add('coord_'+label+'_'+stage+'_demeaned_mae',residual[blocks(q,factor)].abs())
        raw=info['stages']['raw'];final=outputs['x_hat_main']
        self.add('coord_correction_abs',(final.float()-raw.float()).abs()[q])
        obs=batch['m_f'].bool().expand_as(raw)
        self.add('coord_observed_change',(final.float()-raw.float()).abs()[obs])
        for label,record in info['allocations'].items():
            self.add('coord_'+label+'_allocation_entropy',record['entropy'][record['valid']])
    def merge(self,other):
        for k,(v,n) in other.totals.items():self.totals[k][0]+=v;self.totals[k][1]+=n
    def compute(self):
        result={}
        for k,(v,n) in self.totals.items():
            if n:
                result[k[:-4]+'_rmse' if k.endswith('_mse') else k]=(v/n)**.5 if k.endswith('_mse') else v/n
        return result
