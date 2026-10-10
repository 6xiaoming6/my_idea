"""Opt-in X01-X10 controls on the frozen CMFF backbone."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

from .core_validation_coe import CoreValidationCoE
from ..utils.deterministic import pool, resize


def zero_last(module):
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


class HistoryReader(nn.Module):
    def __init__(self, channels, dim, mode):
        super().__init__()
        self.mode = mode
        self.inject = zero_last(nn.Sequential(nn.Conv3d(2*channels,32,1),nn.GELU(),nn.Conv3d(32,dim,1)))
        if mode == 'adaptive':
            self.score = zero_last(nn.Sequential(nn.Conv3d(3*channels+2,16,1),nn.GELU(),nn.Conv3d(16,1,1)))

    def read(self, history, mask, scale):
        if not history:
            raise ValueError('Empty history must bypass reader')
        if self.mode == 'last':
            weights = mask.new_zeros((len(mask),len(history),1,*mask.shape[2:]))
            weights[:, -1] = 1
        elif self.mode == 'mean':
            weights = mask.new_full((len(mask),len(history),1,*mask.shape[2:]),1/len(history))
        else:
            latest = history[-1]/scale
            coverage = mask.mean(1,keepdim=True)
            logits=[]
            for i,pred in enumerate(history):
                age = coverage.new_full(coverage.shape,(len(history)-i)/3)
                features=torch.cat((pred/scale,latest,(pred/scale-latest).abs(),age,coverage),1)
                logits.append(self.score(features).float())
            weights=torch.stack(logits,1).softmax(1)
        result=(torch.stack(history,1)*weights).sum(1)
        return result,weights

    def forward(self, h, history, mask, scale):
        read,weights=self.read(history,mask,scale)
        rms=h.detach().float().square().mean((2,3,4),keepdim=True).sqrt()
        delta=.1*rms*self.inject(torch.cat((read/scale,mask),1)).float().tanh()
        delta=delta*(1-mask).mean(1,keepdim=True)
        return delta.to(h.dtype),weights


class TriPlaneBranch(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode=mode
        self.down=nn.Conv3d(dim,16,1)
        self.planes=nn.ModuleList([nn.Sequential(nn.Conv2d(16,16,3,padding=1,groups=16),
            nn.GELU(),nn.Conv2d(16,16,1)) for _ in range(3)])
        self.up=nn.Conv3d(16,dim,1)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)

    def forward(self, x, coverage):
        q=coverage.float().mean(1,keepdim=True)
        z=self.down(x);views=[];supports=[]
        for axis,network in zip((4,3,2),self.planes):
            count=q.sum(axis)
            value=(z.float()*q).sum(axis)/count.clamp_min(1e-8)
            value=torch.where(count>0,value,torch.zeros_like(value))
            support=F.avg_pool2d(q.mean(axis),3,stride=1,padding=1,count_include_pad=False)
            feature=network(value)*(support>0).to(value.dtype)
            views.append(feature.unsqueeze(axis).expand(-1,-1,*x.shape[2:]))
            supports.append(support.unsqueeze(axis).expand(-1,-1,*x.shape[2:]))
        support=torch.stack(supports,1)
        evidence=(support>0).float() if self.mode=='uniform' else support
        weights=evidence/evidence.sum(1,keepdim=True).clamp_min(1e-8)
        fused=(torch.stack(views,1)*weights).sum(1)
        valid=support.sum(1)>0
        delta=.1*self.up(fused)*valid.to(fused.dtype)
        stats={f'plane_{label}_weight':weights[:,i].detach().mean() for i,label in enumerate(('TH','TW','HW'))}
        stats.update({f'plane_{label}_coverage':support[:,i].detach().mean() for i,label in enumerate(('TH','TW','HW'))})
        stats['plane_delta_abs']=delta.detach().float().abs().mean()
        return delta,stats


class TriPlaneST(nn.Module):
    def __init__(self, old, dim, mode):
        super().__init__()
        self.network=old.network  # Preserve every existing ST parameter/key.
        self.plane=TriPlaneBranch(2*dim,mode)
        self.last_stats={}

    def forward(self, x, coverage):
        h=self.network[:3](x)
        delta,self.last_stats=self.plane(h,coverage)
        local=self.network[3](h)
        return self.network[4:](local+delta.to(local.dtype))


class HistoryPlaneInteractionCoE(CoreValidationCoE):
    @classmethod
    def from_config(cls,cfg):
        coe=cfg['model']['coe']
        if coe.get('post_fusion_ffn',{}).get('enabled'):
            raise ValueError('X series excludes post-fusion FFN')
        m=super().from_config(cfg)
        m.x_spec=copy.deepcopy(coe['history_plane_interaction'])
        spec=m.x_spec
        if set(spec)!={'enabled','history','residual','plane','interaction'} or spec['enabled'] is not True:
            raise ValueError('Invalid X-series specification')
        if (spec['history'] not in ('none','last','mean','adaptive') or spec['plane'] not in ('none','uniform','coverage')
                or spec['interaction'] not in ('none','concat','product') or type(spec['residual']) is not bool):
            raise ValueError('Invalid X-series mode')
        if sum((spec['history']!='none',spec['residual'],spec['plane']!='none',spec['interaction']!='none'))>1:
            raise ValueError('X series isolates extensions; no combinations')
        if m.core_spec['path']!='CMFF' or m.core_spec['communication'] not in ('none','conditional') or m.expert_sharing!='shared':
            raise ValueError('X series requires shared CMFF none/conditional')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg.get('seed',7))+105001)
            if spec['history']!='none':m.x_history=HistoryReader(m.c_in,m.dim,spec['history'])
            torch.random.default_generator.manual_seed(int(cfg.get('seed',7))+105002)
            if spec['plane']!='none':m.pattern_experts['ST']=TriPlaneST(m.pattern_experts['ST'],m.dim,spec['plane'])
            torch.random.default_generator.manual_seed(int(cfg.get('seed',7))+105003)
            if spec['interaction']!='none':
                m.x_interaction=zero_last(nn.Sequential(nn.Conv3d(2*m.dim,32,1),nn.GELU(),nn.Conv3d(32,m.dim,1)))
        return m

    def _initialize(self,x,mask):
        c=super()._initialize(x,mask)
        c['x_pool']=[]
        observed=torch.where(c['mask'].bool(),c['x'].float().square(),0.)
        c['x_scale']=(observed.sum((2,3,4),keepdim=True)/c['mask'].sum((2,3,4),keepdim=True).clamp_min(1)).sqrt().clamp_min(1).detach()
        return c

    def _memory(self,c):
        corrected,route,gate=super()._memory(c)
        c['x_stats']={'pool_size':c['h'].new_tensor(len(c['x_pool']))}
        if self.x_spec['history']!='none' and c['x_pool']:
            delta,weights=self.x_history(c['h'],c['x_pool'],c['mask'],c['x_scale'])
            corrected=corrected+delta
            c['x_stats']['pool_delta_abs']=delta.detach().float().abs().mean()
            for i in range(len(c['x_pool'])):c['x_stats'][f'pool_weight_P{i+1}']=weights[:,i].detach().mean()
        return corrected,route,gate

    def _native_dispatch(self,z,weights,coverage,step,stats):
        """Identical weighted accumulation order; cache two outputs if needed."""
        out=torch.zeros_like(z)
        pair=self.x_spec['interaction']!='none'
        ids=weights.detach().topk(2,1).indices.sort(1).values if pair else None
        first=torch.zeros_like(z) if pair else None
        second=torch.zeros_like(z) if pair else None
        for index,expert in enumerate(self.routed_experts(step)):
            selected=(weights[:,index].detach()!=0).nonzero().flatten()
            if not selected.numel():continue
            x=z.index_select(0,selected)
            if isinstance(expert,TriPlaneST):
                update=expert(x,coverage.index_select(0,selected))
                stats.update(expert.last_stats)
                stats['ST_selected_fraction']=z.new_tensor(len(selected)/len(z))
            else:update=expert(x)
            coeff=weights.index_select(0,selected)[:,index]
            weighted=update*coeff[:,None,None,None,None].to(update.dtype)
            out=out.index_add(0,selected,weighted.to(out.dtype))
            if pair:
                is_first=ids.index_select(0,selected)[:,0]==index
                left=selected[is_first];right=selected[~is_first]
                first=first.index_copy(0,left,update[is_first].to(first.dtype))
                second=second.index_copy(0,right,update[~is_first].to(second.dtype))
        if pair:
            a=F.layer_norm(first.movedim(1,-1),(self.dim,)).movedim(-1,1)
            b=F.layer_norm(second.movedim(1,-1),(self.dim,)).movedim(-1,1)
            features=torch.cat((a,b),1) if self.x_spec['interaction']=='concat' else torch.cat((a*b,a-b),1)
            rms=out.detach().float().square().mean((2,3,4),keepdim=True).sqrt()
            delta=.1*rms*self.x_interaction(features).float().tanh()
            stats['interaction_abs']=delta.detach().abs().mean()
            per_sample=delta.detach().abs().mean((1,2,3,4))
            for i in range(self.num_experts):
                for j in range(i+1,self.num_experts):
                    chosen=(ids[:,0]==i)&(ids[:,1]==j)
                    stats[f'pair_{i}_{j}_fraction']=chosen.float().mean()
                    # Unconditional moment; divide by fraction for conditional magnitude.
                    stats[f'pair_{i}_{j}_interaction']=torch.where(chosen,per_sample,0.).mean()
            out=out+delta.to(out.dtype)
        return out

    def _execute(self,c,corrected,weights,scale_weights,step):
        if self.x_spec['plane']=='none' and self.x_spec['interaction']=='none':
            out,message,bank=super()._execute(c,corrected,weights,scale_weights,step)
        else:
            sid=(2,1,0,0)[step];factor=(1,2,4)[sid]
            if not bool((scale_weights[:,sid]==1).all()):raise ValueError('CMFF path changed')
            h=corrected;mask=c['mask'];completion=c['completion'];support=c['support'];pos=c['pos']
            if factor>1:
                coverage=pool(mask.float(),factor)
                values=pool(torch.where(mask.bool(),completion,0.).float(),factor)/coverage.clamp_min(1e-8)
                completion=torch.where(coverage>0,values,pool(completion,factor));mask=coverage
                h=pool(h,factor);support=pool(support,factor);pos=pool(pos,factor)
            z=self.state_norm(self.state_projection(torch.cat((h,completion,mask,support,pos),1)))
            c['w_stats']['adapter_abs']=z.new_zeros(())
            c['x_stats']['ST_selected_fraction']=z.new_zeros(())
            out=self._native_dispatch(z,weights,mask,step,c['x_stats'])
            if factor>1:out=resize(out,c['h'].shape[-2:])
            message,bank=None,[]
        if self.x_spec['residual']:out=c['h']+out
        return out,message,bank

    def _round(self,c,step,*args,**kwargs):
        new,row=super()._round(c,step,*args,**kwargs)
        self._x_stats.append(c['x_stats'])
        if self.x_spec['history']!='none':
            pred=torch.where(c['mask'].bool(),c['x'],row['prediction']).detach()
            new['x_pool']=c['x_pool']+[pred]
        return new,row

    def forward(self,*args,**kwargs):
        self._x_stats=[]
        try:
            out=super().forward(*args,**kwargs)
            out['coe']['x_progress_diagnostics']=True
            for step,stats in enumerate(self._x_stats,1):
                for key,value in stats.items():out['coe']['diagnostics'][f'x_step{step}_{key}']=value.detach()
            return out
        finally:
            del self._x_stats
