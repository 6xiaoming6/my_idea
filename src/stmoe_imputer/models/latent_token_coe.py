"""Grid-independent latent slots, preserving the temporal axis.

Public latent layout is [B,T,K,D]. The inherited CoE context stores a view
[B,D,T,K,1] only for channelwise pointwise statistics; K is never interpreted as
height and no grid pooling/convolution is applied to latent slots. Experts
always execute on a native F/M/C grid. K can be non-square, including primes.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from .observed_consistency_coe import FreeScaleHistoryConsistencyCoE
from .four_direction_coe import FourDirectionCoE
from .id_priority_coe import IDPriorityCoE
from .temporal_spatial_coe import TemporalSpatialCoE
from ..utils.deterministic import pool
from ..performance import enabled as perf,diagnostics as collect_diagnostics


class CrossMap(nn.Module):
    """Single-head learned queries/keys with full-width values; no dropout/bias.

    Chunk only the independent (batch,time) axis, never the key sequence.
    Constant value fields are preserved at initialization. Zero values produce
    exactly zero messages throughout training, including after weight updates.
    """
    def __init__(self,dim,key_dim=16,chunk=64):
        super().__init__()
        self.query=nn.Linear(dim,key_dim,bias=False)
        self.key=nn.Linear(dim,key_dim,bias=False)
        self.value=nn.Linear(dim,dim,bias=False)
        self.output=nn.Linear(dim,dim,bias=False)
        nn.init.eye_(self.value.weight);nn.init.eye_(self.output.weight)
        self.scale=1/math.sqrt(key_dim);self.chunk=chunk
    def forward(self,queries,keys,values):
        if queries.ndim!=3 or keys.ndim!=3 or keys.shape!=values.shape:
            raise ValueError('CrossMap expects [B*T,sequence,D]')
        parts=[]
        for start in range(0,len(queries),self.chunk):
            end=start+self.chunk
            q=self.query(queries[start:end]);k=self.key(keys[start:end]);v=self.value(values[start:end])
            logits=torch.bmm(q,k.transpose(1,2))*self.scale
            weights=logits.float().softmax(-1).to(v.dtype)
            parts.append(self.output(torch.bmm(weights,v)))
        return torch.cat(parts,0)


def flatten_grid(x):
    b,d,t,h,w=x.shape
    return x.permute(0,2,3,4,1).reshape(b*t,h*w,d)


def flatten_slots(x):
    b,d,t,k,one=x.shape
    if one!=1:raise ValueError('Internal slot view must end in singleton dimension')
    return x.squeeze(-1).permute(0,2,3,1).reshape(b*t,k,d)


class LatentTokenCoE(FreeScaleHistoryConsistencyCoE):
    @classmethod
    def from_config(cls,cfg):
        spec=copy.deepcopy(cfg['model']['coe']['latent_tokens'])
        if set(spec)!={'count','mapping','return_mode','delta','history','path'}:
            raise ValueError('Explicit token configuration required')
        if type(spec['count']) is not int or not 1<=spec['count']<=256:
            raise ValueError('Token count must be in [1,256], independently of grid resolution')
        if spec['mapping'] not in ('shared','per_scale') or spec['return_mode'] not in ('replace','residual'):
            raise ValueError('Unknown token mapping/update')
        if spec['path'] not in ('free_last_f','CMFF') or any(type(spec[k]) is not bool for k in ('delta','history')):
            raise ValueError('Invalid token route/communication')
        m=super().from_config(cfg);m.token_spec=spec
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg['seed'])+114101)
            # Same 256-slot bank draw for every K; common prefix is identical.
            bank=torch.randn(256,m.dim)
            m.token_slots=nn.Parameter(bank[:spec['count']].clone())
            encode=CrossMap(m.dim);decode=CrossMap(m.dim)
            count=1 if spec['mapping']=='shared' else 3
            m.grid_to_tokens=nn.ModuleList([copy.deepcopy(encode) for _ in range(count)])
            m.tokens_to_grid=nn.ModuleList([copy.deepcopy(decode) for _ in range(count)])
        if not spec['delta']:
            m.communication='none';m.delta_steps=();m.history_gate.requires_grad_(False)
        if not spec['history']:m.x_history.requires_grad_(False)
        if spec['path']=='CMFF':
            m.scale_policy='fixed'
            for head in m.scale_heads:head.requires_grad_(False)
        return m

    def _slot_queries(self,bt):
        return self.token_slots.unsqueeze(0).expand(bt,-1,-1)

    def _map_index(self,sid):return 0 if self.token_spec['mapping']=='shared' else sid

    def encode_grid(self,x,pos,sid):
        b,d,t,_,_=x.shape
        values=flatten_grid(x)
        keys=values+flatten_grid(pos)
        slots=self.grid_to_tokens[self._map_index(sid)](self._slot_queries(b*t),keys,values)
        return slots.reshape(b,t,self.token_spec['count'],d).permute(0,3,1,2).unsqueeze(-1)

    def decode_grid(self,h,pos,sid):
        b,d,t,height,width=pos.shape
        values=flatten_slots(h)
        keys=values+self._slot_queries(b*t)
        grid=self.tokens_to_grid[self._map_index(sid)](flatten_grid(pos),keys,values)
        return grid.reshape(b,t,height,width,d).permute(0,4,1,2,3)

    def output_hidden(self,h):
        if getattr(self,'_decoded_state',None) is h:return self._decoded_features
        return self.decode_grid(h,self._token_positions[0],0)

    def decode_state(self,h):
        features=self.decode_grid(h,self._token_positions[0],0)
        self._decoded_state=h;self._decoded_features=features
        return self.decoder(features)

    def summary(self,h,mask):
        # No correspondence between slot index and a spatial location. Only the
        # visible fraction per time/variable may weight temporal summary stats.
        coverage=mask.mean((-2,-1),keepdim=True).expand(-1,-1,-1,h.shape[-2],1)
        return FourDirectionCoE.summary(self,h,coverage)

    def _router_features(self,hidden,values,support_summary,change,missing,pattern_summary=None):
        coverage=missing.mean((1,3,4),keepdim=True).expand(-1,1,-1,hidden.shape[-2],1)
        fraction=missing.mean((1,2,3,4)).unsqueeze(1)
        return torch.cat((hidden.mean((2,3,4)),self._missing_pool(hidden,coverage),
            values.mean((2,3,4)),self._missing_pool(values,missing),support_summary,
            change.mean((2,3,4)),self._missing_pool(change,missing),fraction,
            (fraction==0).to(hidden.dtype)),1)

    def _initialize(self,x,mask):
        c=FourDirectionCoE._initialize(self,x,mask)
        pyramid=[];encodings=[];positions=[]
        for sid,((values,native_mask),factor) in enumerate(zip(self._observations,(1,2,4))):
            pos=pool(c['pos'],factor) if factor>1 else c['pos']
            support=c['support'] if sid==0 and perf(self,'reuse_fine_support') else self._observation_support(native_mask)
            pyramid.append((values,native_mask,support,pos));positions.append(pos)
            encoded=c['h'] if sid==0 else self.encoder(torch.cat((values,native_mask,support,pos),1))*native_mask.mean(1,keepdim=True)
            encodings.append(self.encode_grid(encoded,pos,sid))
        self._token_positions=positions
        h=self.observation_fusion(torch.cat(encodings,1));pred=self.decode_state(h)
        unset=torch.full((len(x),),-1,dtype=torch.long,device=x.device)
        observed=torch.where(c['mask'].bool(),c['x'].float().square(),0.)
        scale=(observed.sum((2,3,4),keepdim=True)/c['mask'].sum((2,3,4),keepdim=True).clamp_min(1)).sqrt().clamp_min(1).detach()
        c.update(h=h,h0=h,prev=h,initial_prediction=pred,last_prediction=pred,
            completion=torch.where(c['mask'].bool(),c['x'],pred),observed_pyramid=pyramid,
            output_scale=unset,previous_output_scale=unset,x_pool=[],x_scale=scale)
        return c

    def _memory(self,c):
        enabled=c.get('exploration_step',-1)+1 in self.delta_steps
        corrected,route,gate=IDPriorityCoE._memory(self,c if enabled else dict(c,has_prev=False))
        c['w_stats']={k:c['h'].new_zeros(()) for k in
            ('feedback_abs','feedback_route_abs','visible_residual_abs','feedback_coverage')}
        stats={'pool_size':c['h'].new_tensor(len(c['x_pool'])),'pool_delta_abs':c['h'].new_zeros(())}
        if c['x_pool'] and self.token_spec['history']:
            # Existing history reader uses h only for per-channel global RMS;
            # its prediction/mask input and its message remain on the F grid.
            fine_delta,weights=self.x_history(c['h'],c['x_pool'],c['mask'],c['x_scale'])
            delta=self.encode_grid(fine_delta,c['pos'],0)*self.history_gain
            # Bound the token message without amplifying a zero/small message.
            hrms=c['h'].detach().float().square().mean((2,3,4),keepdim=True).sqrt()
            drms=delta.detach().float().square().mean((2,3,4),keepdim=True).sqrt()
            delta=delta*(.1*hrms/drms.clamp_min(1e-6)).clamp(max=1).to(delta.dtype)
            corrected=corrected+delta
            if collect_diagnostics(self):
                stats['pool_delta_abs']=delta.detach().float().abs().mean()
                for i in range(len(c['x_pool'])):stats[f'pool_weight_P{i+1}']=weights[:,i].detach().mean()
        c['x_stats']=stats
        return corrected,route,gate

    def _execute(self,c,corrected,weights,scale_weights,step):
        out=torch.zeros_like(c['h'])
        for sid,factor in enumerate((1,2,4)):
            ids=(scale_weights[:,sid].detach()!=0).nonzero().flatten()
            if not ids.numel():continue
            values,mask,support,pos=c['observed_pyramid'][sid];pos=pos.index_select(0,ids)
            current=corrected.index_select(0,ids);native=self.decode_grid(current,pos,sid)
            fallback=c['completion'].index_select(0,ids)
            if factor>1:fallback=pool(fallback,factor)
            nm=mask.index_select(0,ids)
            completion=torch.where(nm.bool(),values.index_select(0,ids),fallback)
            z=self.state_norm(self.state_projection(torch.cat((native,completion,nm,support.index_select(0,ids),pos),1)))
            update=TemporalSpatialCoE._dispatch_weighted(self,z,weights.index_select(0,ids),step)
            if self.token_spec['return_mode']=='residual':update=current+self.encode_grid(update-native,pos,sid)
            else:update=self.encode_grid(update,pos,sid)
            update=update*scale_weights[ids,sid,None,None,None,None].to(update.dtype)
            out=out.index_add(0,ids,update.to(out.dtype))
        c['w_stats']['adapter_abs']=out.new_zeros(())
        return out,None,[]

    def _round(self,c,step,*args,**kwargs):
        shape=(self.token_spec['count'],1)
        if tuple(c['h'].shape[-2:])!=shape:raise RuntimeError('Round input must be independent slots')
        new,row=super()._round(c,step,*args,**kwargs)
        if tuple(new['h'].shape[-2:])!=shape:raise RuntimeError('Round output must be independent slots')
        if not self.token_spec['history']:new['x_pool']=[]
        if collect_diagnostics(self):
            with torch.no_grad():
                slots=new['h'].detach().float().squeeze(-1)
                rms=slots.square().mean().sqrt()
                spread=(slots-slots.mean(-1,keepdim=True)).square().mean().sqrt()
                unit=slots/slots.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
                pair=(unit.sum(-1).square().sum(1)-unit.square().sum((1,3)))/(max(self.token_spec['count']*(self.token_spec['count']-1),1))
                self._token_stats.append({'rms':rms,'slot_spread':spread,'spread_rms_ratio':spread/rms.clamp_min(1e-8),'mean_slot_cosine':pair.mean()})
        return new,row

    def forward(self,*args,**kwargs):
        if hasattr(self,'_token_positions'):raise RuntimeError('Overlapping token forward')
        self._token_stats=[]
        try:
            out=super().forward(*args,**kwargs)
            out['coe']['diagnostics']['latent_token_count']=out['x_hat_main'].new_tensor(float(self.token_spec['count']))
            for step,stats in enumerate(self._token_stats,1):
                for key,value in stats.items():out['coe']['diagnostics'][f'token_step{step}_{key}']=value
            return out
        finally:
            for attr in ('_token_positions','_decoded_state','_decoded_features','_token_stats'):
                if hasattr(self,attr):delattr(self,attr)
