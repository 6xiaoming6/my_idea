"""I42-I53 orchestration, isolated from the frozen I37 round implementation.

Inherited constructors and output wrappers reuse the original experts, decoder,
history scorer and final projection. The new round below owns its data flow.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .observed_consistency_coe import FreeScaleHistoryConsistencyCoE
from .scale_delta_exploration_coe import ScaleDeltaExplorationCoE
from .history_plane_interaction_coe import zero_last
from .observation_regions import ObservationRegions
from .temporal_spatial_coe import SUPPORT_FEATURE_NAMES
from ..utils.deterministic import pool, resize


DEFAULTS = dict(refresh_completion=False, expert_router_input='legacy',
    round_input='absolute', routing_evidence=False, delta='legacy',
    history='legacy', pair='parallel', state_update='replace', fixed_path=None)


class ObservationIterationCoE(FreeScaleHistoryConsistencyCoE):
    @staticmethod
    def rms_match(message,h):
        ratio=(h.detach().float().square().mean((2,3,4),keepdim=True).sqrt()
               /message.detach().float().square().mean((2,3,4),keepdim=True).sqrt().clamp_min(1e-6))
        # Ownership updates can be exactly identity. A valid zero message must
        # not become 0 * inf when the RMS ratio is cast to FP16 (>65504).
        if message.dtype==torch.float16 and bool((ratio>torch.finfo(torch.float16).max).any()):
            return (message.float()*ratio).to(message.dtype)
        return message*ratio.to(message.dtype)

    @classmethod
    def from_config(cls, cfg):
        requested=copy.deepcopy(cfg['model']['coe']['observation_iteration'])
        if not requested.pop('enabled',False) or set(requested)-set(DEFAULTS):
            raise ValueError('Invalid observation_iteration options')
        spec={**DEFAULTS,**requested}
        for key,allowed in dict(expert_router_input=('legacy','native'), round_input=('absolute','remaining_residual'),
                delta=('legacy','physical_split'),history=('legacy','orthogonal_innovation'),
                pair=('parallel','residual_relay'),state_update=('replace','ownership_correction')).items():
            if spec[key] not in allowed: raise ValueError(key)
        if spec['fixed_path'] not in (None,'MMFF'): raise ValueError('Only MMFF control registered')
        m=super().from_config(cfg);m.iteration=spec
        if spec['fixed_path']:
            m.scale_policy='fixed';m.spec['fixed_path']=[1,1,0,0];m.scale_heads.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg['seed'])+420001)
            if spec['routing_evidence']:
                m.x_observation_scale=zero_last(nn.Sequential(nn.Linear(12,16),nn.GELU(),nn.Linear(16,3)))
                m.x_observation_expert=zero_last(nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,8)))
            if spec['delta']=='physical_split':
                m.x_observation_delta=zero_last(nn.Sequential(nn.Conv3d(2*m.c_in,32,1),nn.GELU(),nn.Conv3d(32,m.dim,1)))
                m.history_gate.requires_grad_(False)
        return m

    def _initialize(self,x,mask):
        c=super()._initialize(x,mask)
        c['regions']=ObservationRegions(self._observations)
        c['previous_prediction']=c['initial_prediction']
        return c

    def _scales(self,c,features,step,force=None,probe=False):
        if not self.iteration['routing_evidence']:
            return super()._scales(c,features,step,force,probe)
        bias=self.x_observation_scale(c['regions'].statistics(c['last_prediction']).flatten(1))
        handle=self.scale_heads[step].register_forward_hook(lambda module,args,out:out+bias)
        try:return super()._scales(c,features,step,force,probe)
        finally:handle.remove()

    def _communication(self,c):
        h=c['h'];ledger=c['regions'];stats={};rms=h.detach().float().square().mean((2,3,4),keepdim=True).sqrt()
        if self.iteration['delta']=='legacy':
            corrected,route,gate=ScaleDeltaExplorationCoE._memory(self,c)
        else:
            corrected=h;route=h.new_zeros((len(h),self.num_experts));gate=h.new_zeros((len(h),self.dim))
            c['w_stats']={k:h.new_zeros(()) for k in ('feedback_abs','feedback_route_abs','visible_residual_abs','feedback_coverage')}
            if c['has_prev']:
                total,detail=ledger.split_delta(c['previous_prediction'],c['last_prediction'])
                message=torch.cat((total/ledger.sigma,detail/ledger.sigma),1)
                delta=.1*rms*self.x_observation_delta(message).float().tanh()
                corrected=h+delta.to(h.dtype);gate=(delta/(rms+1e-12)).mean((2,3,4))
                stats.update(delta_total_abs=total.detach().abs().mean(),delta_detail_abs=detail.detach().abs().mean())
        stats['delta_injection_abs']=(corrected-h).detach().float().abs().mean()
        xstats={'pool_size':h.new_tensor(len(c['x_pool'])),'pool_delta_abs':h.new_zeros(())}
        if c['x_pool']:
            if self.iteration['history']=='legacy':
                delta,weights=self.x_history(h,c['x_pool'],c['mask'],c['x_scale'])
            else:
                read,weights=self.x_history.read(c['x_pool'],c['mask'],c['x_scale'])
                current=torch.where(c['mask'].bool(),c['x'],c['last_prediction']).detach()
                previous=torch.where(c['mask'].bool(),c['x'],c['previous_prediction']).detach()
                innovation=ledger.history_innovation(read,current,previous)
                baseline=self.x_history.inject(torch.cat((torch.zeros_like(innovation),c['mask']),1))
                encoded=self.x_history.inject(torch.cat((innovation/ledger.sigma,c['mask']),1))-baseline
                delta=(.1*rms*encoded.float().tanh()*(1-c['mask']).mean(1,keepdim=True)).to(h.dtype)
                stats.update(history_before_abs=(read-current).detach().abs().mean(),
                             history_innovation_abs=innovation.detach().abs().mean())
            delta=delta*self.history_gain;corrected=corrected+delta
            xstats['pool_delta_abs']=delta.detach().float().abs().mean()
            for i in range(len(c['x_pool'])):xstats[f'pool_weight_P{i+1}']=weights[:,i].detach().mean()
        c['x_stats']=xstats;c['iteration_stats']=stats
        return corrected,route,gate

    def _native(self,c,hidden,pred,sid,ids):
        factor=(1,2,4)[sid]
        values,mask,support,pos=[v.index_select(0,ids) for v in c['observed_pyramid'][sid]]
        h=hidden.index_select(0,ids)
        if factor>1:h=pool(h,factor)
        if self.iteration['round_input']=='remaining_residual':
            completion=c['regions'].residuals(pred)[sid].index_select(0,ids)
        else:
            filled=torch.where(c['mask'].bool(),c['x'],pred)
            fallback=filled.index_select(0,ids)
            if factor>1:fallback=pool(fallback,factor)
            completion=torch.where(mask.bool(),values,fallback)
        z=self.state_norm(self.state_projection(torch.cat((h,completion,mask,support,pos),1)))
        sm=(1-mask).repeat(1,len(SUPPORT_FEATURE_NAMES),1,1,1)
        ss=torch.cat((support.mean((2,3,4)),support.std((2,3,4),unbiased=False),support.amax((2,3,4)),self._missing_pool(support,sm)),1)
        features=self._router_features(z,completion,ss,torch.zeros_like(completion),1-mask,None)
        return z,features

    def _restore(self,c,old,native,sid,ids):
        if self.iteration['state_update']=='ownership_correction':
            # Ledger slicing is read-only and has no random operations.
            ledger=copy.copy(c['regions']);ledger.owners=[v.index_select(0,ids) for v in ledger.owners]
            return ledger.update(old,native,sid)
        return resize(native,old.shape[-2:]) if sid else native

    def _round(self,c,step,force_scale=None,force_pair=None,probe=False):
        if not self.iteration['refresh_completion']:
            return super()._round(c,step,force_scale,force_pair,probe)
        c['exploration_step']=step;h=c['h'];zero=h.new_zeros(())
        features=self._router_features(h,c['completion'],c['ss'],torch.zeros_like(c['initial_prediction']),1-c['mask'],None)
        sw,sp,active,scale_record=self._scales(c,features,step,force_scale,probe)
        corrected,route,gate=self._communication(c)
        native={};expert_features=torch.zeros_like(features)
        for sid in range(3):
            indices=active[:,sid].nonzero().flatten()
            if not len(indices):continue
            z,f=self._native(c,corrected,c['last_prediction'],sid,indices)
            native[sid]=(indices,z)
            expert_features=expert_features.index_copy(0,indices,f.to(expert_features.dtype))
        if self.iteration['expert_router_input']=='legacy':expert_features=features
        logits=self.routers[step](expert_features)+route
        if self.iteration['routing_evidence']:
            evidence=c['regions'].statistics(c['last_prediction'])
            logits=logits+self.x_observation_expert(evidence[torch.arange(len(h),device=h.device),sw.detach().argmax(1)])
        ew,ids,importance,coefficient,pair_record=self._pairs(c,expert_features,logits,step,force_pair,probe)
        next_h=torch.zeros_like(h)
        for sid,(indices,z) in native.items():
            w=ew.index_select(0,indices);old=h.index_select(0,indices)
            if self.iteration['pair']=='parallel':
                candidate=self._restore(c,old,self._dispatch_weighted(z,w,step),sid,indices)
            else:
                # Stable score order, ties by expert index; choose only once.
                selected=logits.index_select(0,indices).masked_fill(w<=0,-torch.inf).argsort(dim=1,descending=True,stable=True)[:,:2]
                first=self._restore(c,old,self._dispatch(z,selected[:,0],step),sid,indices)
                temporary=h.index_copy(0,indices,first.to(h.dtype))
                interim_prediction=c['last_prediction'].index_copy(0,indices,self.decoder(first).to(c['last_prediction'].dtype))
                second_input,_=self._native(c,temporary,interim_prediction,sid,indices)
                second=self._restore(c,first,self._dispatch(second_input,selected[:,1],step),sid,indices)
                within=w.gather(1,selected)
                candidate=first*within[:,0,None,None,None,None]+second*within[:,1,None,None,None,None]
                before=c['regions'].statistics(c['last_prediction'])[indices,sid,0].mean()
                after=c['regions'].statistics(interim_prediction)[indices,sid,0].mean()
                c['iteration_stats'][f'expert1_s{sid}_residual_before']=before.detach()
                c['iteration_stats'][f'expert1_s{sid}_residual_after']=after.detach()
            candidate=candidate*sw[indices,sid,None,None,None,None].to(candidate.dtype)
            next_h=next_h.index_add(0,indices,candidate.to(next_h.dtype))
        next_h=next_h*coefficient[:,None,None,None,None].to(next_h.dtype)
        pred=self.decoder(next_h);change=torch.where(c['mask'].bool(),0.,(pred-c['last_prediction']).abs())
        new=dict(c);new.update(h=next_h,prev=h,has_prev=True,last_prediction=pred,
            previous_prediction=c['last_prediction'],completion=torch.where(c['mask'].bool(),c['x'],pred),
            counts=c['counts']+active.to(c['counts'].dtype),last_scale=sw.detach(),
            last_ids=F.one_hot(ids,self.num_experts).flatten(1).to(h.dtype),last_within=ew.gather(1,ids),
            ages=torch.where(active,torch.zeros_like(c['ages']),c['ages']+1),
            x_pool=c['x_pool']+[torch.where(c['mask'].bool(),c['x'],pred).detach()],
            previous_output_scale=c['output_scale'],output_scale=sw.detach().argmax(1))
        row=dict(prediction=pred,change=change,logits=logits,probs=logits.float().softmax(1),weights=ew,
            ids=ids,importance=importance,scale_weights=sw,scale_probs=sp,active=active,gate=gate,
            input_delta=(corrected-h).detach().float().abs().mean(),route_delta=route.detach().float().abs().mean(),
            memory_norm=zero,cache_age=c['ages'].detach().float().mean(),delta=(next_h-h).detach().float().abs().mean(),
            scale_record=scale_record,pair_record=pair_record)
        for key in ('message_rms_mean','rms_ratio_mean','rms_ratio_max','rms_ratio_p95'):row[key]=zero
        stats=c['regions'].statistics(pred).detach().mean(0)
        for sid,label in enumerate(('F','M','C')):
            for j,key in enumerate(('residual_abs','residual_rms','coverage','dof')):
                c['iteration_stats'][label+'_'+key]=stats[sid,j]
        c['iteration_stats']['unconstrained_fraction']=c['regions'].unconstrained.float().mean()
        self._iteration_rows.append(c['iteration_stats']);self._x_stats.append(c['x_stats'])
        c['w_stats']['adapter_abs']=zero;self._w_stats.append(c['w_stats'])
        self._exploration_rows.append((row,ids,c['output_scale'],c['previous_output_scale']))
        return new,row

    def forward(self,*args,**kwargs):
        self._iteration_rows=[]
        try:
            out=super().forward(*args,**kwargs)
            if self.iteration['refresh_completion']:
                out['coe']['initial_completion']=torch.where(out['coe']['observation_mask'].bool(),
                    args[0] if args else kwargs['x_f'],out['coe']['initial_prediction'])
            for i,stats in enumerate(self._iteration_rows,1):
                for key,value in stats.items():out['coe']['diagnostics'][f'observation_step{i}_{key}']=value.detach()
            return out
        finally:del self._iteration_rows
