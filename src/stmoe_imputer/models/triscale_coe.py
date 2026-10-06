"""Three-resolution, genuinely sparse CoE; legacy two-scale models are unchanged."""
from __future__ import annotations
from itertools import permutations
import torch
from torch import nn
from torch.nn import functional as F
from .temporal_spatial_coe import TemporalSpatialCoE
from .spatial_scale_coe import observed_pool, spatial_pool, spatial_resize


class TriScaleCoE(TemporalSpatialCoE):
    factors = (1, 2, 4)  # F, M, C; time resolution is unchanged.
    scale_names = ('fine', 'mid', 'coarse')

    @classmethod
    def from_config(cls, cfg):
        model = super().from_config(cfg)
        spec = cfg['model']['coe']['triscale']
        coe = cfg['model']['coe']
        if (model.num_steps != 4 or model.top_k != 2 or model.routing_mode != 'hard'
                or model.pair_mode != 'native' or model.state_update_mode != 'direct'
                or model.completion_feedback or model.use_shared or not model.use_routed
                or model.router_state != 'dynamic' or model.expert_state != 'dynamic'
                or model.fusion_mode != 'original' or model.acceptance != 'none'
                or model.routing_warmup_epochs or model.routing_transition_epochs
                or model.expert_sharing not in ('shared', 'per_step')
                or any(coe.get(k, {}).get('enabled', False) for k in ('spatial_scale','local_routing','round_memory'))
                or coe.get('expert_pair_constraint', 'none') != 'none'):
            raise ValueError('TriScale requires native four-round direct Top-2 without other routing modules')
        model.triscale_mode = spec.get('mode', 'free')
        model.triscale_update = spec.get('update', 'direct')
        model.fixed_scales = tuple(spec.get('fixed_scales', (2,1,0,0)))
        model.explore_epochs = spec.get('explore_epochs', 8)
        model.recent_memory = spec.get('recent_memory', False)
        if model.triscale_mode not in ('fixed','free','explore_free') or model.triscale_update not in ('direct','detail_preserving'):
            raise ValueError('Invalid triscale mode/update')
        if len(model.fixed_scales) != 4 or any(type(v) is not int or v not in (0,1,2) for v in model.fixed_scales):
            raise ValueError('fixed_scales must contain four F/M/C indices')
        if type(model.explore_epochs) is not int or model.explore_epochs < 1:
            raise ValueError('explore_epochs must be positive')
        features = model.routers[0][0].normalized_shape[0]
        hidden = coe.get('router_hidden_dim', model.dim)
        model.scale_routers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(features+4),nn.Linear(features+4,hidden),nn.GELU(),nn.Linear(hidden,3))
            for _ in range(4)])
        for head in model.scale_routers:
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        if model.triscale_mode == 'fixed':model.scale_routers.requires_grad_(False)
        if model.triscale_mode == 'explore_free':
            generator = torch.Generator(device='cpu').manual_seed(spec.get('explore_seed',20261001))
            model.register_buffer('_explore_rng_state',generator.get_state())
            model.register_buffer('_explore_paths',torch.tensor(sorted(set(permutations((2,1,0,0)))),dtype=torch.long))
        if model.recent_memory:
            # Same signed channel gate as R9; unused query/key modules are omitted.
            model.memory_gate = nn.Sequential(nn.LayerNorm(4*model.dim),nn.Linear(4*model.dim,model.dim))
            nn.init.zeros_(model.memory_gate[-1].weight);nn.init.zeros_(model.memory_gate[-1].bias)
        return model

    def _router_features(self,*args,**kwargs):
        result = super()._router_features(*args,**kwargs)
        self._scale_features = result
        return result

    def _memory_summary(self,hidden,mask):
        return torch.cat((hidden.mean((2,3,4)),self._missing_pool(hidden,(1-mask).mean(1,keepdim=True))),1)

    def _expert_input(self,hidden,completion,mask,support,position,step):
        self._original_hidden = hidden
        corrected = hidden
        if self.recent_memory:
            gate = hidden.new_zeros((hidden.shape[0],self.dim))
            if self._previous_hidden is not None:
                current = self._memory_summary(hidden,mask)
                previous = self._memory_summary(self._previous_hidden,mask)
                gate = self.memory_gate(torch.cat((current,previous),1)).tanh()
                corrected = hidden + gate[:,:,None,None,None]*(self._previous_hidden-hidden)
            self._previous_hidden = hidden  # Retain gradient, within this forward only.
            self._memory_gates.append(gate)
        self._scale_inputs = (corrected,completion,mask,support,position)
        return super()._expert_input(corrected,completion,mask,support,position,step)

    def _choose_scale(self,step,reference):
        b = reference.shape[0]
        if self.triscale_mode == 'fixed':
            choice = torch.full((b,),self.fixed_scales[step],device=reference.device,dtype=torch.long)
            probs = F.one_hot(choice,3).float();weights=probs
        else:
            remaining = reference.new_full((b,1),(self.num_steps-step)/self.num_steps)
            features = torch.cat((self._scale_features,(self._scale_counts/self.num_steps).to(reference.dtype),remaining),1)
            logits = self.scale_routers[step](features).float()
            prior = logits.new_zeros(3);prior[self.fixed_scales[step]]=1e-3
            probs = (logits+prior).softmax(-1)
            if self.training and self.triscale_mode == 'explore_free' and self.routing_epoch <= self.explore_epochs:
                if step == 0:
                    generator=torch.Generator(device='cpu');generator.set_state(self._explore_rng_state.cpu())
                    indices=torch.randint(len(self._explore_paths),(b,),generator=generator).to(reference.device)
                    self._exploration_path=self._explore_paths.index_select(0,indices)
                    self._explore_rng_state.copy_(generator.get_state().to(self._explore_rng_state.device))
                choice=self._exploration_path[:,step]
            else:choice=probs.argmax(-1)
            hard=F.one_hot(choice,3).to(probs.dtype)
            weights=hard+(probs-probs.detach()) if self.training else hard
        self._scale_counts += F.one_hot(choice.detach(),3).to(self._scale_counts.dtype)
        self._choices.append(choice.detach());self._probabilities.append(probs.detach())
        return weights

    def _restore_resolution(self,update,original,factor):
        if factor == 1:return update  # Exact direct behavior at F, without roundoff cancellation.
        update=spatial_resize(update,original.shape[-2:])
        if self.triscale_update == 'detail_preserving':
            detail=original-spatial_resize(spatial_pool(original,factor),original.shape[-2:])
            update=detail+update
        return update

    def _dispatch_weighted(self,unified,weights,step=0):
        scale_weights=self._choose_scale(step,unified)
        result=torch.zeros_like(unified)
        for sid,factor in enumerate(self.factors):
            selected=torch.nonzero(scale_weights[:,sid].detach()!=0,as_tuple=False).flatten()
            if not selected.numel():continue
            original=self._original_hidden.index_select(0,selected)
            if factor==1:inputs=unified.index_select(0,selected)
            else:
                hidden,completion,mask,support,position=[x.index_select(0,selected) for x in self._scale_inputs]
                values,coverage=observed_pool(completion,mask,factor)
                values=torch.where(coverage>0,values,spatial_pool(completion,factor))
                inputs=self.state_norm(self.state_projection(torch.cat((spatial_pool(hidden,factor),values,coverage,
                    spatial_pool(support,factor),spatial_pool(position,factor)),1)))
            update=super()._dispatch_weighted(inputs,weights.index_select(0,selected),step)
            update=self._restore_resolution(update,original,factor)
            coefficient=scale_weights.index_select(0,selected)[:,sid,None,None,None,None]
            result=result.index_add(0,selected,(update*coefficient.to(update.dtype)).to(result.dtype))
        return result

    def forward(self,x_f,m_f,**kwargs):
        if any(size%4 for size in x_f.shape[-2:]):raise ValueError('TriScale spatial dimensions must be divisible by four')
        self._scale_counts=x_f.new_zeros((x_f.shape[0],3))
        self._choices=[];self._probabilities=[];self._memory_gates=[]
        self._previous_hidden=None;self._exploration_path=None
        try:
            output=super().forward(x_f,m_f,**kwargs)
            output['coe']['triscale_choices']=torch.stack(self._choices,1)
            output['coe']['triscale_probabilities']=torch.stack(self._probabilities,1)
            diagnostics=output['diagnostics']['coe']
            diagnostics['scale_exploration_fraction']=x_f.new_tensor(float(self.training and self.triscale_mode=='explore_free' and self.routing_epoch<=self.explore_epochs))
            if self.recent_memory:
                gates=torch.stack(self._memory_gates,1);output['coe']['round_memory_gates']=gates
                history=gates.new_zeros((x_f.shape[0],4,4))
                for step in range(1,4):history[:,step,step-1]=1
                output['coe']['round_memory_weights']=history
                for step in range(4):
                    diagnostics[f'step{step+1}_memory_gate_abs_mean']=gates[:,step].detach().float().abs().mean()
                    diagnostics[f'step{step+1}_memory_gate_signed_mean']=gates[:,step].detach().float().mean()
            return output
        finally:
            self._previous_hidden=None;self._original_hidden=None;self._scale_inputs=None
            self._scale_features=None;self._exploration_path=None;self._memory_gates=[]
