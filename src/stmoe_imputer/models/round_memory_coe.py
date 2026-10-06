"""N7: B3 plus zero-initialized, content-addressed cross-round hidden memory.

The normal Router still reads the latest hidden state. Memory retrieval only
changes the expert input. No extra experts, candidate execution, spatial scales,
or decoded-completion feedback are introduced. Historical tensors are retained
inside one forward only and keep gradients to their generating computation.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .temporal_spatial_coe import TemporalSpatialCoE


class RoundMemoryCoE(TemporalSpatialCoE):
    @classmethod
    def from_config(cls, cfg):
        model=super().from_config(cfg)
        spec=cfg['model']['coe']['round_memory']
        key_dim=spec.get('key_dim',16)
        if type(key_dim) is not int or key_dim<1:
            raise ValueError('round_memory.key_dim must be a positive integer')
        if (model.num_steps!=4 or model.top_k!=2 or model.expert_sharing!='shared'
                or model.routing_mode!='hard' or model.pair_mode!='native'
                or model.state_update_mode!='direct' or model.completion_feedback
                or model.use_shared or not model.use_routed
                or model.router_state!='dynamic' or model.expert_state!='dynamic'
                or model.fusion_mode!='original' or model.acceptance!='none'
                or model.routing_warmup_epochs or model.routing_transition_epochs):
            raise ValueError('N7 requires the four-round B3 shared native Top-2 protocol')
        model.memory_mode=spec.get('mode','content')
        if model.memory_mode not in ('content','uniform','recent'):
            raise ValueError('Unknown round-memory mode')
        model.memory_key_dim=key_dim
        summary_dim=2*model.dim
        # Added after constructing the backbone: all existing B3 parameters keep
        # exactly their original initialization under the same seed.
        model.memory_query=nn.Sequential(nn.LayerNorm(summary_dim),nn.Linear(summary_dim,key_dim,bias=False))
        model.memory_key=nn.Sequential(nn.LayerNorm(summary_dim),nn.Linear(summary_dim,key_dim,bias=False))
        model.memory_gate=nn.Sequential(nn.LayerNorm(2*summary_dim),nn.Linear(2*summary_dim,model.dim))
        nn.init.zeros_(model.memory_gate[-1].weight)
        nn.init.zeros_(model.memory_gate[-1].bias)
        if model.memory_mode != 'content':
            model.memory_query.requires_grad_(False)
            model.memory_key.requires_grad_(False)
        return model

    def _memory_summary(self, hidden, mask):
        missing=(1-mask).mean(dim=1,keepdim=True)
        return torch.cat((hidden.mean((2,3,4)),self._missing_pool(hidden,missing)),dim=1)

    def _expert_input(self, hidden, completion, mask, support, position, step):
        current=self._memory_summary(hidden,mask)
        if self._memory_states:
            history=torch.stack(self._memory_summaries,dim=1)
            if self.memory_mode == 'content':
                query=self.memory_query(current)
                keys=self.memory_key(history)
                logits=(keys.float()*query.float().unsqueeze(1)).sum(-1)/math.sqrt(self.memory_key_dim)
                weights=logits.softmax(-1)
            elif self.memory_mode == 'uniform':
                weights=history.new_full(history.shape[:2],1/history.shape[1],dtype=torch.float32)
            else:
                weights=history.new_zeros(history.shape[:2],dtype=torch.float32)
                weights[:,-1]=1.
            # Stream the full-grid values instead of allocating a [B,R,C,T,H,W]
            # stack. No value or key is detached: task gradients can reach history.
            memory=sum(state*weights[:,i,None,None,None,None].to(state.dtype)
                       for i,state in enumerate(self._memory_states))
            memory_summary=(history*weights.to(history.dtype).unsqueeze(-1)).sum(1)
            gate=self.memory_gate(torch.cat((current,memory_summary),dim=1)).tanh()
            corrected=hidden+gate[:,:,None,None,None]*(memory-hidden)
            padded=weights.new_zeros((hidden.shape[0],self.num_steps))
            padded[:,:step]=weights
        else:
            corrected=hidden
            gate=hidden.new_zeros((hidden.shape[0],self.dim))
            padded=hidden.new_zeros((hidden.shape[0],self.num_steps))
        self._memory_states.append(hidden)
        self._memory_summaries.append(current)
        self._memory_weight_history.append(padded)
        self._memory_gate_history.append(gate)
        self._memory_delta_history.append((corrected-hidden).detach().float().abs().mean())
        return super()._expert_input(corrected,completion,mask,support,position,step)

    def forward(self,x_f,m_f,**kwargs):
        self._memory_states=[]
        self._memory_summaries=[]
        self._memory_weight_history=[]
        self._memory_gate_history=[]
        self._memory_delta_history=[]
        try:
            output=super().forward(x_f,m_f,**kwargs)
            weights=torch.stack(self._memory_weight_history,dim=1)
            gates=torch.stack(self._memory_gate_history,dim=1)
            output['coe']['round_memory_weights']=weights
            output['coe']['round_memory_gates']=gates
            diagnostics=output['diagnostics']['coe']
            for step in range(self.num_steps):
                diagnostics[f'step{step+1}_memory_gate_abs_mean']=gates[:,step].detach().float().abs().mean()
                diagnostics[f'step{step+1}_memory_gate_signed_mean']=gates[:,step].detach().float().mean()
                diagnostics[f'step{step+1}_memory_input_delta']=self._memory_delta_history[step]
                for source in range(step):
                    diagnostics[f'step{step+1}_memory_from_h{source}_weight']=weights[:,step,source].detach().float().mean()
            return output
        finally:
            # Never carry state between samples, batches, train/eval or OOD sets.
            self._memory_states=[]
            self._memory_summaries=[]
            self._memory_weight_history=[]
            self._memory_gate_history=[]
            self._memory_delta_history=[]
