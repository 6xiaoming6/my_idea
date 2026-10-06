"""R10: fixed direction-complementarity constraint, native sparse fusion/loss."""
from __future__ import annotations
import torch
from .temporal_spatial_coe import TemporalSpatialCoE


class DirectionPairCoE(TemporalSpatialCoE):
    @classmethod
    def from_config(cls,cfg):
        model=super().from_config(cfg)
        if (model.top_k!=2 or model.routing_mode!='hard' or model.pair_mode!='native'
                or model.fusion_mode!='original' or model.routing_warmup_epochs
                or model.routing_transition_epochs):
            raise ValueError('Direction constraint requires native hard Top-2 without warmup')
        temporal={'T','TD','TA','TL'};spatial={'S','SD','SL'}
        if set(model.expert_names)!=temporal|spatial|{'ST'}:
            raise ValueError('Direction constraint requires the eight standard experts')
        allowed=[]
        for i,j in model.pair_indices.tolist():
            a,b=model.expert_names[i],model.expert_names[j]
            allowed.append('ST' in (a,b) or (a in temporal and b in spatial) or (b in temporal and a in spatial))
        model.register_buffer('direction_allowed_pairs',torch.tensor(allowed,dtype=torch.bool),persistent=False)
        return model

    def _select_native_topk(self,logits):
        native=super()._select_native_topk(logits)
        native_sorted=native.sort(-1).values
        pairs=self.pair_indices
        scores=(logits.float()[:,pairs[:,0]]+logits.float()[:,pairs[:,1]])
        scores=scores.masked_fill(~self.direction_allowed_pairs,-torch.inf)
        best=scores.argmax(-1)
        native_ids=(pairs[None]==native_sorted[:,None]).all(-1).long().argmax(-1)
        preserve=self.direction_allowed_pairs[native_ids] & (scores.gather(1,native_ids[:,None]).squeeze(1)==scores.max(-1).values)
        return torch.where(preserve[:,None],native,pairs[best])
