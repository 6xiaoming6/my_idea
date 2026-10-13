"""Opt-in v25 experiments; preserve the frozen observed-input backbone."""
from __future__ import annotations

import copy
import torch
from torch.nn import functional as F

from .observed_multiscale_coe import ObservedMultiScaleCoE
from .temporal_spatial_coe import TemporalSpatialCoE
from ..utils.deterministic import pool, resize


class ScaleDeltaExplorationCoE(ObservedMultiScaleCoE):
    @classmethod
    def from_config(cls, cfg):
        spec = copy.deepcopy(cfg['model']['coe']['scale_delta_exploration'])
        if not spec.get('enabled'):
            raise ValueError('Exploration must be explicitly enabled')
        path = spec['path']
        ks = tuple(spec['top_k_per_step'])
        steps = tuple(spec['delta_steps'])
        if path not in ('CMFF', 'CFFF', 'free') or len(ks) != 4 or any(k not in (1,2,4,8) for k in ks):
            raise ValueError('Invalid path / per-round Top-k')
        if len(set(steps)) != len(steps) or any(s not in (2,3,4) for s in steps):
            raise ValueError('Delta steps are one-based expert INPUT rounds 2/3/4')
        if spec.get('same_scale_only', False) and (path != 'CFFF' or steps != (4,)):
            raise ValueError('Same-scale control must be CFFF, fourth round only')
        # All parameters and RNG draws remain exactly those of original B1/B2.
        ctor = copy.deepcopy(cfg)
        ctor['model']['coe']['top_k'] = 2
        ctor['model']['coe']['core_validation'].update(path='CMFF', communication='conditional' if steps else 'none')
        ctor['model']['coe']['id_priority'].update(scale_policy='fixed', fixed_epochs=0,
            scale_soft_warmup_epochs=0, scale_soft_transition_epochs=0)
        m = super().from_config(ctor)
        m.exploration = spec
        m.step_top_k = ks
        m.delta_steps = steps
        m.spec['fixed_path'] = [2,1,0,0] if path != 'CFFF' else [2,0,0,0]
        if path == 'free':
            m.scale_policy = 'st'
            for h in m.scale_heads:
                h.requires_grad_(True)
        return m

    def _initialize(self, x, mask):
        c = super()._initialize(x, mask)
        initial = torch.full((len(x),), -1, dtype=torch.long, device=x.device)
        c.update(output_scale=initial, previous_output_scale=initial)
        return c

    def _memory(self, c):
        enabled = c.get('exploration_step', -1) + 1 in self.delta_steps
        # Calling the inherited disabled path also initializes its W-stat contract.
        corrected, route, gate = super()._memory(c if enabled else dict(c, has_prev=False))
        if not enabled:
            c['w_stats'] = {k: c['h'].new_zeros(()) for k in
                ('feedback_abs','feedback_route_abs','visible_residual_abs','feedback_coverage')}
        if enabled and self.exploration.get('same_scale_only'):
            if not bool(((c['output_scale'] == 0) & (c['previous_output_scale'] == 0)).all()):
                raise RuntimeError('Same-scale delta read a non-F output')
        return corrected, route, gate

    def _pairs(self, c, features, logits, step, force=None, probe=False):
        k = self.step_top_k[step]
        if k == 2:
            return super()._pairs(c, features, logits, step, force, probe)
        if force is not None:
            raise ValueError('Forced pairs are not defined for variable Top-k')
        p = logits.float().softmax(1)
        chosen = logits.topk(k, 1).indices
        selected = F.one_hot(chosen, self.num_experts).any(1)
        weights = p * selected
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        coefficient = logits.new_ones((len(logits),), dtype=torch.float32)
        if k == 1 and self.training and not probe:
            probability = p.gather(1, chosen).squeeze(1)
            coefficient = 1 + (probability - probability.detach())
        left,right = self.pair_indices.unbind(1)
        pair_scores = logits.float()[:,left] + logits.float()[:,right]
        importance = self._pair_importance(logits, pair_scores.softmax(1), 1.)
        # The inherited forward temporarily requires two ids for its pair fields.
        # Replace those fields with the ACTUAL Top-k set before returning outputs.
        compat_ids = chosen[:, :2].sort(1).values if k > 1 else chosen.expand(-1,2)
        return weights, compat_ids, importance, coefficient, {
            'features': features, 'scores': None, 'base_logits': logits,
            'native': logits.topk(2,1).indices, 'chosen': chosen}

    def _execute(self, c, corrected, weights, scale_weights, step):
        if self.exploration['path'] == 'CMFF':
            return super()._execute(c, corrected, weights, scale_weights, step)
        out = torch.zeros_like(c['h'])
        for sid, factor in enumerate((1,2,4)):
            ids = (scale_weights[:,sid].detach() != 0).nonzero().flatten()
            if not ids.numel():
                continue
            values, mask, support, pos = c['observed_pyramid'][sid]
            h = corrected.index_select(0,ids)
            fallback = c['completion'].index_select(0,ids)
            if factor > 1:
                h, fallback = pool(h,factor), pool(fallback,factor)
            native_mask = mask.index_select(0,ids)
            completion = torch.where(native_mask.bool(), values.index_select(0,ids), fallback)
            z = self.state_norm(self.state_projection(torch.cat(
                (h,completion,native_mask,support.index_select(0,ids),pos.index_select(0,ids)),1)))
            update = TemporalSpatialCoE._dispatch_weighted(self,z,weights.index_select(0,ids),step)
            if factor > 1:
                update = resize(update,c['h'].shape[-2:])
            update = update * scale_weights[ids,sid,None,None,None,None].to(update.dtype)
            out = out.index_add(0,ids,update.to(out.dtype))
        c['w_stats']['adapter_abs'] = out.new_zeros(())
        return out,None,[]

    def _round(self, c, step, *args, **kwargs):
        c['exploration_step'] = step
        new, row = super()._round(c,step,*args,**kwargs)
        new['previous_output_scale'] = c['output_scale']
        new['output_scale'] = row['scale_weights'].detach().argmax(1)
        selected = row['weights'].detach() > 0
        if not bool((selected.sum(1) == self.step_top_k[step]).all()):
            raise RuntimeError('Executed expert set differs from configured Top-k')
        actual = row['weights'].topk(self.step_top_k[step],1).indices.sort(1).values
        with torch.no_grad():
            if step+1 in self.delta_steps:
                delta = (c['h']-c['prev']).float()
                rms = delta.square().mean((2,3,4)).sqrt()
                ratio = c['h'].detach().float().square().mean((2,3,4)).sqrt()/rms.clamp_min(1e-6)
                row['message_rms_mean'] = rms.mean()
                row['rms_ratio_mean'] = ratio.mean()
                row['rms_ratio_max'] = ratio.max()
                row['rms_ratio_p95'] = ratio.flatten().quantile(.95)
            else:
                for key in ('message_rms_mean','rms_ratio_mean','rms_ratio_max','rms_ratio_p95'):
                    row[key] = row['gate'].new_zeros(())
        self._exploration_rows.append((row,actual,c['output_scale'],c['previous_output_scale']))
        return new,row

    def forward(self, *args, **kwargs):
        if hasattr(self, '_exploration_rows'):
            raise RuntimeError('Overlapping exploration forward contexts')
        self._exploration_rows = []
        try:
            out = super().forward(*args, **kwargs)
            coe = out['coe']; diagnostics = coe['diagnostics']
            max_k = max(self.step_top_k)
            coe['selected_experts'] = torch.stack([
                F.pad(ids,(0,max_k-ids.shape[1]),value=-1) for _,ids,_,_ in self._exploration_rows],1)
            coe['top_k'] = max_k
            coe['top_k_per_step'] = self.step_top_k
            if self.step_top_k != (2,2,2,2):
                coe.pop('pair_ids',None)
                for i in range(1,5):
                    diagnostics.pop(f'step{i}_pair_vs_top2_disagreement_rate',None)
            for i,(row,ids,current_source,old_source) in enumerate(self._exploration_rows,1):
                for key in ('message_rms_mean','rms_ratio_mean','rms_ratio_max','rms_ratio_p95'):
                    diagnostics[f'step{i}_{key}'] = row[key]
                diagnostics[f'step{i}_delta_enabled'] = row['gate'].new_tensor(float(i in self.delta_steps))
                diagnostics[f'step{i}_delta_current_source_scale'] = current_source.float().mean()
                diagnostics[f'step{i}_delta_previous_source_scale'] = old_source.float().mean()
                diagnostics[f'step{i}_delta_same_scale_fraction'] = (
                    (current_source==old_source) & (current_source>=0)).float().mean()
            return out
        finally:
            del self._exploration_rows
