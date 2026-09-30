"""N5/N6: B3 experts at two spatial resolutions with an exact per-window budget.

Fine-grid hidden state is retained between rounds. Coarse operations replace it
with the spatially upsampled expert result, without residual additions. Only the
selected resolution and selected Top-2 experts are evaluated for each sample.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .temporal_spatial_coe import TemporalSpatialCoE


def spatial_pool(x, factor=2):
    return F.avg_pool3d(x, kernel_size=(1, factor, factor), stride=(1, factor, factor))


def observed_pool(values, mask, factor=2):
    """Observed-only mean plus fractional coverage; empty bins have mean zero."""
    coverage = spatial_pool(mask.float(), factor)
    clean = torch.where(mask.bool(), values, torch.zeros_like(values))
    mean = spatial_pool(clean.float(), factor) / coverage.clamp_min(1e-8)
    return mean.to(values.dtype), coverage.to(values.dtype)


def spatial_resize(x, size):
    b, c, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    flat = F.interpolate(flat, size=size, mode='bilinear', align_corners=False)
    return flat.reshape(b, t, c, *size).permute(0, 2, 1, 3, 4)


class SpatialScaleCoE(TemporalSpatialCoE):
    @classmethod
    def from_config(cls, cfg):
        model = super().from_config(cfg)
        scale = cfg['model']['coe']['spatial_scale']
        model.scale_mode = scale['mode']
        model.scale_factor = scale.get('factor', 2)
        model.coarse_rounds = scale.get('coarse_rounds', 2)
        model.fixed_scales = tuple(scale.get('fixed_scales', ['coarse','coarse','fine','fine']))
        if (model.scale_mode not in ('fixed', 'adaptive') or model.scale_factor != 2
                or model.num_steps != 4 or model.coarse_rounds != 2
                or len(model.fixed_scales) != 4 or model.fixed_scales.count('coarse') != 2
                or model.fixed_scales.count('fine') != 2):
            raise ValueError('N5/N6 require four rounds, two coarse/two fine, factor=2')
        if (model.routing_mode != 'hard' or model.top_k != 2 or model.pair_mode != 'native'
                or model.state_update_mode != 'direct' or model.expert_sharing != 'shared'
                or model.completion_feedback or model.use_shared or not model.use_routed
                or model.router_state != 'dynamic' or model.expert_state != 'dynamic'
                or model.fusion_mode != 'original' or model.acceptance != 'none'
                or model.routing_warmup_epochs or model.routing_transition_epochs):
            raise ValueError('Spatial scales require the B3 direct native Top-2 protocol')
        features = model.routers[0][0].normalized_shape[0]
        hidden = cfg['model']['coe'].get('router_hidden_dim', model.dim)
        # Both scale variants own the same heads/state dict, but fixed N5 does
        # not use or optimize them. Appending modules preserves B3 initialization.
        model.scale_routers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(features+2), nn.Linear(features+2, hidden),
                          nn.GELU(), nn.Linear(hidden, 2)) for _ in range(4)])
        for head in model.scale_routers:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if model.scale_mode == 'fixed':
            model.scale_routers.requires_grad_(False)
        return model

    def _router_features(self, *args, **kwargs):
        result = super()._router_features(*args, **kwargs)
        self._scale_features = result
        return result

    def _expert_input(self, hidden, completion, mask, support, position, step):
        self._scale_inputs = (hidden, completion, mask, support, position)
        return super()._expert_input(hidden, completion, mask, support, position, step)

    def _choose_scale(self, step, reference):
        b = reference.shape[0]
        if self.scale_mode == 'fixed':
            coarse = torch.full((b,), self.fixed_scales[step] == 'coarse', device=reference.device, dtype=torch.long)
            weights = F.one_hot(coarse, 2).float()
            free = torch.zeros(b, device=reference.device, dtype=torch.bool)
        else:
            remaining = self.coarse_rounds - self._coarse_used
            rounds_left = self.num_steps - step
            free = (remaining > 0) & (remaining < rounds_left)
            budget = torch.stack((remaining.float()/self.coarse_rounds,
                                  torch.full_like(remaining, rounds_left).float()/self.num_steps), dim=1)
            features = torch.cat((self._scale_features, budget.to(self._scale_features.dtype)), dim=1)
            logits = self.scale_routers[step](features).float()
            # Zero-initialized heads reproduce N5's coarse/coarse/fine/fine
            # choice. A fixed small prior breaks ties, not a warmup schedule.
            logits = logits + logits.new_tensor([0., 1e-3])
            allowed = torch.stack((remaining < rounds_left, remaining > 0), dim=1)
            probs = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
            coarse = probs.argmax(-1)
            hard = F.one_hot(coarse, 2).to(probs.dtype)
            weights = hard + (probs - probs.detach()) if self.training else hard
        self._coarse_used = self._coarse_used + coarse.detach()
        self._scale_history.append(coarse.detach())
        self._scale_free_history.append(free.detach())
        return weights

    def _dispatch_weighted(self, unified, weights, step=0):
        scale_weights = self._choose_scale(step, unified)
        result = torch.zeros_like(unified)
        for scale_id in (0, 1):
            selected = torch.nonzero(scale_weights[:, scale_id].detach() != 0, as_tuple=False).flatten()
            if not selected.numel():
                continue
            expert_weights = weights.index_select(0, selected)
            if scale_id == 0:
                inputs = unified.index_select(0, selected)
            else:
                hidden, completion, mask, support, position = [x.index_select(0, selected) for x in self._scale_inputs]
                values, coverage = observed_pool(completion, mask, self.scale_factor)
                # Empty bins retain a learned initial estimate, not an observation.
                fallback = spatial_pool(completion, self.scale_factor)
                values = torch.where(coverage > 0, values, fallback)
                inputs = self.state_norm(self.state_projection(torch.cat([
                    spatial_pool(hidden, self.scale_factor), values, coverage,
                    spatial_pool(support, self.scale_factor), spatial_pool(position, self.scale_factor)
                ], dim=1)))
            updates = super()._dispatch_weighted(inputs, expert_weights, step)
            if scale_id == 1:
                updates = spatial_resize(updates, unified.shape[-2:])
            coefficient = scale_weights.index_select(0, selected)[:, scale_id, None, None, None, None]
            result = result.index_add(0, selected, (updates*coefficient.to(updates.dtype)).to(result.dtype))
        return result

    def forward(self, x_f, m_f, **kwargs):
        if any(n % self.scale_factor for n in x_f.shape[-2:]):
            raise ValueError('Spatial shape must be divisible by scale factor')
        self._coarse_used = torch.zeros(x_f.shape[0], device=x_f.device, dtype=torch.long)
        self._scale_history, self._scale_free_history = [], []
        try:
            output = super().forward(x_f, m_f, **kwargs)
            selected = torch.stack(self._scale_history, dim=1)
            if not bool((selected.sum(1) == self.coarse_rounds).all()):
                raise RuntimeError('Per-window coarse/fine budget violated')
            output['coe']['selected_scales'] = selected
            output['coe']['spatial_scale_mode'] = self.scale_mode
            diagnostics = output['diagnostics']['coe']
            for step in range(self.num_steps):
                diagnostics[f'step{step+1}_coarse_fraction'] = selected[:,step].float().mean()
                diagnostics[f'step{step+1}_scale_free_fraction'] = self._scale_free_history[step].float().mean()
            diagnostics['coarse_rounds_mean'] = selected.sum(1).float().mean()
            diagnostics['expert_grid_equivalents'] = selected.new_tensor(
                self.top_k*(self.num_steps-self.coarse_rounds+self.coarse_rounds/self.scale_factor**2), dtype=torch.float32)
            for code in (3,5,6,9,10,12):
                sequence = sum(selected[:,step]*(1 << step) for step in range(4))
                label = ''.join('c' if code & (1 << step) else 'f' for step in range(4))
                diagnostics[f'scale_path_{label}_fraction'] = (sequence == code).float().mean()
            return output
        finally:
            self._scale_inputs = None
            self._scale_features = None
