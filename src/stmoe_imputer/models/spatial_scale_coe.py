"""Spatial expert execution: legacy N5/N6 plus R-series controlled/free scales.

Fine-grid hidden state is retained between rounds. Coarse operations replace it
with the spatially upsampled expert result, without residual additions. Only the
selected resolutions and selected Top-2 experts are evaluated for each sample.
Soft scale variants execute both resolutions; legacy hard variants remain sparse.
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
        if (model.scale_mode not in ('fixed', 'adaptive', 'fixed_st', 'perturb', 'free', 'soft_start', 'soft_fusion') or model.scale_factor != 2
                or model.num_steps != 4 or model.coarse_rounds != 2
                or len(model.fixed_scales) != 4 or model.fixed_scales.count('coarse') != 2
                or model.fixed_scales.count('fine') != 2):
            raise ValueError('Scale experiments require four rounds, factor=2, and a two-coarse fixed reference')
        if (model.routing_mode != 'hard' or model.top_k != 2 or model.pair_mode != 'native'
                or model.state_update_mode != 'direct' or model.expert_sharing not in ('shared', 'per_step')
                or model.completion_feedback or model.use_shared or not model.use_routed
                or model.router_state != 'dynamic' or model.expert_state != 'dynamic'
                or model.fusion_mode != 'original' or model.acceptance != 'none'
                or model.routing_warmup_epochs or model.routing_transition_epochs):
            raise ValueError('Spatial scales require direct native Top-2 without feedback or warmup')
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
        if model.scale_mode in ('fixed', 'perturb'):
            model.scale_routers.requires_grad_(False)
        if model.scale_mode == 'soft_start':
            model.scale_warmup_epochs = int(scale.get('warmup_epochs', 5))
            model.scale_transition_epochs = int(scale.get('transition_epochs', 5))
            if model.scale_warmup_epochs < 1 or model.scale_transition_epochs < 0:
                raise ValueError('Scale soft start requires positive warmup and nonnegative transition epochs')
        if model.scale_mode == 'perturb':
            model.perturb_epochs = int(scale.get('perturb_epochs', 8))
            if model.perturb_epochs < 1:
                raise ValueError('perturb_epochs must be positive')
            generator = torch.Generator(device='cpu').manual_seed(int(scale.get('perturb_seed', 20260930)))
            # A persistent RNG buffer supports checkpoint continuation without
            # touching the model/dropout, loader, or mask random streams.
            model.register_buffer('_perturb_rng_state', generator.get_state())
        return model

    def _router_features(self, *args, **kwargs):
        result = super()._router_features(*args, **kwargs)
        self._scale_features = result
        return result

    def _expert_input(self, hidden, completion, mask, support, position, step):
        self._scale_inputs = (hidden, completion, mask, support, position)
        return super()._expert_input(hidden, completion, mask, support, position, step)

    def _scale_soft_mass(self):
        if self.scale_mode == 'soft_fusion':
            return 1.0
        if self.scale_mode != 'soft_start' or not self.training:
            return 0.0
        elapsed = self.routing_epoch - self.scale_warmup_epochs
        if elapsed <= 0:
            return 1.0
        return max(0.0, 1.0 - elapsed / (self.scale_transition_epochs + 1))

    def _choose_scale(self, step, reference):
        b = reference.shape[0]
        free_mode = self.scale_mode in ('free', 'soft_start', 'soft_fusion')
        if self.scale_mode in ('fixed', 'perturb'):
            if self.scale_mode == 'perturb' and self.training and self.routing_epoch <= self.perturb_epochs:
                if step == 0:
                    generator = torch.Generator(device='cpu')
                    generator.set_state(self._perturb_rng_state.cpu())
                    self._perturb_swap = (torch.rand(b, generator=generator) < .5).to(reference.device)
                    self._perturb_rng_state.copy_(generator.get_state().to(self._perturb_rng_state.device))
                # CCFF or FCCF, both with exactly two coarse operations.
                coarse = (torch.ones(b, device=reference.device, dtype=torch.long) if step == 1
                          else self._perturb_swap.long() if step == 2
                          else (~self._perturb_swap).long() if step == 0
                          else torch.zeros(b, device=reference.device, dtype=torch.long))
            else:
                coarse = torch.full((b,), self.fixed_scales[step] == 'coarse', device=reference.device, dtype=torch.long)
            weights = F.one_hot(coarse, 2).float()
            probs = weights
            free = torch.zeros(b, device=reference.device, dtype=torch.bool)
        else:
            remaining = self.coarse_rounds - self._coarse_used
            rounds_left = self.num_steps - step
            if free_mode:
                free = torch.ones(b, device=reference.device, dtype=torch.bool)
                budget = torch.stack((self._coarse_used.float()/self.num_steps,
                                      torch.full_like(remaining, rounds_left).float()/self.num_steps), dim=1)
            else:
                free = (remaining > 0) & (remaining < rounds_left)
                budget = torch.stack((remaining.float()/self.coarse_rounds,
                                      torch.full_like(remaining, rounds_left).float()/self.num_steps), dim=1)
            features = torch.cat((self._scale_features, budget.to(self._scale_features.dtype)), dim=1)
            logits = self.scale_routers[step](features).float()
            # Free selection has no quota; per-round tie priors preserve CCFF
            # at initialization. Old adaptive uses its original coarse prior.
            prior = [1e-3, 0.] if free_mode and step >= 2 else [0., 1e-3]
            logits = logits + logits.new_tensor(prior)
            allowed = (torch.ones_like(logits, dtype=torch.bool) if free_mode
                       else torch.stack((remaining < rounds_left, remaining > 0), dim=1))
            probs = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
            if self.scale_mode == 'fixed_st':
                coarse = torch.full((b,), self.fixed_scales[step] == 'coarse', device=reference.device, dtype=torch.long)
            else:
                coarse = probs.argmax(-1)
            hard = F.one_hot(coarse, 2).to(probs.dtype)
            weights = hard + (probs - probs.detach()) if self.training else hard
            if self.scale_mode in ('soft_start', 'soft_fusion'):
                soft_mass = self._scale_soft_mass()
                weights = soft_mass * probs + (1.0 - soft_mass) * weights
        # Track actual execution separately from the dominant (argmax) path.
        # Both soft branches execute even if a saturated probability is zero.
        executed = (torch.ones_like(weights, dtype=torch.bool) if self._scale_soft_mass() > 0
                    else weights.detach() != 0)
        self._scale_weight_history.append(weights.detach())
        self._scale_execution_history.append(executed)
        used = weights[:, 1].detach() if self.scale_mode in ('soft_start', 'soft_fusion') else coarse.detach()
        self._coarse_used = self._coarse_used + used
        self._scale_history.append(coarse.detach())
        self._scale_probability_history.append(probs.detach())
        self._scale_free_history.append(free.detach())
        return weights

    def _dispatch_weighted(self, unified, weights, step=0):
        scale_weights = self._choose_scale(step, unified)
        result = torch.zeros_like(unified)
        for scale_id in (0, 1):
            selected = torch.nonzero(self._scale_execution_history[-1][:, scale_id], as_tuple=False).flatten()
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
        self._scale_history, self._scale_free_history, self._scale_probability_history = [], [], []
        self._scale_weight_history, self._scale_execution_history = [], []
        try:
            output = super().forward(x_f, m_f, **kwargs)
            selected = torch.stack(self._scale_history, dim=1)
            if self.scale_mode not in ('free', 'soft_start', 'soft_fusion') and not bool((selected.sum(1) == self.coarse_rounds).all()):
                raise RuntimeError('Per-window coarse/fine budget violated')
            output['coe']['selected_scales'] = selected
            output['coe']['spatial_scale_mode'] = self.scale_mode
            output['coe']['scale_probabilities'] = torch.stack(self._scale_probability_history, dim=1)
            # New fields only for new modes: old logs and exact aggregation stay unchanged.
            if self.scale_mode in ('soft_start', 'soft_fusion'):
                output['coe']['scale_weights'] = torch.stack(self._scale_weight_history, dim=1)
                output['coe']['scale_executed'] = torch.stack(self._scale_execution_history, dim=1)
            diagnostics = output['diagnostics']['coe']
            for step in range(self.num_steps):
                diagnostics[f'step{step+1}_coarse_fraction'] = selected[:,step].float().mean()
                diagnostics[f'step{step+1}_coarse_probability'] = self._scale_probability_history[step][:,1].mean()
                diagnostics[f'step{step+1}_scale_free_fraction'] = self._scale_free_history[step].float().mean()
            diagnostics['coarse_rounds_mean'] = selected.sum(1).float().mean()
            coarse_count = selected.sum(1).float()
            diagnostics['expert_grid_equivalents'] = (self.top_k*(self.num_steps-coarse_count+coarse_count/self.scale_factor**2)).mean()
            if self.scale_mode in ('soft_start', 'soft_fusion'):
                execution = output['coe']['scale_executed'].float()
                diagnostics['scale_soft_mass'] = execution.new_tensor(self._scale_soft_mass())
                diagnostics['expert_grid_equivalents'] = self.top_k * (execution[..., 0] + execution[..., 1]/4).sum(1).mean()
                diagnostics['expert_execution_count'] = self.top_k * execution.sum((1, 2)).mean()
            for code in range(16):
                sequence = sum(selected[:,step]*(1 << step) for step in range(4))
                label = ''.join('c' if code & (1 << step) else 'f' for step in range(4))
                diagnostics[f'scale_path_{label}_fraction'] = (sequence == code).float().mean()
            return output
        finally:
            self._scale_inputs = None
            self._scale_features = None
            self._perturb_swap = None
            self._scale_weight_history = []
            self._scale_execution_history = []
