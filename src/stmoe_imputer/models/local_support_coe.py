"""C3: region-sparse chains with executed-path observation-support state.

Support denotes structural value-path reachability conditional on the routes;
it is not uncertainty, calibrated confidence, or unique observation counts.
Only mask operators are evaluated for all candidates, never neural experts.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_spatial_coe import TemporalSpatialCoE, compute_observation_support


def split_regions(x, region_size):
    b, c, t, h, w = x.shape
    rh, rw = region_size
    if h % rh or w % rw:
        raise ValueError('Spatial dimensions must be divisible by local region_size')
    return x.reshape(b, c, t, h // rh, rh, w // rw, rw).permute(
        0, 3, 5, 1, 2, 4, 6).reshape(-1, c, t, rh, rw)


def merge_regions(x, layout, region_size):
    b, h, w = layout
    rh, rw = region_size
    _, c, t, _, _ = x.shape
    return x.reshape(b, h // rh, w // rw, c, t, rh, rw).permute(
        0, 3, 4, 1, 5, 2, 6).reshape(b, c, t, h, w)


def footprint(expert):
    # Read the actual spatial/temporal operator rather than assuming name-specific offsets.
    if hasattr(expert, 'qkv'):
        return None  # Full temporal attention, independently at each cell.
    for module in expert.modules():
        if isinstance(module, nn.Conv3d) and module.kernel_size != (1, 1, 1):
            return module.kernel_size, module.dilation, module.padding
    raise ValueError('No supported mixing operator found')


def transport_support(reach, density, expert):
    """Boolean union and normalized mass along an expert's actual receptive field."""
    geometry = footprint(expert)
    if geometry is None:
        return (reach.amax(2, keepdim=True).expand_as(reach),
                density.mean(2, keepdim=True).expand_as(density))
    kernel, dilation, padding = geometry
    c = reach.shape[1]
    filt = torch.ones((c, 1, *kernel), device=reach.device, dtype=torch.float32)
    with torch.autocast(device_type=reach.device.type, enabled=False):
        reachable = F.conv3d(reach.float(), filt, padding=padding, dilation=dilation, groups=c) > 0
        mass = F.conv3d(density.float(), filt, padding=padding, dilation=dilation, groups=c)
        count = F.conv3d(torch.ones_like(density, dtype=torch.float32), filt,
                         padding=padding, dilation=dilation, groups=c)
    return reachable.float(), mass / count.clamp_min(1)


class SupportRouter(nn.Module):
    """Keep the native regional router; add a zero-initialized support correction."""
    def __init__(self, base, support_dim, hidden_dim):
        super().__init__()
        self.base = base
        self.base_dim = base[0].normalized_shape[0]
        self.support_head = nn.Sequential(nn.Linear(support_dim, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, base[-1].out_features))
        nn.init.zeros_(self.support_head[-1].weight)
        nn.init.zeros_(self.support_head[-1].bias)

    def forward(self, features):
        return self.base(features[:, :self.base_dim]) + self.support_head(features[:, self.base_dim:])


class LocalSupportCoE(TemporalSpatialCoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.routing_mode != 'hard' or self.top_k != 2 or self.pair_mode != 'native'
                or self.state_update_mode != 'direct' or self.use_shared or not self.use_routed
                or self.router_state != 'dynamic' or self.expert_state != 'dynamic'
                or self.acceptance != 'none'
                or self.router_features != 'legacy' or self.previous_expert_context
                or self.global_route_weights or any(v is not None for v in self.fixed_expert_steps)
                or self.routing_warmup_epochs or self.routing_transition_epochs):
            raise ValueError('C3 requires native hard Top-2, direct dynamic states, no warmup or gates')
        self.region_size = (8, 8)
        self.support_evolution = True
        self.support_enabled = True
        self.routers = nn.ModuleList([
            SupportRouter(router, (4 + 2 * self.num_experts) * self.c_in, router[1].out_features)
            for router in self.routers])

    @classmethod
    def from_config(cls, cfg):
        local = cfg['model']['coe']['local_routing']
        model = super().from_config(cfg)
        region = local.get('region_size', [8, 8])
        if len(region) != 2 or any(type(v) is not int or v < 1 for v in region):
            raise ValueError('local_routing.region_size requires two positive integers')
        model.region_size = tuple(region)
        model.support_evolution = bool(local.get('support_evolution', True))
        model.support_enabled = bool(local.get('support_enabled', True))
        return model

    def _split(self, x):
        return split_regions(x, self.region_size)

    def _merge(self, x):
        return merge_regions(x, self._layout, self.region_size)

    def _position(self, x):
        return self._split(super()._position(self._merge(x)))

    def _observation_support(self, mask):
        return self._split(compute_observation_support(
            self._merge(mask), self.temporal_kernel, self.spatial_kernel)).to(mask.dtype)

    def _router_features(self, hidden, values, support_summary, change, missing, pattern_summary=None):
        native = super()._router_features(hidden, values, support_summary, change, missing, pattern_summary)
        with torch.no_grad():
            candidates = [transport_support(self._reach, self._density, expert)
                          for expert in self.routed_experts()]
            self._candidates = candidates
            reach, density = self._split(self._reach), self._split(self._density)
            features = [reach.mean((2, 3, 4)), density.mean((2, 3, 4)),
                        self._missing_pool(reach, missing), self._missing_pool(density, missing)]
            for next_reach, next_density in candidates:
                features.extend((self._missing_pool(self._split(next_reach) - reach, missing),
                                 self._missing_pool(self._split(next_density) - density, missing)))
            extra = torch.cat(features, 1).to(native.dtype)
            if not self.support_enabled:
                extra = torch.zeros_like(extra)
        return torch.cat((native, extra), 1)

    def _dispatch_weighted(self, unified, weights, step=0):
        """Batch selected region/expert pairs by exact clipped halo shape.

        No padded fake cells pass through biased pointwise layers at global edges.
        Every crop reads the same pre-round global state. Only cropped cores commit.
        """
        full = self._merge(unified)
        _, h, w = self._layout
        rh, rw = self.region_size
        nr = (h // rh) * (w // rw)
        result = torch.zeros_like(unified)
        executed = 0
        halo_points = 0
        for eid, expert in enumerate(self.routed_experts(step)):
            ids = torch.nonzero(weights[:, eid].detach() != 0, as_tuple=False).flatten().cpu().tolist()
            if not ids:
                continue
            geometry = footprint(expert)
            hy, hx = (0, 0) if geometry is None else geometry[2][1:]
            groups = defaultdict(list)
            for idx in ids:
                sample, region = divmod(idx, nr)
                iy, ix = divmod(region, w // rw)
                y, x = iy * rh, ix * rw
                ya, yb, xa, xb = max(0, y-hy), min(h, y+rh+hy), max(0, x-hx), min(w, x+rw+hx)
                key = (yb-ya, xb-xa, y-ya, x-xa)
                groups[key].append((idx, sample, ya, yb, xa, xb))
            for (ph, pw, cy, cx), entries in groups.items():
                selected = torch.tensor([v[0] for v in entries], device=unified.device)
                inputs = torch.stack([full[b, :, :, ya:yb, xa:xb] for _, b, ya, yb, xa, xb in entries])
                updates = expert(inputs)[..., cy:cy+rh, cx:cx+rw]
                coeff = weights.index_select(0, selected)[:, eid, None, None, None, None].to(updates.dtype)
                result = result.index_add(0, selected, (updates * coeff).to(result.dtype))
                executed += len(entries)
                halo_points += len(entries) * ph * pw
        self._execution.append((executed, halo_points / (unified.shape[0] * rh * rw)))
        with torch.no_grad():
            next_reach = torch.zeros_like(self._split(self._reach))
            next_density = torch.zeros_like(next_reach)
            for eid, (candidate_r, candidate_d) in enumerate(self._candidates):
                chosen = (weights[:, eid].detach() != 0)[:, None, None, None, None]
                next_reach = torch.maximum(next_reach, self._split(candidate_r) * chosen)
                next_density += self._split(candidate_d) * weights[:, eid].detach()[:, None, None, None, None]
            if self.support_evolution:
                self._reach = torch.maximum(self._merge(next_reach), self._original_mask)
                self._density = torch.where(self._original_mask.bool(), 1., self._merge(next_density)).clamp(0, 1)
            self._support_history.append((self._reach.clone(), self._density.clone()))
        return result

    def forward(self, x_f, m_f, **kwargs):
        if kwargs:
            raise ValueError('C3 does not support forced global partner/pair overrides')
        self._layout = (x_f.shape[0], x_f.shape[-2], x_f.shape[-1])
        self._original_mask = m_f.expand_as(x_f).float()
        self._reach = self._original_mask.clone()
        self._density = self._original_mask.clone()
        self._execution, self._support_history = [], []
        output = super().forward(self._split(x_f), self._split(m_f))
        coe = output['coe']
        # Route records keep the region batch; task predictions keep the window batch.
        for key in ('x_hat_main', 'h_st_aux'):
            output[key] = self._merge(output[key])
        for key in ('predictions', 'completions', 'changes', 'candidate_predictions',
                    'candidate_completions', 'candidate_changes'):
            coe[key] = [self._merge(v) for v in coe[key]]
        for key in ('initial_prediction', 'initial_completion'):
            coe[key] = self._merge(coe[key])
        coe['acceptance_weights'] = torch.stack([
            self._merge(v) for v in coe['acceptance_weights'].unbind(1)], 1)
        coe['region_size'] = self.region_size
        coe['routing_region_count'] = (x_f.shape[-2] // self.region_size[0]) * (x_f.shape[-1] // self.region_size[1])
        coe['support_history'] = self._support_history
        diagnostics = coe['diagnostics']
        diagnostics['regions_per_window'] = x_f.new_tensor(coe['routing_region_count'])
        missing = 1 - self._original_mask
        for step, ((reach, density), (executed, ratio)) in enumerate(zip(self._support_history, self._execution), 1):
            diagnostics[f'step{step}_support_reachable_missing'] = self._missing_pool(reach, missing).mean()
            diagnostics[f'step{step}_support_density_missing'] = self._missing_pool(density, missing).mean()
            diagnostics[f'step{step}_executed_experts_per_region'] = x_f.new_tensor(executed / (x_f.shape[0] * coe['routing_region_count']))
            diagnostics[f'step{step}_expert_area_ratio'] = x_f.new_tensor(ratio)
            pairs = coe['pair_ids'][:, step-1].reshape(x_f.shape[0], -1)
            diagnostics[f'step{step}_within_window_pair_disagreement'] = (pairs != pairs[:, :1]).float().mean()
        return output
