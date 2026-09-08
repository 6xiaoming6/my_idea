"""Full Idea-2: route assignments BEFORE pooling; complete region tokens with fine data.

Kept outside src to preserve historical V22 source fingerprints. Install explicitly
in the worker. Regions are overlapping soft regions around persistent anchors,
not a hard segmentation, a physical sum, or independently observed coarse truth.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from stmoe_imputer.models.blocks import ResidualSTBlock
from stmoe_imputer.models.v_single.v22_coarsening_moe import LocalCoarsener


def flatten(x):
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).float()


def restore(x, b, t, h, w):
    return x.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()


class AssignmentExpert(LocalCoarsener):
    """Only construct sparse region memberships; no private completion network."""

    def forward(self, features):
        b, d, t, h, w = features.shape
        indices, valid, distance, hc, wc = self.geometry(h, w, features.device)
        f = flatten(features)
        if self.radius:
            anchors = F.avg_pool2d(features.permute(0, 2, 1, 3, 4).reshape(b*t, d, h, w),
                                  self.stride, self.stride, ceil_mode=True, count_include_pad=False)
            anchors = anchors.flatten(2).transpose(1, 2)
            query = self.query(f.to(features.dtype)).float()
            key = self.key(anchors).float()
            logits = torch.stack([(query * key[:, indices[:, k]]).sum(-1)
                                  for k in range(indices.shape[1])], -1)
            logits = logits / math.sqrt(query.shape[-1]) - distance[None]
            weights = (logits / self.temperature).masked_fill(~valid[None], -torch.inf).softmax(-1)
        else:
            weights = f.new_ones(b*t, h*w, 1)
        return weights, indices, valid, (hc, wc)


class RegionGraphBlock(nn.Module):
    """Sparse content/centroid attention on region nodes, then time convolution.

Anchor neighbours define candidate edges, NOT regular-grid convolution of region
values. Supports a changing soft membership at each time with stable anchor IDs.
Complexity O(B*T*Nc*9*D), no dense Nf*Nc or Nc*Nc matrix.
"""

    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.geometry = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.out = nn.Linear(dim, dim)
        self.temporal = nn.Conv1d(dim, dim, 3, padding=1)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2*dim, dim))

    def forward(self, tokens, centroid, support, shape, b, t):
        hc, wc = shape
        y, x = torch.meshgrid(torch.arange(hc, device=tokens.device),
                              torch.arange(wc, device=tokens.device), indexing='ij')
        offsets = torch.arange(-1, 2, device=tokens.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        ay, ax = y.flatten()[:, None] + dy.flatten(), x.flatten()[:, None] + dx.flatten()
        valid = (ay >= 0) & (ay < hc) & (ax >= 0) & (ax < wc)
        indices = ay.clamp(0, hc-1)*wc + ax.clamp(0, wc-1)
        z = self.norm(tokens)
        q, k, v = self.q(z), self.k(z), self.v(z)
        logits = []
        for edge in range(9):
            idx = indices[:, edge]
            geometry = torch.cat((centroid[:, idx] - centroid, support[:, idx]), -1)
            score = (q * k[:, idx]).sum(-1) / math.sqrt(q.shape[-1])
            logits.append(score.float() + self.geometry(geometry.to(z.dtype)).squeeze(-1).float())
        weights = torch.stack(logits, -1).masked_fill(~valid[None], -torch.inf).softmax(-1)
        message = sum(weights[..., edge:edge+1] * v[:, indices[:, edge]] for edge in range(9))
        tokens = tokens + self.out(message.to(z.dtype))
        # Region IDs persist over time; space is NOT convolved as a rectangular image.
        n, d = tokens.shape[1:]
        temporal = tokens.reshape(b, t, n, d).permute(0, 2, 3, 1).reshape(b*n, d, t)
        temporal = self.temporal(temporal).reshape(b, n, d, t).permute(0, 3, 1, 2).reshape(b*t, n, d)
        tokens = tokens + temporal
        return tokens + self.ff(tokens)


class RoutedRegions(nn.Module):
    def __init__(self, dim, channels, stride, options):
        super().__init__()
        self.stride = stride
        self.routing = options.get('routing', 'moe')
        self.top_k = int(options.get('top_k', 3))
        if self.routing not in ('moe', 'uniform', 'fixed') or self.top_k not in (2, 3):
            raise ValueError('routing=moe/uniform/fixed; top_k must be 2 or 3')
        radii = options.get('radii', [0, 1, 2])
        if len(radii) != 3 or radii[0] != 0 or not 0 < radii[1] < radii[2]:
            raise ValueError('Expected radii=[0, local_radius, wider_radius]')
        self.experts = nn.ModuleList([AssignmentExpert(dim, stride, radius, options.get('temperature', 1.))
                                      for radius in radii])
        self.router = nn.Sequential(nn.Conv3d(dim+7, dim, 1), nn.GELU(), nn.Conv3d(dim, 3, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        if self.routing != 'moe':
            self.router.requires_grad_(False)
        if self.routing == 'fixed':
            self.experts[1:].requires_grad_(False)
        # Mean value (C), latent evidence (D), support/presence/variance (3),
        # region centroid (2), observed centroid displacement (2), log mass (1).
        self.encode = nn.Linear(channels+dim+8, dim)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.feedback = nn.Linear(dim, dim)
        self.complete = RegionGraphBlock(dim, options.get('dropout', .1))

    def forward(self, fine, evidence, values, mask, coords):
        b, d, t, h, w = fine.shape
        m, coords_flat = flatten(mask), flatten(coords)
        # Router evidence is constructed exclusively from sanitized observations.
        density = F.avg_pool3d(mask, (1, 3, 3), stride=1, padding=(0, 1, 1), count_include_pad=False)
        mean = F.avg_pool3d(values, (1, 3, 3), stride=1, padding=(0, 1, 1), count_include_pad=False) / density.clamp_min(1e-6)
        second = F.avg_pool3d(values.square(), (1, 3, 3), stride=1, padding=(0, 1, 1), count_include_pad=False) / density.clamp_min(1e-6)
        variance = (second-mean.square()).clamp_min(0).mean(1, keepdim=True)
        stats = torch.cat((mask, density, variance, coords, mean.mean(1, keepdim=True), (density > 0).float()), 1)
        probs = self.router(torch.cat((fine, stats.to(fine.dtype)), 1)).float().softmax(1) if self.routing == 'moe' else fine.new_full((b, 3, t, h, w), 1/3)
        gate = probs
        if self.routing == 'moe' and self.top_k < 3:
            selected = gate.topk(self.top_k, dim=1).indices
            gate = gate * torch.zeros_like(gate).scatter_(1, selected, 1.)
            gate = gate / gate.sum(1, keepdim=True).clamp_min(1e-8)
        if self.routing == 'fixed':
            gate = torch.cat((torch.ones_like(mask), torch.zeros_like(mask), torch.zeros_like(mask)), 1)
        edges, indices, references = [], [], []
        for e in ([0] if self.routing == 'fixed' else range(3)):
            a, idx, valid, shape = self.experts[e](fine)
            edges.append(a * flatten(gate[:, e:e+1]))
            indices.append(idx)
            # Mask-independent reference geometry prevents erasing unobserved queries.
            ref = valid.float() / valid.sum(-1, keepdim=True)
            references.append(ref[None].expand(b*t, -1, -1) / (1 if self.routing == 'fixed' else 3))
        weights, index = torch.cat(edges, -1), torch.cat(indices, -1)
        n = shape[0]*shape[1]
        scatter = lambda value, weight: LocalCoarsener.scatter(value, weight, index, n)
        mass = scatter(torch.ones_like(m), weights)
        observed_mass = scatter(torch.ones_like(m), weights*m)
        present = (observed_mass > 1e-6).float()
        pooled = scatter(torch.cat((flatten(values), flatten(evidence)), -1), weights*m) / observed_mass.clamp_min(1e-6)
        c = values.shape[1]
        raw_mean = pooled[..., :c]
        second = scatter(flatten(values).square(), weights*m) / observed_mass.clamp_min(1e-6)
        raw_variance = (second-raw_mean.square()).clamp_min(0).mean(-1, keepdim=True)
        centroid = scatter(coords_flat, weights) / mass.clamp_min(1e-6)
        observed_centroid = scatter(coords_flat, weights*m) / observed_mass.clamp_min(1e-6)
        displacement = (observed_centroid-centroid)*present
        support = observed_mass / mass.clamp_min(1e-6)
        token_stats = torch.cat((support, present, raw_variance, centroid, displacement, mass.log1p()), -1)
        tokens = self.encode(torch.cat((pooled, token_stats), -1).to(fine.dtype)) + self.empty_token*(1-present)
        reference_mass = scatter(torch.ones_like(m), torch.cat(references, -1))
        mass_loss = (mass/reference_mass.clamp_min(1e-6)-1).square().mean()
        usage = probs.mean((0, 2, 3, 4))
        balance = (usage*3-1).square().mean() if self.routing == 'moe' else mass_loss*0
        diagnostics = {'empty_fraction': (1-present).mean(), 'support': support.mean(),
                       'route_entropy': -(gate*gate.clamp_min(1e-8).log()).sum(1).mean(),
                       'region_mass_min': mass.min(), 'region_mass_max': mass.max(),
                       **{f'route_e{e}': gate[:, e].mean() for e in range(3)}}
        return dict(tokens=tokens, assignment=(weights, index), shape=shape, mass=mass,
                    observed_mass=observed_mass, mean=raw_mean, support=support,
                    centroid=centroid, mass_loss=mass_loss, balance=balance, diagnostics=diagnostics)

    def exchange(self, state, fine):
        b, _, t, h, w = fine.shape
        weights, indices = state['assignment']
        # This is inferred FEATURE feedback, never new physical observed evidence.
        feedback = LocalCoarsener.scatter(flatten(fine), weights, indices, state['mass'].shape[1])
        feedback = feedback / state['mass'].clamp_min(1e-6)
        tokens = state['tokens'] + self.feedback(feedback.to(fine.dtype))
        state['tokens'] = self.complete(tokens, state['centroid'], state['support'], state['shape'], b, t)
        lifted = LocalCoarsener.prolong(state['tokens'].float(), state['assignment'])
        return restore(lifted, b, t, h, w).to(fine.dtype)


class FullCoarseningMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        opt = cfg['model']['v22']['full_idea']
        self.channels = int(cfg['model']['c_in'])
        self.dim = int(opt.get('dim', 48))
        self.rounds = int(opt.get('exchange_rounds', 2))
        self.floor = float(opt.get('normalization_floor', 1.))
        self.strides = list(opt.get('strides', [2, 4]))
        if self.dim < 4 or self.rounds < 1 or self.floor <= 0 or not self.strides or len(set(self.strides)) != len(self.strides):
            raise ValueError('Invalid full Idea-2 dimensions/rounds/normalization/strides')
        groups, dropout = int(opt.get('num_groups', 8)), float(opt.get('dropout', .1))
        self.stem = nn.Sequential(nn.Conv3d(self.channels+3, self.dim, 1), nn.GELU())
        self.encoder = ResidualSTBlock(self.dim, groups, dropout)
        self.regions = nn.ModuleList([RoutedRegions(self.dim, self.channels, s, opt) for s in self.strides])
        self.fuse = nn.Conv3d(self.dim*(len(self.strides)+1), self.dim, 1)
        self.fine_complete = ResidualSTBlock(self.dim, groups, dropout)
        self.head = nn.Conv3d(self.dim, self.channels, 1)

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def forward(self, x_f, m_f, return_regions=False, **unused):
        if m_f.shape != (x_f.shape[0], 1, *x_f.shape[2:]):
            raise ValueError('Channel-shared mask [B,1,T,H,W] required')
        mask = m_f.float()
        observed = torch.where(mask.bool(), x_f.float(), torch.zeros_like(x_f, dtype=torch.float32))
        count = mask.sum((2, 3, 4), keepdim=True).clamp_min(1)
        center = (observed.sum((2, 3, 4), keepdim=True)/count).detach()
        scale = (((observed-center).square()*mask).sum((2, 3, 4), keepdim=True)/count).sqrt().clamp_min(self.floor).detach()
        values = ((observed-center)/scale)*mask
        b, _, t, h, w = values.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=values.device), torch.linspace(-1, 1, w, device=values.device), indexing='ij')
        coords = torch.stack((yy, xx)).view(1, 2, 1, h, w).expand(b, -1, t, -1, -1)
        evidence = self.stem(torch.cat((values, mask, coords), 1))
        fine = self.encoder(evidence)
        states = [module(fine, evidence, values, mask, coords) for module in self.regions]
        for _ in range(self.rounds):
            views = [module.exchange(state, fine) for module, state in zip(self.regions, states)]
            fine = self.fine_complete(fine + self.fuse(torch.cat([fine, *views], 1)))
        prediction = self.head(fine).float()*scale+center
        output = dict(x_hat_main=prediction, h_st_aux=fine, gates={}, v22_center=center, v22_scale=scale,
                      v22_mass_loss=torch.stack([s['mass_loss'] for s in states]).mean(),
                      v22_balance_loss=torch.stack([s['balance'] for s in states]).mean(),
                      diagnostics={'v22': {f's{stride}_{k}': v for stride, s in zip(self.strides, states)
                                           for k, v in s['diagnostics'].items()}})
        if return_regions:
            # Explicit opt-in for inspection/tests; do not retain all batches in logs.
            output['regions'] = states
        return output


def install_full_builder():
    from stmoe_imputer.models.registry import MODEL_REGISTRY
    # Worker-local compatibility with the unchanged V22 train/val/test engine.
    # FullCoarseningMoE requires explicit full_idea options, so old configs fail
    # loudly here instead of silently becoming a different architecture.
    MODEL_REGISTRY['v22_coarsening_moe'] = FullCoarseningMoE.from_config
