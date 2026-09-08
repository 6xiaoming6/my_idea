"""V22: bounded, sparse soft-region coarsening, not prediction-expert MoE.

Assignments include missing query cells; only observed cells contribute evidence.
Each expert prolongs through its OWN assignment before fine-grid fusion. Coarse
tokens are latent features, not physical sums or independently sensed values.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import ResidualSTBlock


class LocalCoarsener(nn.Module):
    """O(B*T*N*K*D) local assignment with K=(2*radius+1)^2, no N*Nc tensor."""

    def __init__(self, dim: int, stride: int, radius: int, temperature: float = 1.0):
        super().__init__()
        if stride < 2 or radius < 0 or temperature <= 0:
            raise ValueError("stride>=2, radius>=0 and temperature>0 are required")
        self.stride, self.radius, self.temperature = stride, radius, temperature
        self.query = nn.Linear(dim, min(dim, 16), bias=False) if radius else None
        self.key = nn.Linear(dim, min(dim, 16), bias=False) if radius else None

    def geometry(self, h, w, device):
        hc, wc = math.ceil(h / self.stride), math.ceil(w / self.stride)
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        cy, cx = y.flatten() // self.stride, x.flatten() // self.stride
        offsets = torch.arange(-self.radius, self.radius + 1, device=device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        ay, ax = cy[:, None] + dy.flatten(), cx[:, None] + dx.flatten()
        valid = (ay >= 0) & (ay < hc) & (ax >= 0) & (ax < wc)
        indices = (ay.clamp(0, hc - 1) * wc + ax.clamp(0, wc - 1)).long()
        # Grid-cell centers, measured in stride units (partial edge cells allowed).
        distance = ((y.flatten()[:, None] + .5) / self.stride - (ay + .5)).square()
        distance += ((x.flatten()[:, None] + .5) / self.stride - (ax + .5)).square()
        return indices, valid, distance, hc, wc

    @staticmethod
    def scatter(values, weights, indices, count):
        # values [BT,N,D], weights [BT,N,K]. Loop K avoids a BT*N*K*D buffer.
        out = values.new_zeros(values.shape[0], count, values.shape[-1])
        for k in range(indices.shape[1]):
            index = indices[:, k].view(1, -1, 1).expand(values.shape[0], -1, values.shape[-1])
            out.scatter_add_(1, index, values * weights[..., k:k + 1])
        return out

    @staticmethod
    def prolong(values, assignment):
        weights, indices = assignment
        out = values.new_zeros(values.shape[0], indices.shape[0], values.shape[-1])
        for k in range(indices.shape[1]):
            out = out + values[:, indices[:, k]] * weights[..., k:k + 1]
        return out

    def forward(self, features, evidence, mask):
        b, d, t, h, w = features.shape
        flat = lambda x: x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, -1).float()
        f, e, m = flat(features), flat(evidence), flat(mask)
        indices, valid, distance, hc, wc = self.geometry(h, w, features.device)
        if self.radius:
            # Queries and keys see only the sanitized input and its context.
            anchors = F.avg_pool2d(
                features.permute(0, 2, 1, 3, 4).reshape(b * t, d, h, w),
                self.stride, self.stride, ceil_mode=True, count_include_pad=False,
            ).flatten(2).transpose(1, 2)
            q, key = self.query(f.to(features.dtype)).float(), self.key(anchors).float()
            logits = torch.stack([(q * key[:, indices[:, k]]).sum(-1) for k in range(indices.shape[1])], -1)
            logits = logits / math.sqrt(q.shape[-1]) - distance[None]
            weights = (logits / self.temperature).masked_fill(~valid[None], -torch.inf).softmax(-1)
        else:
            weights = f.new_ones(b * t, h * w, 1)
        # FP32 sums and denominators remain stable with sparse support under AMP.
        observed_weights = weights * m
        mass = self.scatter(torch.ones_like(m), weights, indices, hc * wc)
        count = self.scatter(torch.ones_like(m), observed_weights, indices, hc * wc)
        mean = self.scatter(e, observed_weights, indices, hc * wc) / count.clamp_min(1e-6)
        second = self.scatter(e.square(), observed_weights, indices, hc * wc) / count.clamp_min(1e-6)
        variance = (second - mean.square()).clamp_min(0).mean(-1, keepdim=True)
        support = count / mass.clamp_min(1e-6)
        present = (count > 1e-6).float()
        # Observation centroid distinguishes equal-coverage masks with different geometry.
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=f.device), torch.linspace(-1, 1, w, device=f.device), indexing="ij")
        coords = torch.stack((yy, xx), -1).reshape(1, h * w, 2).expand(b * t, -1, -1)
        centroid = self.scatter(coords, observed_weights, indices, hc * wc) / count.clamp_min(1e-6)
        stats = torch.cat((support, present, variance, centroid), -1)
        assignment = (weights, indices)
        diagnostics = {
            "empty_fraction": (1 - present).mean(),
            "support": support.mean(),
            "assignment_entropy": -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean(),
            "displacement": (weights * distance[None]).sum(-1).mean(),
        }
        # Match geometric mass, NOT observation mass: missing regions must retain queries.
        uniform = valid.float() / valid.sum(-1, keepdim=True)
        reference = self.scatter(torch.ones_like(m), uniform[None].expand(b * t, -1, -1), indices, hc * wc)
        balance = ((mass / reference.clamp_min(1e-6) - 1).square()).mean()
        return mean, stats, assignment, (hc, wc), balance, diagnostics


class CoarseningScale(nn.Module):
    MODES = {"fixed", "fixed_stats", "single_local", "single_wide", "uniform", "moe", "moe_no_stats"}

    def __init__(self, dim, stride, cfg, groups, dropout):
        super().__init__()
        self.mode = cfg.get("mode", "moe")
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown V22 mode: {self.mode}")
        radii = list(cfg.get("radii", [0, 1, 2]))
        if len(radii) != 3 or radii[0] != 0 or not (0 < radii[1] < radii[2]):
            raise ValueError("V22 radii must be [0, local_radius, larger_radius]")
        self.coarseners = nn.ModuleList([LocalCoarsener(dim, stride, r, cfg.get("temperature", 1.0)) for r in radii])
        # Shared processing avoids confusing coarsening with additional prediction experts.
        self.coarse_input = nn.Conv3d(dim + 5, dim, 1)
        self.coarse_block = ResidualSTBlock(dim, groups, dropout)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.router = nn.Sequential(nn.Conv3d(dim + 7, dim, 1), nn.GELU(), nn.Conv3d(dim, 3, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.active = {"fixed": [0], "fixed_stats": [0], "single_local": [1], "single_wide": [2]}.get(self.mode, [0, 1, 2])
        # Instantiate in the same order for seed-matched shared initialization,
        # but do not advertise inactive parameters as trainable capacity.
        for i, module in enumerate(self.coarseners):
            if i not in self.active:
                module.requires_grad_(False)
        if self.mode not in {"moe", "moe_no_stats"}:
            self.router.requires_grad_(False)

    def forward(self, features, evidence, mask):
        b, d, t, h, w = features.shape
        restore = lambda z, hh, ww: z.reshape(b, t, hh, ww, -1).permute(0, 4, 1, 2, 3).contiguous()
        active = self.active
        views, supports, variances, penalties, diagnostics = [], [], [], [], {}
        for index in active:
            coarsener = self.coarseners[index]
            mean, stats, assignment, (hc, wc), penalty, diag = coarsener(features, evidence, mask)
            present = stats[..., 1:2]
            mean = mean * present + self.empty_token * (1 - present)
            supplied_stats = torch.zeros_like(stats) if self.mode == "fixed" else stats
            coarse = self.coarse_input(restore(torch.cat((mean, supplied_stats), -1), hc, wc))
            coarse = self.coarse_block(coarse)
            flat = coarse.permute(0, 2, 3, 4, 1).reshape(b * t, hc * wc, d).float()
            views.append(restore(coarsener.prolong(flat, assignment), h, w))
            supports.append(restore(coarsener.prolong(stats[..., :1], assignment), h, w))
            variances.append(restore(coarsener.prolong(stats[..., 2:3], assignment), h, w))
            penalties.append(penalty)
            diagnostics.update({f"e{index}_{key}": value for key, value in diag.items()})
        if len(active) == 1:
            gate = features.new_ones(b, 1, t, h, w)
        elif self.mode == "uniform":
            gate = features.new_full((b, 3, t, h, w), 1 / 3)
        else:
            evidence_stats = torch.cat([mask, *supports, *variances], 1)
            if self.mode == "moe_no_stats":
                evidence_stats = torch.zeros_like(evidence_stats)
            gate = self.router(torch.cat((features, evidence_stats.to(features.dtype)), 1)).float().softmax(1)
        mixed = sum(gate[:, i:i + 1] * view for i, view in enumerate(views))
        usage = gate.mean((0, 2, 3, 4))
        diagnostics.update({f"route_e{e}": usage[i] for i, e in enumerate(active)})
        diagnostics["route_entropy"] = -(gate * gate.clamp_min(1e-8).log()).sum(1).mean()
        return mixed.to(features.dtype), torch.stack(penalties).mean(), (usage * len(active) - 1).square().mean(), diagnostics


class V22CoarseningMoE(nn.Module):
    """Standalone encoder -> coarsening experts -> common-grid fusion -> decoder."""

    def __init__(self, cfg):
        super().__init__()
        opt = cfg["model"].get("v22", {})
        c, d = int(cfg["model"]["c_in"]), int(opt.get("dim", 48))
        self.normalization_floor = float(opt.get("normalization_floor", 1.0))
        strides = list(opt.get("strides", [2, 4]))
        if not strides or len(set(strides)) != len(strides) or d < 4 or self.normalization_floor <= 0:
            raise ValueError("V22 needs unique strides, dim>=4 and positive normalization_floor")
        groups, dropout = int(opt.get("num_groups", 8)), float(opt.get("dropout", .1))
        self.stem = nn.Sequential(nn.Conv3d(c + 3, d, 1), nn.GELU())
        self.encoder = ResidualSTBlock(d, groups, dropout)
        self.scales = nn.ModuleList([CoarseningScale(d, s, opt, groups, dropout) for s in strides])
        self.fusion = nn.Conv3d(d * (1 + len(strides)), d, 1)
        self.decoder = ResidualSTBlock(d, groups, dropout)
        self.head = nn.Conv3d(d, c, 1)

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def forward(self, x_f, m_f, x_m=None, m_m=None, x_c=None, m_c=None, **unused):
        if m_f.shape[1] != 1:
            raise ValueError("V22 currently requires a channel-shared fine mask [B,1,T,H,W]")
        # Do not read coarse targets, optional GT, or hidden placeholders.
        mask = m_f.float()
        observed = torch.where(mask.bool(), x_f.float(), torch.zeros_like(x_f, dtype=torch.float32))
        reduce = (2, 3, 4)
        count = mask.sum(reduce, keepdim=True).clamp_min(1)
        center = (observed.sum(reduce, keepdim=True) / count).detach()
        scale = (((observed - center).square() * mask).sum(reduce, keepdim=True) / count).sqrt()
        scale = scale.clamp_min(self.normalization_floor).detach()
        normalized = (observed - center) / scale * mask
        b, _, t, h, w = x_f.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x_f.device), torch.linspace(-1, 1, w, device=x_f.device), indexing="ij")
        coords = torch.stack((yy, xx)).view(1, 2, 1, h, w).expand(b, -1, t, -1, -1)
        evidence = self.stem(torch.cat((normalized, mask, coords), 1))
        fine = self.encoder(evidence)
        views, masses, balances, diagnostics = [], [], [], {}
        for stride, module in zip([s.coarseners[0].stride for s in self.scales], self.scales):
            view, mass, balance, diag = module(fine, evidence, mask)
            views.append(view)
            masses.append(mass)
            balances.append(balance)
            diagnostics.update({f"s{stride}_{key}": value for key, value in diag.items()})
        fused = self.decoder(fine + self.fusion(torch.cat([fine, *views], 1)))
        prediction = self.head(fused).float() * scale + center
        return {
            "x_hat_main": prediction, "h_st_aux": fused, "gates": {},
            "v22_mass_loss": torch.stack(masses).mean(),
            "v22_balance_loss": torch.stack(balances).mean(),
            "v22_center": center, "v22_scale": scale,
            "diagnostics": {"v22": diagnostics},
        }
