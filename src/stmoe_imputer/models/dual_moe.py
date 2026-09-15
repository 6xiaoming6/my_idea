"""Learned-region dual MoE with opt-in observation-preserving target readout.

Historical source-gated, anchored and regular-bin designs retain their behavior.
Target Top-K mixes aligned organization representations, never input support.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResidualSTBlock


def _matmul(a, b):
    with torch.autocast(device_type=a.device.type, enabled=False):
        return torch.matmul(a.float(), b.float())


def _flat(x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b*t, h*w, c)


def _pool(x: torch.Tensor, size: int) -> torch.Tensor:
    # Explicit padding works even when a grid is narrower than the kernel.
    pad = size//2
    def average(z):
        return F.avg_pool3d(F.pad(z, (pad, pad, pad, pad)), (1, size, size), stride=1)
    return average(x)/average(torch.ones_like(x[:, :1])).clamp_min(1e-6)


class PointRouter(nn.Module):
    """Legacy dense gates or explicit Top-K weights with full-softmax metadata.

    Top-K sparsifies routing, not this backbone's batched expert computation.
    """
    def __init__(self, channels: int, experts: int, mode: str, top_k=None):
        super().__init__()
        if mode not in {"learned", "static", "uniform", "topk"}:
            raise ValueError(f"Unknown router mode: {mode}")
        if mode == 'topk' and (type(top_k) is not int or not 1 <= top_k <= experts):
            raise ValueError('Top-K routing requires an explicit integer top_k in [1, experts]')
        self.mode = mode
        self.top_k = top_k
        self.routing_info = None
        self.head = nn.Conv3d(channels, experts, 1)
        self.logits = nn.Parameter(torch.zeros(1, experts, 1, 1, 1))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if mode == 'topk':
            # Break exact all-zero ties for different inputs; do not add random
            # routing at inference. Historical dense initialization is unchanged.
            # Do not shift initialization of the shared downstream B01 modules.
            with torch.random.fork_rng(devices=[]):
                nn.init.normal_(self.head.weight, std=1e-3)
        self.head.requires_grad_(mode in {"learned", "topk"})
        self.logits.requires_grad_(mode == "static")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"learned", "topk"}:
            logits = self.head(x)
        elif self.mode == "static":
            logits = self.logits.expand(x.shape[0], -1, *x.shape[2:])
        else:
            logits = torch.zeros_like(self.logits).expand(x.shape[0], -1, *x.shape[2:])
        return self.route_logits(logits)

    def route_logits(self, logits):
        probabilities = logits.float().softmax(dim=1)
        self.routing_info = None
        if self.mode != 'topk':
            return probabilities
        indices = logits.float().topk(self.top_k, dim=1).indices
        selected = torch.zeros_like(probabilities).scatter_(1, indices, 1.)
        # Softmax over selected logits avoids underflow at extreme finite logits.
        weights = logits.float().masked_fill(~selected.bool(), -torch.inf).softmax(dim=1)
        if self.top_k == 1:
            # Renormalized Top-1 is constant 1 and otherwise has NO task gradient.
            # Forward remains hard one-hot; backward uses selected full-softmax
            # probabilities (explicit straight-through approximation, not exact).
            surrogate = probabilities*selected
            weights = weights.detach()+(surrogate-surrogate.detach())
        self.routing_info = {'probabilities': probabilities, 'selected': selected.detach(), 'top_k': self.top_k}
        return weights


class TargetReadoutRouter(nn.Module):
    """Score aligned organization representations for each destination query.

    All experts see observations BEFORE this gate. Shared additive attention
    uses the candidate content, not an arbitrary coarse-node/expert index.
    Top-K is sparse mixing only; regional computation remains dense.
    """
    route_logits = PointRouter.route_logits

    def __init__(self, dim, key_dim, experts, mode, top_k):
        super().__init__()
        if mode not in {'uniform', 'learned', 'topk', 'static'}:
            raise ValueError('Target readout supports uniform, learned, topk or static routing')
        if mode == 'topk' and (type(top_k) is not int or not 1 <= top_k <= experts):
            raise ValueError('Target Top-K must be an integer in [1, experts]')
        self.mode, self.top_k, self.experts = mode, top_k, experts
        self.routing_info = None
        self.query_norm, self.key_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.query = nn.Linear(dim, key_dim, bias=False)
        self.key = nn.Linear(dim, key_dim, bias=False)
        self.support = nn.Linear(3, key_dim, bias=False)
        self.score = nn.Linear(key_dim, 1, bias=False)
        nn.init.normal_(self.score.weight, std=.02)
        self.requires_grad_(mode not in {'uniform', 'static'})
        # Opt-in global-weight control only. Existing modes retain exactly their
        # old state dict and RNG sequence; each scale has its own constant gate.
        if mode == 'static':
            self.logits = nn.Parameter(torch.zeros(1, experts, 1, 1, 1))

    def forward(self, fine, candidates, support):
        b, d, t, h, w = fine.shape
        if self.mode == 'uniform':
            logits = fine.new_zeros(b, self.experts, t, h, w)
        elif self.mode == 'static':
            logits = self.logits.expand(b, -1, t, h, w)
        else:
            query = fine.permute(0, 2, 3, 4, 1).reshape(b, 1, t, h*w, d)
            scores = self.score(torch.tanh(self.query(self.query_norm(query))
                                           + self.key(self.key_norm(candidates))
                                           + self.support(support.detach())))
            logits = scores.reshape(b, self.experts, t, h, w)
        gates = self.route_logits(logits)
        if self.routing_info is not None:
            self.routing_info['valid_domain'] = 'missing'
        return gates



class LatentRegionExpert(nn.Module):
    """Global learned memberships; no centers, radius masks or fixed regions."""
    def __init__(self, dim, nodes, key_dim, temperature):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(nodes, key_dim))
        self.query = nn.Sequential(nn.Linear(2*dim+3, dim), nn.GELU(), nn.Linear(dim, key_dim))
        self.temperature = temperature

    def forward(self, inputs, nodes):
        q, keys = self.query(inputs).float(), self.slots[:nodes].float()
        return (_matmul(q, keys.t())/(math.sqrt(keys.shape[-1])*self.temperature)).softmax(-1)


class RegionInteraction(nn.Module):
    """Permutation-equivariant node attention and shared temporal processing."""
    def __init__(self, dim, dropout):
        super().__init__()
        heads = next(h for h in (4, 2, 1) if dim % h == 0)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))

    def forward(self, x):
        b, e, t, k, d = x.shape
        z = x.reshape(b*e*t, k, d)
        q = self.norm(z)
        z = (z+self.attention(q, q, q, need_weights=False)[0]).reshape(b, e, t, k, d)
        temporal = z.permute(0, 1, 3, 4, 2).reshape(b*e*k, d, t)
        temporal = self.temporal(temporal).reshape(b, e, k, d, t).permute(0, 1, 4, 2, 3)
        z = z+temporal
        return z+self.ff(z)


def restore_regions(assignment, regions):
    """Each expert uses its own [B,E,T,N,K] mapping, before mixing experts."""
    return _matmul(assignment, regions)


class ObservationAggregationMoE(nn.Module):
    def __init__(self, dim, nodes, key_dim, experts, mode, geometry, temperature, top_k=None):
        super().__init__()
        self.nodes, self.geometry = nodes, geometry
        self.experts = nn.ModuleList([LatentRegionExpert(dim, nodes, key_dim, temperature) for _ in range(experts)])
        self.router = PointRouter(dim+6, experts, mode, top_k)

    def forward(self, features, mask, normalized_values):
        b, dim, t, h, w = features.shape
        n, k = h*w, min(self.nodes, h*w)
        d3, d7, d15 = (_pool(mask.float(), size) for size in (3, 7, 15))
        context = _pool(features.float()*mask, 7)/d7.clamp_min(1e-6)
        dx = F.pad(d7[..., 1:]-d7[..., :-1], (0, 1))
        dy = F.pad(d7[..., 1:, :]-d7[..., :-1, :], (0, 0, 0, 1))
        geometry = torch.cat((mask, d3, d7, d15, dx, dy), 1)
        gates = self.router(torch.cat((context, geometry if self.geometry else torch.zeros_like(geometry)), 1))
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=features.device),
                              torch.linspace(-1, 1, w, device=features.device), indexing="ij")
        coords = torch.stack((y, x), 0)[None, :, None].expand(b, 2, t, h, w)
        # Coordinates are scoring inputs, not prescribed region boundaries.
        inputs = _flat(torch.cat((features, context, coords, mask), 1))
        a = torch.stack([expert(inputs, k).reshape(b, t, n, k) for expert in self.experts], 1)
        source = gates.reshape(b, len(self.experts), t, n, 1)*mask.reshape(b, 1, t, n, 1)
        weights = a*source
        mass = weights.sum(-2)
        values = features.float().permute(0, 2, 3, 4, 1).reshape(b, 1, t, n, dim)
        pooled = _matmul(weights.transpose(-1, -2), values)/mass[..., None].clamp_min(1e-6)
        # Keep each expert's regions separate. Support is not calibrated confidence.
        with torch.no_grad():
            neff = mass.detach().square()/weights.detach().square().sum(-2).clamp_min(1e-12)
            z = normalized_values.permute(0, 2, 3, 4, 1).reshape(b, 1, t, n, -1).float()
            mean = _matmul(weights.detach().transpose(-1, -2), z)/mass.detach()[..., None].clamp_min(1e-6)
            second = _matmul(weights.detach().transpose(-1, -2), z.square())/mass.detach()[..., None].clamp_min(1e-6)
            var = (second-mean.square()).clamp_min(0).mean(-1)
            nominal = n/k
            support = torch.stack((mass.detach()/(mass.detach()+nominal),
                                   neff/(neff+nominal), var/(1+var)), -1)
        # An observed-source mixture prior combines expert outputs AFTER readout.
        # Missing-source gates never affect pooling or this prior. A, not alpha,
        # supplies the learned region correspondence for missing queries.
        observed_count = mask.reshape(b, 1, t, n, 1).sum(-2)
        prior = source.sum(-2)/observed_count.clamp_min(1)
        prior = torch.where(observed_count > 0, prior, torch.full_like(prior, 1/len(self.experts)))
        # Optional mutual-information partition regularizer, without labels.
        m = mask.reshape(b, 1, t, n, 1)
        count = m.sum(-2)
        marginal = (a*m).sum(-2)/count.clamp_min(1)
        conditional = (-(a*a.clamp_min(1e-8).log()).sum(-1)*m[..., 0]).sum(-1)/count[..., 0].clamp_min(1)
        marginal_entropy = -(marginal*marginal.clamp_min(1e-8).log()).sum(-1)
        valid = (count[..., 0] > 0).expand_as(conditional)
        partition = ((conditional-marginal_entropy)*valid).sum()/valid.sum().clamp_min(1)
        return {"features": pooled, "support": support, "mass": mass.detach(),
                "effective_count": neff, "gates": gates, "assignment": a, "prior": prior,
                "partition_loss": partition, "routing_info": self.router.routing_info}


class ScaleCompletionExpert(nn.Module):
    def __init__(self, dim, channels, groups, dropout, depth):
        super().__init__()
        self.input = nn.Linear(dim+3, dim)
        self.blocks = nn.Sequential(*[RegionInteraction(dim, dropout) for _ in range(depth)])
        self.condition = nn.Sequential(nn.Conv3d(2*dim, dim, 1), nn.GELU())
        self.head = nn.Conv3d(dim, channels, 1)

    def forward(self, aggregation, fine, readout_router=None):
        regions = self.blocks(self.input(torch.cat((aggregation["features"], aggregation["support"]), -1)))
        restored = restore_regions(aggregation["assignment"], regions)
        prior = aggregation["prior"][..., None]
        restored_support = restore_regions(aggregation["assignment"].detach(), aggregation["support"])
        if readout_router is not None:
            # A_e restores each expert to the SAME fine coordinates first;
            # coarse node indices from different experts are never mixed.
            gates = readout_router(fine, restored, restored_support)
            b, e, t, h, w = gates.shape
            prior = gates.reshape(b, e, t, h*w, 1)
            aggregation['source_gates'] = aggregation['gates']
            aggregation['gates'] = gates
            aggregation['routing_info'] = readout_router.routing_info
        aligned = (restored*prior).sum(1)
        b, d, t, h, w = fine.shape
        aligned = aligned.reshape(b, t, h, w, d).permute(0, 4, 1, 2, 3).contiguous()
        support = (restored_support*prior.detach()).sum(1)
        support = support.reshape(b, t, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
        hidden = self.condition(torch.cat((aligned, fine), 1))
        return self.head(hidden), hidden, support

class MultiScaleCompletionHead(nn.Module):
    """Independent pointwise expert over already contextualized multi-scale features."""
    def __init__(self, inputs, hidden, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(inputs, hidden, 1), nn.GELU(),
                                 nn.Conv3d(hidden, channels, 1))

    def forward(self, x):
        return self.net(x).float()


class ContextCompletionHead(MultiScaleCompletionHead):
    """Local spatial + bidirectional temporal mixing of contextual hidden features.

    Depthwise separable convolutions avoid quadratic attention on the fine grid.
    The shared expert/router are unchanged; only routed experts opt into this.
    Dilation changes receptive field without changing parameter count.
    """
    def __init__(self, inputs, hidden, channels, dilation=1):
        super().__init__(inputs, hidden, channels)
        # Preserve the base MLP and subsequent experts' initialization.
        with torch.random.fork_rng(devices=[]):
            self.context = nn.Sequential(
                nn.Conv3d(hidden, hidden, (1, 3, 3),
                          padding=(0, dilation, dilation), dilation=(1, dilation, dilation), groups=hidden),
                nn.GELU(),
                nn.Conv3d(hidden, hidden, (3, 1, 1),
                          padding=(dilation, 0, 0), dilation=(dilation, 1, 1), groups=hidden),
                nn.GELU(), nn.Conv3d(hidden, hidden, 1))

    def forward(self, x):
        hidden = self.net[1](self.net[0](x))
        return self.net[2](hidden + 0.1*self.context(hidden)).float()


class DualMoEBackbone(nn.Module):
    """V23: a new backbone, not a wrapper around the V14 correction network."""
    architecture = "dual_moe"

    def __init__(self, channels: int, options: dict):
        super().__init__()
        dim = int(options.get("dim", 32))
        nodes = options.get("coarse_nodes", [32, 8])
        experts = options.get("aggregation_experts", 3)
        temperature = float(options.get("assignment_temperature", 0.5))
        if "strides" in options:
            raise ValueError("Legacy strides/radius config is invalid for learned regions; use coarse_nodes")
        key_dim = int(options.get("key_dim", 8))
        depth = int(options.get("depth", 1))
        groups = int(options.get("num_groups", 4))
        dropout = float(options.get("dropout", 0.0))
        if channels < 1 or dim < 4 or key_dim < 1 or depth < 1 or groups < 1:
            raise ValueError("Positive dimensions/depth/groups and dim>=4 required")
        if len(nodes) != 2 or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in nodes) or nodes[0] <= nodes[1]:
            raise ValueError("coarse_nodes requires two decreasing positive integer counts")
        if isinstance(experts, bool) or not isinstance(experts, int) or experts < 1:
            raise ValueError("aggregation_experts must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("assignment_temperature must be finite and positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        # Keep >=2 channels/group even for one-voxel coarse feature maps.
        groups = min(groups, dim//2)
        self.channels = channels
        self.target_readout = options.get('design') == 'target_readout_v1'
        self.use_support = bool(options.get("completion_use_support", True))
        self.min_std = float(options.get("min_std", 1.0))
        if not math.isfinite(self.min_std) or self.min_std <= 0:
            raise ValueError("min_std must be positive and finite")
        self.stem = nn.Sequential(nn.Conv3d(channels+1, dim, 1), nn.GELU())
        self.fine_expert = nn.Sequential(*[ResidualSTBlock(dim, groups, dropout) for _ in range(depth)])
        self.fine_head = nn.Conv3d(dim, channels, 1)
        self.aggregation = nn.ModuleDict({name: ObservationAggregationMoE(
            dim, count, key_dim, experts, 'uniform' if self.target_readout else options.get("aggregation_mode", "learned"),
            bool(options.get("aggregation_geometry", True)), temperature,
            options.get('aggregation_top_k'),
        ) for name, count in zip(("mid", "coarse"), nodes)})
        self.scale_experts = nn.ModuleDict({name: ScaleCompletionExpert(dim, channels, groups, dropout, depth)
                                           for name in ("mid", "coarse")})
        self.completion_layout = options.get('completion_layout', 'scale')
        if self.completion_layout not in {'scale', 'routed_shared'}:
            raise ValueError('completion_layout must be scale or routed_shared')
        routed_shared = self.completion_layout == 'routed_shared'
        expert_type = options.get('completion_expert_type', 'mlp')
        expert_hidden = options.get('completion_expert_hidden', dim)
        if expert_type not in {'mlp', 'st_local', 'st_dilated'}:
            raise ValueError('completion_expert_type must be mlp, st_local or st_dilated')
        if type(expert_hidden) is not int or expert_hidden < 1:
            raise ValueError('completion_expert_hidden must be a positive integer')
        if not routed_shared and (expert_type != 'mlp' or expert_hidden != dim):
            raise ValueError('Custom completion experts require routed_shared layout')
        routed_count = options.get('completion_experts', 8 if routed_shared else 3)
        if type(routed_count) is not int or routed_count < 1:
            raise ValueError('completion_experts must be a positive integer')
        if routed_shared:
            if not self.target_readout or options.get('completion_mode') != 'topk':
                raise ValueError('routed_shared requires target_readout_v1 and topk completion')
            if options.get('completion_blend', 'none') != 'none':
                raise ValueError('Equal blending activates unselected experts; disable completion_blend for routed_shared')
        elif routed_count != 3:
            raise ValueError('The scale layout has exactly three heads; select routed_shared to change expert count')
        # Preserve initialization of existing frontend modules. The new layout
        # replaces this small reference router AFTER the frontend is initialized.
        self.completion_router = PointRouter(3*dim+7, 3, options.get("completion_mode", "learned"),
                                             3 if routed_shared else options.get('completion_top_k'))
        # Opt-in shrinkage only; historical configs/state dicts/RNG are unchanged.
        self.completion_blend = options.get('completion_blend', 'none')
        if self.completion_blend not in {'none', 'fixed', 'learned'}:
            raise ValueError('completion_blend must be none, fixed or learned')
        self.backend_diagnostics = options.get('backend_diagnostics', False)
        if type(self.backend_diagnostics) is not bool:
            raise ValueError('backend_diagnostics must be boolean')
        if self.completion_blend != 'none':
            if options.get('completion_mode') != 'topk' or options.get('completion_top_k') != 3:
                raise ValueError('Blended completion requires dense Top-3 of 3')
            alpha = options.get('completion_alpha', .5)
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
                raise ValueError('completion_alpha must be finite in [0,1]')
            if self.completion_blend == 'learned':
                if not 0 < alpha < 1:
                    raise ValueError('Learnable alpha initialization must be strictly inside (0,1)')
                self.completion_alpha_logit = nn.Parameter(torch.tensor(math.log(alpha/(1-alpha))))
            else:
                self.register_buffer('completion_alpha_fixed', torch.tensor(float(alpha)))
        if self.target_readout:
            # Extra parameters must not shift initialization of shared modules
            # in source/target and uniform/Top-K comparisons with the same seed.
            with torch.random.fork_rng(devices=[]):
                self.readout_routers = nn.ModuleDict({name: TargetReadoutRouter(
                    dim, key_dim, experts, options.get('aggregation_mode', 'topk'),
                    options.get('aggregation_top_k')) for name in ('mid', 'coarse')})
        if routed_shared:
            with torch.random.fork_rng(devices=[]):
                self.completion_router = PointRouter(3*dim+7, routed_count, 'topk', options.get('completion_top_k'))
                self.completion_shared = MultiScaleCompletionHead(3*dim+7, dim, channels)
                self.completion_experts = nn.ModuleList([
                    (MultiScaleCompletionHead(3*dim+7, expert_hidden, channels) if expert_type == 'mlp'
                     else ContextCompletionHead(3*dim+7, expert_hidden, channels,
                                                1 if expert_type == 'st_local' else 2))
                    for _ in range(routed_count)])

    @classmethod
    def from_config(cls, cfg):
        if cfg["model"].get("aux", {}).get("enabled", False):
            raise ValueError("dual_moe has no legacy auxiliary residual branch")
        options = cfg["model"].get("dual_moe", {})
        design = options.get("design", "learned_regions_v2")
        if design == "anchored_scale_moe":
            return AnchoredScaleMoEBackbone(cfg["model"]["c_in"], options)
        if design not in {"learned_regions_v2", "target_readout_v1"}:
            raise ValueError(f"Unknown dual_moe design: {design}")
        return cls(cfg["model"]["c_in"], options)

    def forward(self, x_f, m_f, x_m=None, m_m=None, x_c=None, m_c=None, r_m=None, r_c=None):
        # Extra arguments are accepted only for the historical training interface.
        # They are deliberately never read, even if they contain targets/NaNs.
        if x_f.ndim != 5 or x_f.shape[1] != self.channels or m_f.shape != (x_f.shape[0], 1, *x_f.shape[2:]):
            raise ValueError("Expected x_f [B,C,T,H,W] and binary m_f [B,1,T,H,W]")
        if not torch.all((m_f == 0) | (m_f == 1)):
            raise ValueError("m_f must be a finite binary observed mask (1=observed)")
        mask = m_f.float()
        observed = torch.where(mask.bool(), x_f.float(), torch.zeros_like(x_f, dtype=torch.float32))
        if not torch.isfinite(observed).all():
            raise ValueError("Observed inputs must be finite; only hidden values may be NaN")
        count = mask.sum((2, 3, 4), keepdim=True).clamp_min(1)
        center = (observed.sum((2, 3, 4), keepdim=True)/count).detach()
        variance = ((observed-center).square()*mask).sum((2, 3, 4), keepdim=True)/count
        scale = variance.clamp_min(self.min_std**2).sqrt().detach()
        z = (observed-center)/scale*mask
        features = self.stem(torch.cat((z, mask), 1))
        fine = self.fine_expert(features)
        normalized = {"fine": self.fine_head(fine).float()}
        hidden = {"fine": fine}
        aggregated, aligned_support = {}, []
        for name in ("mid", "coarse"):
            result = self.aggregation[name](fine, mask, z)
            if self.target_readout:
                normalized[name], hidden[name], support = self.scale_experts[name](result, fine, self.readout_routers[name])
            else:
                # Historical grid/anchored variants override the two-input
                # completion interface; do not pass a new argument to them.
                normalized[name], hidden[name], support = self.scale_experts[name](result, fine)
            normalized[name] = normalized[name].float()
            aggregated[name] = result
            aligned_support.append(support)
        support_input = torch.cat(aligned_support, 1)
        if not self.use_support:
            support_input = torch.zeros_like(support_input)
        completion_input = torch.cat((*hidden.values(), mask, support_input), 1)
        completion = self.completion_router(completion_input)
        alpha = completion.new_tensor(0. if self.completion_router.mode == 'uniform' else 1.)
        if self.completion_blend != 'none':
            alpha = (self.completion_alpha_logit.sigmoid() if self.completion_blend == 'learned'
                     else self.completion_alpha_fixed)
            completion = (1-alpha)/3 + alpha*completion
        extra = {}
        if self.completion_layout == 'routed_shared':
            # Shared base is always active; ONLY K selected routed corrections
            # contribute at each target. Experts are evaluated densely for now;
            # this is sparse mixing, not a compute/memory speedup claim.
            shared = self.completion_shared(completion_input)
            residuals = torch.stack([expert(completion_input) for expert in self.completion_experts], dim=1)
            correction = (completion.unsqueeze(2)*residuals).sum(1)
            pred_z = shared+correction
            extra = {'completion_layout': 'routed_shared',
                     'shared_prediction': shared*scale+center,
                     'completion_routed_predictions': {
                         f'e{i}': (shared+residuals[:, i])*scale+center for i in range(len(self.completion_experts))}}
        else:
            pred_z = sum(completion[:, i:i+1]*normalized[name] for i, name in enumerate(("fine", "mid", "coarse")))
        prediction = pred_z*scale+center
        return {
            "architecture": self.architecture,
            "region_protocol": "target_readout_v1" if self.target_readout else "learned_regions_v2",
            "aggregation_gate_domain": "missing" if self.target_readout else "observed",
            "partition_loss": torch.stack([v["partition_loss"] for v in aggregated.values()]).mean(),
            "region_assignments": {name: v["assignment"] for name, v in aggregated.items()},
            "x_hat_main": prediction,
            "h_st_aux": fine,
            "normalization": {"center": center, "scale": scale},
            "scale_predictions": {name: value*scale+center for name, value in normalized.items()},
            "aggregation_gates": {name: value["gates"] for name, value in aggregated.items()},
            "aggregation_support": {name: value["support"] for name, value in aggregated.items()},
            "aggregation_mass": {name: value["mass"] for name, value in aggregated.items()},
            "aggregation_effective_count": {name: value["effective_count"] for name, value in aggregated.items()},
            "completion_gates": completion,
            **extra,
            **({'backend_diagnostics': True, 'completion_alpha': alpha.detach()}
               if self.backend_diagnostics else {}),
            "routing_details": {
                **{f'aggregation_{name}': value['routing_info'] for name, value in aggregated.items() if value.get('routing_info') is not None},
                **({'completion': self.completion_router.routing_info} if self.completion_router.routing_info is not None else {}),
            },
            "gates": {},  # Not the legacy sample-wise top-k router protocol.
        }


class GridObservationAggregation(nn.Module):
    """Parameter-free, observed-normalized spatial bins; no dense N-by-K matrix.

    Partial edge bins divide by their actual observed count, not padded area.
    Coarse features are internal representations, not extra observed sensors.
    """
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, features, mask, normalized_values):
        b, d, t, h, w = features.shape
        s = self.stride
        hp, wp = (-h) % s, (-w) % s
        def sums(x):
            return F.avg_pool3d(F.pad(x.float(), (0, wp, 0, hp)), (1, s, s), stride=(1, s, s))*s*s
        mass_grid = sums(mask)
        area = sums(torch.ones_like(mask))
        pooled = sums(features.float()*mask)/mass_grid.clamp_min(1)
        hc, wc = pooled.shape[-2:]
        def nodes(x):
            return x.permute(0, 2, 3, 4, 1).reshape(b, 1, t, hc*wc, x.shape[1])
        with torch.no_grad():
            z = normalized_values.float()
            mean = sums(z*mask)/mass_grid.clamp_min(1)
            second = sums(z.square()*mask)/mass_grid.clamp_min(1)
            variance = (second-mean.square()).clamp_min(0).mean(1, keepdim=True)
            support = torch.cat((mass_grid/(mass_grid+area), mass_grid/(mass_grid+area),
                                 variance/(1+variance)), 1)
        iy = torch.arange(h, device=features.device)//s
        ix = torch.arange(w, device=features.device)//s
        indices = (iy[:, None]*wc+ix[None, :]).reshape(-1)
        mass = nodes(mass_grid)[..., 0]
        return {"features": nodes(pooled), "support": nodes(support), "mass": mass,
                "effective_count": mass, "gates": torch.ones_like(mask),
                "assignment": None, "fine_indices": indices,
                "prior": mask.new_ones((b, 1, t, 1)),
                "partition_loss": features.float().sum()*0}


def aligned_readout(aggregation, regions, detach_assignment=False):
    """Learned maps or exact bin membership, both aligned before expert mixing."""
    assignment = aggregation['assignment']
    if assignment is None:
        return regions.index_select(-2, aggregation['fine_indices'])
    return restore_regions(assignment.detach() if detach_assignment else assignment, regions)


class AlignedResidualScaleExpert(ScaleCompletionExpert):
    def forward(self, aggregation, fine):
        regions = self.blocks(self.input(torch.cat((aggregation['features'], aggregation['support']), -1)))
        prior = aggregation['prior'][..., None]
        aligned = (aligned_readout(aggregation, regions)*prior).sum(1)
        b, d, t, h, w = fine.shape
        aligned = aligned.reshape(b, t, h, w, d).permute(0, 4, 1, 2, 3).contiguous()
        support = (aligned_readout(aggregation, aggregation['support'], True)*prior.detach()).sum(1)
        support = support.reshape(b, t, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
        hidden = self.condition(torch.cat((aligned, fine), 1))
        return self.head(hidden), hidden, support


class AnchoredScaleMoEBackbone(DualMoEBackbone):
    """Single-stage scale MoE with a shared fine prediction and bounded corrections.

    Conservative default: E=3 learned aggregators, UNIFORM aggregation mixing.
    Optional regular bins test whether learned memberships are necessary. The
    original learned_regions_v2 class/parameter names remain loadable unchanged.
    """
    def __init__(self, channels, options):
        if options.get('aggregation_mode', 'uniform') != 'uniform':
            raise ValueError('anchored_scale_moe uses uniform aggregation, not a second learned router')
        kind = options.get('aggregation_kind', 'learned_regions')
        if kind not in ('learned_regions', 'regular_grid'):
            raise ValueError(f'Unknown aggregation_kind: {kind}')
        bound = float(options.get('residual_bound', 1.0))
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('residual_bound must be finite and positive')
        strides = options.get('grid_strides', [4, 8])
        if len(strides) != 2 or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in strides) or strides[0] >= strides[1]:
            raise ValueError('grid_strides must be two increasing positive integers')
        # Construct common components in the same order for both aggregation
        # choices, preserving common initial weights in paired comparisons.
        super().__init__(channels, {**options, 'aggregation_mode': 'uniform'})
        self.aggregation_kind, self.residual_bound = kind, bound
        if kind == 'regular_grid':
            self.aggregation = nn.ModuleDict({name: GridObservationAggregation(s)
                                              for name, s in zip(('mid', 'coarse'), strides)})
        d = self.fine_head.in_channels
        self.scale_experts = nn.ModuleDict({name: AlignedResidualScaleExpert(
            d, channels, int(options.get('num_groups', 4)), float(options.get('dropout', 0)),
            int(options.get('depth', 1))) for name in ('mid', 'coarse')})
        for expert in self.scale_experts.values():
            nn.init.zeros_(expert.head.weight)
            nn.init.zeros_(expert.head.bias)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        anchor = out['scale_predictions']['fine']
        center, scale = out['normalization']['center'], out['normalization']['scale']
        candidates = {'fine': anchor}
        residuals = {}
        for name in ('mid', 'coarse'):
            raw = (out['scale_predictions'][name]-center)/scale
            residuals[name] = self.residual_bound*torch.tanh(raw)
            candidates[name] = anchor+scale*residuals[name]
        gates = out['completion_gates']
        correction = sum(gates[:, i:i+1]*residuals[name] for i, name in ((1, 'mid'), (2, 'coarse')))
        out.update(x_hat_main=anchor+scale*correction,
                   scale_predictions=candidates, base_prediction=anchor,
                   normalized_residuals=residuals, normalized_correction=correction,
                   residual_bound=self.residual_bound,
                   region_protocol=f'anchored_scale_moe_v1:{self.aggregation_kind}')
        return out
