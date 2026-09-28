"""Single-scale, state-conditioned temporal/spatial Chain of Experts.

All tensors use ``[B, C, T, H, W]``.  The original observation mask stays
fixed throughout the chain; a decoded estimate never becomes an observation.
Hard routing dispatches only selected sample/expert pairs in training and evaluation.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .coe_pattern_experts import (
    DilatedDirectionalExpert,
    JointSpatioTemporalExpert,
    TemporalAttentionExpert,
)


from .coe_router import GroupedRouter, PairRouter, PartnerScorer, PreviousExpertRouter, observed_pattern_features


SUPPORTED_EXPERT_NAMES = ("T", "S", "TD", "SD", "TA", "ST", "TL", "SL")


def _expert_kind(name: str) -> str:
    """Map an instance label (T1, S2, T_A, ...) to its operator kind."""
    upper = str(name).upper().replace("_", "")
    for kind in ("TD", "SD", "TA", "ST", "TL", "SL", "T", "S"):
        if upper == kind or upper.startswith(kind) and upper[len(kind):].isdigit():
            return kind
    return upper


SUPPORT_FEATURE_NAMES = (
    "temporal_coverage",
    "spatial_coverage",
    "gap_length",
    "previous_distance",
    "next_distance",
    "no_temporal_observation",
    "no_previous_observation",
    "no_next_observation",
    "empty_spatial_neighborhood",
    "no_spatial_observation",
)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _kernel_size(name: str, value: int) -> int:
    value = _positive_int(name, value)
    if value % 2 != 1:
        raise ValueError(f"{name} must be odd, got {value}")
    return value


def compute_observation_support(
    mask: torch.Tensor, temporal_kernel: int = 3, spatial_kernel: int = 3
) -> torch.Tensor:
    """Compute offline support solely from a binary, per-variable input mask.

    The result has ``10 * C`` channels in ``SUPPORT_FEATURE_NAMES`` order,
    with C consecutive channels per feature.  Local temporal coverage includes
    the current time; spatial coverage excludes the center cell.  Both divide
    by the actual number of in-window positions, including at boundaries.
    Missing run lengths and visible-observation distances are divided by T.
    Unavailable observations have clipped distance 1 and an explicit flag.
    """
    temporal_kernel = _kernel_size("temporal_kernel", temporal_kernel)
    spatial_kernel = _kernel_size("spatial_kernel", spatial_kernel)
    if mask.ndim != 5 or any(size < 1 for size in mask.shape):
        raise ValueError("mask must have nonempty shape [B, C, T, H, W]")
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError("mask must contain only finite binary values")
    # FP32 support arithmetic avoids reduced-precision distance rounding under
    # mixed precision.  Each observed variable has its own source statistics.
    visible = mask.bool()
    b, c, t, h, w = mask.shape
    with torch.autocast(device_type=mask.device.type, enabled=False):
        values = visible.to(dtype=torch.float32)
        flat = values.reshape(b * c, 1, t, h, w)
        ones = torch.ones((1, 1, t, h, w), device=mask.device)
        time_filter = torch.ones((1, 1, temporal_kernel, 1, 1), device=mask.device)
        time_padding = (temporal_kernel // 2, 0, 0)
        time_count = F.conv3d(flat, time_filter, padding=time_padding)
        time_denominator = F.conv3d(ones, time_filter, padding=time_padding)
        time_coverage = (time_count / time_denominator).reshape(b, c, t, h, w)

        space_filter = torch.ones(
            (1, 1, 1, spatial_kernel, spatial_kernel), device=mask.device
        )
        space_filter[..., spatial_kernel // 2, spatial_kernel // 2] = 0
        space_padding = (0, spatial_kernel // 2, spatial_kernel // 2)
        space_count = F.conv3d(flat, space_filter, padding=space_padding)
        space_denominator = F.conv3d(ones, space_filter, padding=space_padding)
        space_coverage = (space_count / space_denominator.clamp_min(1)).reshape(
            b, c, t, h, w
        )
        empty_neighbors = (space_denominator == 0).expand(b * c, 1, t, h, w)
        no_space_observation = (space_count == 0) & ~empty_neighbors

        index = torch.arange(t, device=mask.device).reshape(1, 1, t, 1, 1)
        previous = torch.where(visible, index, -1).cummax(dim=2).values
        next_visible = torch.where(visible, index, t).flip(2).cummin(dim=2).values.flip(2)
        no_previous = previous < 0
        no_next = next_visible >= t
        previous_distance = torch.where(no_previous, t, index - previous).float() / t
        next_distance = torch.where(no_next, t, next_visible - index).float() / t
        gap_length = torch.where(visible, 0, next_visible - previous - 1).float() / t
        no_time_observation = ~visible.any(dim=2, keepdim=True)
        result = torch.cat(
            [
                time_coverage,
                space_coverage,
                gap_length,
                previous_distance,
                next_distance,
                no_time_observation.expand_as(visible).float(),
                no_previous.float(),
                no_next.float(),
                empty_neighbors.reshape(b, c, t, h, w).float(),
                no_space_observation.reshape(b, c, t, h, w).float(),
            ],
            dim=1,
        )
    return result


class PointwiseLayerNorm(nn.Module):
    """Normalize channels at each grid/time point, without mixing positions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class PointwiseExpert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        hidden_dim = max(4, dim // 2) if hidden_dim is None else hidden_dim
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.network = nn.Sequential(
            PointwiseLayerNorm(dim),
            nn.Conv3d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DirectionalExpert(nn.Module):
    """A nonlinear local operator mixing only one declared direction."""

    def __init__(self, dim: int, direction: str, kernel_size: int = 3) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        kernel_size = _kernel_size("kernel_size", kernel_size)
        if direction not in {"temporal", "spatial"}:
            raise ValueError("direction must be 'temporal' or 'spatial'")
        self.direction = direction
        kernel = (kernel_size, 1, 1) if direction == "temporal" else (1, kernel_size, kernel_size)
        hidden_dim = dim * 2
        self.network = nn.Sequential(
            PointwiseLayerNorm(dim),
            nn.Conv3d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel,
                padding=tuple(size // 2 for size in kernel),
                groups=hidden_dim,
            ),
            PointwiseLayerNorm(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalSpatialCoE(nn.Module):
    """Configurable pattern expert pool with independent, per-step routers.

    Defaults preserve the two-expert model and its checkpoint parameter names.
    ``fixed_path`` uses configured expert labels and is active in fixed mode.
    Each expert is shared across all steps. Fixed and shared-only variants
    bypass their unused routing heads. ``expert_state='initial'`` freezes the
    expert input while residual states and router inputs can still evolve.
    ``parallel`` is a single-round, learned soft mixture of initial-state experts.
    Hard routing supports ``top_k`` simultaneous experts; their clean routing
    probabilities are renormalized over the selected experts before fusion.
    """

    requires_multiscale = False

    def __init__(
        self,
        c_in: int,
        dim: int = 64,
        num_steps: int = 2,
        routing_mode: str = "hard",
        fixed_path: Sequence[str] | None = None,
        fixed_expert_steps: Sequence[str | None] | None = None,
        router_state: str = "dynamic",
        use_shared: bool = True,
        use_routed: bool = True,
        temperature: float = 1.0,
        temporal_kernel: int = 3,
        spatial_kernel: int = 3,
        residual_init: float = 0.1,
        router_hidden_dim: int | None = None,
        expert_pool: Sequence[str] | None = None,
        temporal_dilation: int = 2,
        spatial_dilation: int = 2,
        attention_heads: int = 4,
        expert_state: str = "dynamic",
        router_features: str = "legacy",
        value_mean: Sequence[float] | None = None,
        value_std: Sequence[float] | None = None,
        router_fp32: bool = False,
        router_init_std: float | None = None,
        routing_warmup_epochs: int = 0,
        routing_transition_epochs: int = 0,
        sampling_temperature_start: float = 2.0,
        uniform_mix_start: float = 0.5,
        previous_expert_context: bool = False,
        global_route_weights: bool = False,
        router_input_noise_std: float = 0.0,
        router_input_noise_steps: int = 0,
        top_k: int = 1,
        pair_mode: str = "native",
        acceptance: str = "none",
        expert_sharing: str = "shared",
        completion_feedback: bool = True,
        partner_fusion: str = "individual",
        partner_aux_head_only: bool = False,
        fusion_mode: str = "original",
        pair_dense_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.c_in = _positive_int("c_in", c_in)
        self.dim = _positive_int("dim", dim)
        self.num_steps = _positive_int("num_steps", num_steps)
        if expert_pool is None:
            expert_pool = ("T", "S")
        if isinstance(expert_pool, str) or not isinstance(expert_pool, Sequence) or not expert_pool:
            raise ValueError("expert_pool must be a nonempty sequence of expert labels")
        self.expert_names = tuple(str(name).upper() for name in expert_pool)
        if len(set(self.expert_names)) != len(self.expert_names):
            raise ValueError("expert_pool must not contain duplicate expert labels")
        unknown = {name for name in self.expert_names if _expert_kind(name) not in SUPPORTED_EXPERT_NAMES}
        if unknown:
            raise ValueError(f"Unknown expert labels: {sorted(unknown)}")
        self.num_experts = len(self.expert_names)
        if type(top_k) is not int or not 1 <= top_k <= self.num_experts:
            raise ValueError("top_k must be an integer in [1, len(expert_pool)]")
        if routing_mode in {"fixed", "parallel"} and top_k != 1:
            raise ValueError("top_k > 1 is supported only for hard or soft routing")
        self.top_k = top_k
        if fusion_mode not in {"original", "context", "response", "local_response", "proposal"}:
            raise ValueError("fusion_mode must be original, context, response, local_response or proposal")
        if fusion_mode != "original" and (routing_mode != "hard" or top_k != 2 or pair_mode != "native"):
            raise ValueError("Conditional fusion requires native hard Top-2 routing")
        if type(pair_dense_warmup_steps) is not int or not 0 <= pair_dense_warmup_steps <= num_steps:
            raise ValueError("pair_dense_warmup_steps must be in [0, num_steps]")
        if pair_dense_warmup_steps and (pair_mode != "interaction" or routing_mode != "hard" or
                                        top_k != 2 or routing_warmup_epochs < 1):
            raise ValueError("Dense pair warmup requires interaction hard Top-2 and warmup epochs")
        self.fusion_mode = fusion_mode
        self.pair_dense_warmup_steps = pair_dense_warmup_steps
        if pair_mode not in {"native", "additive", "interaction", "partner", "partner_residual"}:
            raise ValueError("pair_mode must be native, additive, interaction, partner or partner_residual")
        if acceptance not in {"none", "point", "window"}:
            raise ValueError("acceptance must be none, point or window")
        if expert_sharing not in {"shared", "per_step"}:
            raise ValueError("expert_sharing must be shared or per_step")
        if pair_mode != "native" and (routing_mode != "hard" or top_k != 2 or
                router_features != "legacy" or previous_expert_context or global_route_weights):
            raise ValueError("Pair routing requires legacy hard Top-2 without previous/global routing")
        if partner_fusion not in {"individual", "corrected"}:
            raise ValueError("partner_fusion must be individual or corrected")
        if partner_fusion == "corrected" and pair_mode != "partner_residual":
            raise ValueError("Corrected partner fusion requires residual partner routing")
        if type(partner_aux_head_only) is not bool or (partner_aux_head_only and pair_mode != "partner_residual"):
            raise ValueError("partner_aux_head_only requires residual partner routing")
        self.partner_fusion = partner_fusion
        self.partner_aux_head_only = partner_aux_head_only
        if type(completion_feedback) is not bool:
            raise ValueError("completion_feedback must be boolean")
        if not completion_feedback and acceptance != "none":
            raise ValueError("Completion-free layered MoE does not use an acceptance gate")
        self.pair_mode, self.acceptance, self.expert_sharing = pair_mode, acceptance, expert_sharing
        self.completion_feedback = completion_feedback
        pairs = (torch.combinations(torch.arange(self.num_experts), r=2) if top_k == 2
                 else torch.empty((0, 2), dtype=torch.long))
        self.register_buffer("pair_indices", pairs, persistent=False)
        temporal_dilation = _positive_int("temporal_dilation", temporal_dilation)
        spatial_dilation = _positive_int("spatial_dilation", spatial_dilation)
        attention_heads = _positive_int("attention_heads", attention_heads)
        if routing_mode not in {"hard", "soft", "fixed", "parallel"}:
            raise ValueError("routing_mode must be 'hard', 'soft', 'fixed', or 'parallel'")
        if routing_mode == "parallel" and self.num_steps != 1:
            raise ValueError("parallel routing requires num_steps=1")
        if router_state not in {"dynamic", "initial"}:
            raise ValueError("router_state must be 'dynamic' or 'initial'")
        if expert_state not in {"dynamic", "initial"}:
            raise ValueError("expert_state must be 'dynamic' or 'initial'")
        if not isinstance(use_shared, bool) or not isinstance(use_routed, bool):
            raise ValueError("use_shared and use_routed must be booleans")
        if not use_shared and not use_routed:
            raise ValueError("At least one of use_shared/use_routed must be enabled")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(float(residual_init)) or not 0 < float(residual_init) < 1:
            raise ValueError("residual_init must be finite and strictly between 0 and 1")
        self.routing_mode = routing_mode
        self.router_state = router_state
        self.expert_state = expert_state
        self.use_shared = use_shared
        self.use_routed = use_routed
        self.temperature = float(temperature)
        if router_features not in {"legacy", "grouped"}:
            raise ValueError("router_features must be legacy or grouped")
        for value in (routing_warmup_epochs, routing_transition_epochs):
            if type(value) is not int or value < 0:
                raise ValueError("routing schedule epochs must be nonnegative integers")
        if not math.isfinite(sampling_temperature_start) or sampling_temperature_start < 1:
            raise ValueError("sampling_temperature_start must be finite and >= 1")
        if not 0 <= uniform_mix_start <= 1:
            raise ValueError("uniform_mix_start must be in [0,1]")
        if not math.isfinite(router_input_noise_std) or router_input_noise_std < 0:
            raise ValueError("router_input_noise_std must be finite and nonnegative")
        if type(router_input_noise_steps) is not int or router_input_noise_steps < 0:
            raise ValueError("router_input_noise_steps must be a nonnegative integer")
        if router_init_std is not None and (not math.isfinite(router_init_std) or router_init_std <= 0):
            raise ValueError("router_init_std must be positive and finite")
        if routing_warmup_epochs + routing_transition_epochs and routing_mode != "hard":
            raise ValueError("routing schedule is supported only for hard routing")
        self.router_features = router_features
        if type(previous_expert_context) is not bool:
            raise ValueError('previous_expert_context must be boolean')
        if type(global_route_weights) is not bool:
            raise ValueError('global_route_weights must be boolean')
        if global_route_weights and routing_mode != "soft":
            raise ValueError('global_route_weights requires soft routing_mode')
        if previous_expert_context and (routing_mode != 'hard' or not use_routed or
                router_features != 'legacy' or routing_warmup_epochs + routing_transition_epochs):
            raise ValueError('Previous expert context requires legacy hard routing without soft warmup')
        self.previous_expert_context = previous_expert_context
        self.global_route_weights = global_route_weights
        self.router_input_noise_std = float(router_input_noise_std)
        self.router_input_noise_steps = min(int(router_input_noise_steps), self.num_steps)
        self.router_fp32 = bool(router_fp32)
        self.routing_warmup_epochs = routing_warmup_epochs
        self.routing_transition_epochs = routing_transition_epochs
        self.sampling_temperature_start = float(sampling_temperature_start)
        self.uniform_mix_start = float(uniform_mix_start)
        self.routing_epoch = 1
        if router_features == "grouped":
            mean = torch.as_tensor(value_mean if value_mean is not None else [0.] * c_in, dtype=torch.float32)
            std = torch.as_tensor(value_std if value_std is not None else [1.] * c_in, dtype=torch.float32)
            if mean.shape != (c_in,) or std.shape != (c_in,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError("value_mean/std must be finite per-channel vectors with positive std")
            self.register_buffer("router_value_mean", mean.reshape(1, c_in, 1, 1, 1))
            self.register_buffer("router_value_std", std.reshape(1, c_in, 1, 1, 1))
        self.temporal_kernel = _kernel_size("temporal_kernel", temporal_kernel)
        self.spatial_kernel = _kernel_size("spatial_kernel", spatial_kernel)
        if fixed_path is None:
            fixed_path = [self.expert_names[step % self.num_experts] for step in range(num_steps)]
        if isinstance(fixed_path, str) or not isinstance(fixed_path, Sequence):
            raise ValueError("fixed_path must be a sequence of expert labels")
        self.fixed_path = tuple(str(label).upper() for label in fixed_path)
        # A base config may carry a two-step fixed_path while a dynamic-depth
        # override changes K. Unused fixed paths must not constrain that model.
        if routing_mode == "fixed" and (
            len(self.fixed_path) != num_steps
            or any(label not in self.expert_names for label in self.fixed_path)
        ):
            raise ValueError("fixed_path must contain num_steps labels from expert_pool")
        if fixed_expert_steps is None:
            fixed_expert_steps = [None] * num_steps
        if isinstance(fixed_expert_steps, (str, bytes)) or not isinstance(fixed_expert_steps, Sequence):
            raise ValueError("fixed_expert_steps must be a sequence of labels or nulls")
        if len(fixed_expert_steps) != num_steps:
            raise ValueError("fixed_expert_steps must have num_steps entries")
        normalized_fixed_steps = []
        for label in fixed_expert_steps:
            if label is None:
                normalized_fixed_steps.append(None)
            else:
                normalized = str(label).upper()
                if normalized not in self.expert_names:
                    raise ValueError("fixed_expert_steps labels must belong to expert_pool")
                normalized_fixed_steps.append(normalized)
        if routing_mode == "fixed" and any(label is not None for label in normalized_fixed_steps):
            raise ValueError("fixed_expert_steps cannot be combined with fixed routing_mode")
        self.fixed_expert_steps = tuple(normalized_fixed_steps)

        support_dim = len(SUPPORT_FEATURE_NAMES) * c_in
        self.position_projection = nn.Conv3d(3, dim, 1, bias=False)
        self.encoder = nn.Sequential(
            nn.Conv3d(2 * c_in + support_dim + dim, dim, 1),
            PointwiseLayerNorm(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, 1),
        )
        # One projection builds the same H/V/M/s/P interface for every expert.
        self.state_projection = nn.Conv3d(2 * dim + 2 * c_in + support_dim, dim, 1)
        self.state_norm = PointwiseLayerNorm(dim)
        # Every configured label receives its own module. Suffixes such as T1
        # and T2 deliberately create independent parameter instances.
        def make_pool() -> nn.ModuleDict:
            pool = nn.ModuleDict()
            for name in self.expert_names:
                kind = _expert_kind(name)
                if kind == "T":
                    pool[name] = DirectionalExpert(dim, "temporal", self.temporal_kernel)
                elif kind == "S":
                    pool[name] = DirectionalExpert(dim, "spatial", self.spatial_kernel)
                elif kind == "TD":
                    pool[name] = DilatedDirectionalExpert(dim, "temporal", self.temporal_kernel, temporal_dilation)
                elif kind == "SD":
                    pool[name] = DilatedDirectionalExpert(dim, "spatial", self.spatial_kernel, spatial_dilation)
                elif kind == "TA":
                    pool[name] = TemporalAttentionExpert(dim, attention_heads)
                elif kind == "ST":
                    pool[name] = JointSpatioTemporalExpert(dim)
                elif kind == "TL":
                    pool[name] = DirectionalExpert(dim, "temporal", self.temporal_kernel + 2)
                elif kind == "SL":
                    pool[name] = DirectionalExpert(dim, "spatial", self.spatial_kernel + 2)
            return pool

        # Keep the first pool and all common modules initialized exactly as in
        # E1; append independent later-round pools only after common modules.
        self.pattern_experts = make_pool()
        self.step_pattern_experts = nn.ModuleList()
        self.shared_expert = PointwiseExpert(dim)
        self.decoder = nn.Sequential(
            PointwiseLayerNorm(dim),
            nn.Conv3d(dim, max(4, dim // 2), 1),
            nn.GELU(),
            nn.Conv3d(max(4, dim // 2), c_in, 1),
        )
        # H/V all+missing means; s all mean/std/max+missing mean;
        # change all+missing means; missing fraction and empty-missing flag.
        router_input_dim = 2 * dim + 4 * c_in + 4 * support_dim + 2
        if router_hidden_dim is None:
            router_hidden_dim = max(16, dim)
        router_hidden_dim = _positive_int("router_hidden_dim", router_hidden_dim)
        self.routers = nn.ModuleList([
            (PairRouter(router_input_dim, router_hidden_dim, self.num_experts,
                        len(self.pair_indices), False)
             if pair_mode != "native" else
             nn.Sequential(nn.LayerNorm(router_input_dim), nn.Linear(router_input_dim, router_hidden_dim),
                           nn.GELU(), nn.Linear(router_hidden_dim, self.num_experts)))
            for _ in range(num_steps)
        ])
        if router_features == "grouped":
            self.routers = nn.ModuleList([
                GroupedRouter([2 * dim, 2 * c_in, 4 * support_dim, 2 * c_in, 2, 9 * c_in],
                              router_hidden_dim, self.num_experts)
                for _ in range(num_steps)
            ])
        if router_init_std is not None:
            for router in self.routers:
                final = (router.head[-1] if isinstance(router, GroupedRouter) else
                         router.layers[-1] if isinstance(router, PairRouter) else router[-1])
                nn.init.normal_(final.weight, std=router_init_std)
                nn.init.zeros_(final.bias)
        residual_logit = math.log(float(residual_init) / (1 - float(residual_init)))
        self.shared_scale_logits = nn.Parameter(torch.full((num_steps,), residual_logit))
        self.routed_scale_logits = nn.Parameter(torch.full((num_steps,), residual_logit))
        if self.global_route_weights:
            self.global_route_logits = nn.Parameter(torch.zeros(num_steps, self.num_experts))
        self.acceptance_head = None
        if acceptance != "none":
            head_dim = max(8, dim // 4)
            self.acceptance_head = nn.Sequential(
                nn.Conv3d(dim + 14 * c_in, head_dim, 1), nn.GELU(),
                nn.Conv3d(head_dim, c_in, 1),
            )
            nn.init.zeros_(self.acceptance_head[-1].weight)
            nn.init.constant_(self.acceptance_head[-1].bias, math.log(9.0))
        if self.previous_expert_context:
            for step in range(1, num_steps):
                self.routers[step] = PreviousExpertRouter(self.routers[step], self.num_experts)
        if expert_sharing == "per_step":
            self.step_pattern_experts.extend(make_pool() for _ in range(num_steps - 1))
        if pair_mode == "interaction":
            for router in self.routers:
                router.enable_interaction()
        self.partner_scorer = (
            PartnerScorer(router_input_dim + 2 * dim + 2 * c_in, router_hidden_dim,
                          self.num_experts)
            if pair_mode in {"partner", "partner_residual"} else None
        )
        self.fusion_identity = None
        self.fusion_gate = None
        if fusion_mode in {"context", "response"}:
            self.fusion_identity = nn.Embedding(self.num_experts, 8)
            feature_dim = router_input_dim + 2 + 16 + 5 * dim
            self.fusion_gate = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, router_hidden_dim),
                nn.GELU(), nn.Linear(router_hidden_dim, 1),
            )
        elif fusion_mode == "local_response":
            self.fusion_gate = nn.Sequential(
                nn.Conv3d(3 * dim + c_in + support_dim, max(8, dim // 4),
                          kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GELU(), nn.Conv3d(max(8, dim // 4), 1, 1),
            )
        elif fusion_mode == "proposal":
            self.fusion_gate = nn.Sequential(
                nn.Conv3d(3 * dim + 6 * c_in + support_dim, max(8, dim // 4),
                          kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GELU(), nn.Conv3d(max(8, dim // 4), 1, 1),
            )
        if self.fusion_gate is not None:
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.zeros_(self.fusion_gate[-1].bias)

    @classmethod
    def from_config(cls, cfg: dict) -> "TemporalSpatialCoE":
        model = cfg["model"]
        coe = model.get("coe", {})
        main = model.get("main", {})
        return cls(
            c_in=model["c_in"],
            dim=coe.get("dim", model.get("dim", main.get("dim", 64))),
            num_steps=coe.get("num_steps", 2),
            routing_mode=coe.get("routing_mode", "hard"),
            fixed_path=coe.get("fixed_path"),
            fixed_expert_steps=coe.get("fixed_expert_steps"),
            router_state=coe.get("router_state", "dynamic"),
            use_shared=coe.get("use_shared", True),
            use_routed=coe.get("use_routed", True),
            temperature=coe.get("temperature", 1.0),
            temporal_kernel=coe.get("temporal_kernel", 3),
            spatial_kernel=coe.get("spatial_kernel", 3),
            residual_init=coe.get("residual_init", 0.1),
            router_hidden_dim=coe.get("router_hidden_dim"),
            expert_pool=coe.get("expert_pool"),
            temporal_dilation=coe.get("temporal_dilation", 2),
            spatial_dilation=coe.get("spatial_dilation", 2),
            attention_heads=coe.get("attention_heads", 4),
            expert_state=coe.get("expert_state", "dynamic"),
            **{name: coe[name] for name in ("router_features", "value_mean", "value_std",
                "router_fp32", "router_init_std", "routing_warmup_epochs",
                "routing_transition_epochs", "sampling_temperature_start", "uniform_mix_start",
                "previous_expert_context") if name in coe},
            global_route_weights=coe.get("global_route_weights", False),
            router_input_noise_std=coe.get("router_input_noise_std", 0.0),
            router_input_noise_steps=coe.get("router_input_noise_steps", 0),
            top_k=coe.get("top_k", 1),
            pair_mode=coe.get("pair_mode", "native"),
            acceptance=coe.get("acceptance", "none"),
            expert_sharing=coe.get("expert_sharing", "shared"),
            completion_feedback=coe.get("completion_feedback", True),
            partner_fusion=coe.get("partner_fusion", "individual"),
            partner_aux_head_only=coe.get("partner_aux_head_only", False),
            fusion_mode=coe.get("fusion_mode", "original"),
            pair_dense_warmup_steps=coe.get("pair_dense_warmup_steps", 0),
        )

    def set_routing_epoch(self, epoch: int) -> None:
        self.routing_epoch = int(epoch)

    def routing_schedule(self) -> tuple[float, float, float]:
        """Hard fraction, uniform floor mixture, actual sampling temperature."""
        if not self.training or not (self.routing_warmup_epochs + self.routing_transition_epochs):
            return 1., 0., 1.
        if self.routing_epoch <= self.routing_warmup_epochs:
            hard = 0.
        else:
            hard = min(1., (self.routing_epoch - self.routing_warmup_epochs) /
                       max(1, self.routing_transition_epochs))
        return hard, (1 - hard) * self.uniform_mix_start, 1 + (1 - hard) * (self.sampling_temperature_start - 1)

    def routed_experts(self, step: int = 0) -> tuple[nn.Module, ...]:
        """Return the operators for one round in router-column order."""
        pool = (self.pattern_experts if self.expert_sharing == "shared" or step == 0
                else self.step_pattern_experts[step - 1])
        return tuple(pool[name] for name in self.expert_names)

    def _position(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = x.shape
        axes = [
            torch.linspace(-1, 1, size, device=x.device, dtype=x.dtype)
            if size > 1
            else torch.zeros(1, device=x.device, dtype=x.dtype)
            for size in (t, h, w)
        ]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        return self.position_projection(coordinates.unsqueeze(0)).expand(b, -1, -1, -1, -1)

    @staticmethod
    def _missing_pool(x: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        denominator = missing.sum(dim=(2, 3, 4))
        # Point-level masks can be fractional for multivariate observations;
        # clamping their denominator to 1 would dilute a small missing region.
        weighted = (x * missing).sum(dim=(2, 3, 4)) / denominator.clamp_min(
            torch.finfo(denominator.dtype).tiny
        )
        return torch.where(denominator > 0, weighted, x.mean(dim=(2, 3, 4)))

    def _router_features(
        self,
        hidden: torch.Tensor,
        values: torch.Tensor,
        support_summary: torch.Tensor,
        change: torch.Tensor,
        missing: torch.Tensor,
        pattern_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.router_features == "grouped":
            values = (values - self.router_value_mean) / self.router_value_std
            change = change / self.router_value_std
        # H describes a point shared by all variables; weight it by the fraction
        # of missing variables. V/change retain per-variable missing weights.
        point_missing = missing.mean(dim=1, keepdim=True)
        missing_fraction = missing.mean(dim=(1, 2, 3, 4)).unsqueeze(1)
        return torch.cat(
            [
                hidden.mean(dim=(2, 3, 4)),
                self._missing_pool(hidden, point_missing),
                values.mean(dim=(2, 3, 4)),
                self._missing_pool(values, missing),
                support_summary,
                change.mean(dim=(2, 3, 4)),
                self._missing_pool(change, missing),
                missing_fraction,
                (missing_fraction == 0).to(hidden.dtype),
            ] + ([pattern_summary] if self.router_features == "grouped" else []),
            dim=1,
        )

    def _add_router_input_noise(self, features: torch.Tensor, step: int) -> torch.Tensor:
        """Add relative Gaussian noise to selected early routers in training only."""
        if (not self.training or self.router_input_noise_std <= 0 or
                step >= self.router_input_noise_steps):
            return features
        base = features.float()
        scale = base.detach().std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
        noisy = base + torch.randn_like(base) * (self.router_input_noise_std * scale)
        return noisy.to(dtype=features.dtype)

    def _pair_route(self, logits: torch.Tensor, pair_bias: torch.Tensor | None,
                    sampling_temperature: float, dense_fraction: float = 0.0,
                    uniform_mix: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic pair forward with a differentiable soft pair surrogate."""
        individual = logits.float()
        pair_logits = individual[:, self.pair_indices[:, 0]] + individual[:, self.pair_indices[:, 1]]
        if pair_bias is not None:
            pair_logits = pair_logits + pair_bias.float()
        pair_probs = F.softmax(pair_logits / sampling_temperature, dim=-1)
        pair_ids = pair_logits.argmax(dim=-1)
        # Match torch.topk's native tie rule when the additive score has
        # multiple maxima (including the zero-initialized interaction head).
        native_top2 = individual.topk(2, dim=-1).indices.sort(dim=-1).values
        native_ids = (self.pair_indices.unsqueeze(0) == native_top2.unsqueeze(1)).all(dim=-1).long().argmax(dim=-1)
        native_scores = pair_logits.gather(1, native_ids.unsqueeze(1)).squeeze(1)
        pair_ids = torch.where(native_scores == pair_logits.max(dim=-1).values, native_ids, pair_ids)
        hard = F.one_hot(pair_ids, len(self.pair_indices)).to(pair_probs.dtype)
        pair_choice = (pair_probs - pair_probs.detach()) + hard if self.training else hard
        pair_individual = individual[:, self.pair_indices]
        within_pair = F.softmax(pair_individual / sampling_temperature, dim=-1)
        indices = self.pair_indices.unsqueeze(0).expand(individual.shape[0], -1, -1)
        pair_to_expert = torch.zeros(
            (individual.shape[0], len(self.pair_indices), self.num_experts),
            device=individual.device, dtype=within_pair.dtype,
        ).scatter(-1, indices, within_pair)
        weights = torch.einsum("bp,bpe->be", pair_choice, pair_to_expert)
        if dense_fraction > 0:
            # The first configured rounds execute all experts during early epochs.
            # Their dense contribution fades to the ordinary hard pair route.
            dense_weights = torch.einsum("bp,bpe->be", pair_probs, pair_to_expert)
            dense_weights = uniform_mix / self.num_experts + (1 - uniform_mix) * dense_weights
            weights = (1 - dense_fraction) * weights + dense_fraction * dense_weights
        return weights, pair_ids, pair_logits, pair_probs

    def _pair_importance(self, logits: torch.Tensor, pair_probs: torch.Tensor,
                         temperature: float, partner_scores: torch.Tensor | None = None,
                         primary: torch.Tensor | None = None) -> torch.Tensor:
        """Candidate marginal using each pair's actual within-pair fusion logits."""
        pair_logits = logits.float()[:, self.pair_indices]
        if partner_scores is not None:
            if primary is None:
                raise ValueError("Corrected pair importance requires the primary expert")
            left, right = self.pair_indices.unbind(-1)
            pair_logits = torch.stack((
                torch.where(primary[:, None] == right[None], partner_scores[:, left], pair_logits[..., 0]),
                torch.where(primary[:, None] == left[None], partner_scores[:, right], pair_logits[..., 1]),
            ), dim=-1)
        within = F.softmax(pair_logits / temperature, dim=-1)
        indices = self.pair_indices.unsqueeze(0).expand(logits.shape[0], -1, -1)
        contributions = torch.zeros(
            logits.shape[0], len(self.pair_indices), self.num_experts,
            device=logits.device, dtype=within.dtype,
        ).scatter(-1, indices, within)
        return torch.einsum("bp,bpe->be", pair_probs.float(), contributions)

    def _dispatch(self, unified: torch.Tensor, paths: torch.Tensor, step: int = 0) -> torch.Tensor:
        """Evaluate only selected whole windows, keeping their full context."""
        result = torch.zeros_like(unified)
        for expert_index, expert in enumerate(self.routed_experts(step)):
            selected = torch.nonzero(paths == expert_index, as_tuple=False).flatten()
            if selected.numel():
                updates = expert(unified.index_select(0, selected))
                # Autocast can return FP16/BF16 expert outputs while the
                # normalized state/accumulator is FP32. index_copy requires
                # matching dtypes; the differentiable cast preserves gradients
                # and accumulates every expert in the state's precision.
                result = result.index_copy(0, selected, updates.to(result.dtype))
        return result

    def _dispatch_weighted(self, unified: torch.Tensor, weights: torch.Tensor,
                           step: int = 0) -> torch.Tensor:
        """Evaluate an expert only on windows that selected it."""
        if weights.shape != (unified.shape[0], self.num_experts):
            raise ValueError("Sparse routing weights must have shape [batch, experts]")
        result = torch.zeros_like(unified)
        for expert_index, expert in enumerate(self.routed_experts(step)):
            selected = torch.nonzero(weights[:, expert_index].detach() != 0,
                                     as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            inputs = unified.index_select(0, selected)
            updates = expert(inputs)
            coefficients = weights.index_select(0, selected)[:, expert_index]
            weighted = updates * coefficients[:, None, None, None, None].to(updates.dtype)
            result = result.index_add(0, selected, weighted.to(result.dtype))
        return result

    def _dispatch_pair_updates(self, unified: torch.Tensor, selected: torch.Tensor,
                               step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run each selected expert once per window and return separate updates."""
        batch = unified.shape[0]
        flat_ids = selected.reshape(-1)
        result = unified.new_zeros((batch * 2, *unified.shape[1:]))
        for expert_index, expert in enumerate(self.routed_experts(step)):
            positions = torch.nonzero(flat_ids == expert_index, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            updates = expert(unified.index_select(0, positions // 2))
            result = result.index_copy(0, positions, updates.to(result.dtype))
        first, second = result.reshape(batch, 2, *unified.shape[1:]).unbind(dim=1)
        return first, second

    def _conditional_pair_update(self, unified: torch.Tensor, selected: torch.Tensor,
                                 logits: torch.Tensor, router_features: torch.Tensor,
                                 missing: torch.Tensor, original_mask: torch.Tensor,
                                 support: torch.Tensor, hidden: torch.Tensor,
                                 shared_update: torch.Tensor, completion: torch.Tensor,
                                 x_input: torch.Tensor, step: int,
                                 temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second = self._dispatch_pair_updates(unified, selected, step)
        pair_logits = logits.float().gather(1, selected)
        native_second = F.softmax(pair_logits / temperature, dim=-1)[:, 1]
        if self.fusion_mode in {"local_response", "proposal"}:
            gate_features = [first, second, (first - second).abs()]
            if self.fusion_mode == "proposal":
                # Probe the two selected experts' candidate predictions using only
                # observed inputs. Detaching the probes isolates the fusion change.
                with torch.no_grad():
                    base_state = hidden.detach() + shared_update.detach()
                    routed_scale = self.routed_scale_logits[step].detach().sigmoid()
                    first_proposal = self.decoder(base_state + routed_scale * first.detach())
                    second_proposal = self.decoder(base_state + routed_scale * second.detach())
                    gate_features.extend((
                        (first_proposal - x_input).abs() * original_mask,
                        (second_proposal - x_input).abs() * original_mask,
                        (first_proposal - completion).abs() * missing,
                        (second_proposal - completion).abs() * missing,
                        (first_proposal - second_proposal).abs(),
                    ))
            gate_features.extend((original_mask, support))
            gate_features = torch.cat(gate_features, dim=1)
            correction = self.fusion_gate(gate_features).float()
            second_weight = torch.sigmoid(
                ((pair_logits[:, 1] - pair_logits[:, 0])[:, None, None, None, None]
                 + correction) / temperature
            )
            baseline = native_second[:, None, None, None, None]
        else:
            point_missing = missing.mean(dim=1, keepdim=True)
            response = torch.cat((
                first.mean(dim=(2, 3, 4)), second.mean(dim=(2, 3, 4)),
                self._missing_pool(first, point_missing),
                self._missing_pool(second, point_missing),
                self._missing_pool((first - second).abs(), point_missing),
            ), dim=1)
            if self.fusion_mode == "context":
                response = torch.zeros_like(response)
            identities = self.fusion_identity(selected).flatten(start_dim=1)
            features = torch.cat((router_features, pair_logits.to(router_features.dtype),
                                  identities, response), dim=1)
            correction = self.fusion_gate(features).float()
            corrected_logits = torch.stack((pair_logits[:, 0],
                                            pair_logits[:, 1] + correction[:, 0]), dim=1)
            second_weight = F.softmax(corrected_logits / temperature, dim=-1)[:, 1]
            baseline = native_second
        if second_weight.ndim == 1:
            weight = second_weight[:, None, None, None, None]
        else:
            weight = second_weight
        update = first * (1 - weight) + second * weight
        return update, second_weight.detach().float().mean(), (second_weight - baseline).detach().abs().mean()

    def forward(self, x_f: torch.Tensor, m_f: torch.Tensor, **kwargs: object) -> dict:
        if x_f.ndim != 5 or any(size < 1 for size in x_f.shape):
            raise ValueError("x_f must have nonempty shape [B, C, T, H, W]")
        if not x_f.is_floating_point():
            raise ValueError("x_f must have a floating-point dtype")
        if x_f.shape[1] != self.c_in:
            raise ValueError(f"Expected c_in={self.c_in}, got {x_f.shape[1]}")
        if (
            m_f.ndim != 5
            or m_f.shape[0] != x_f.shape[0]
            or m_f.shape[2:] != x_f.shape[2:]
            or m_f.shape[1] not in (1, self.c_in)
        ):
            raise ValueError("m_f must match x_f with either 1 or c_in mask channels")
        if m_f.device != x_f.device:
            raise ValueError("x_f and m_f must be on the same device")
        if not bool(torch.all((m_f == 0) | (m_f == 1))):
            raise ValueError("m_f must contain only finite binary values")
        forced_partner_step = kwargs.get("forced_partner_step")
        forced_partner_ids = kwargs.get("forced_partner_ids")
        forced_pair_step = kwargs.get("forced_pair_step")
        forced_pair_indices = kwargs.get("forced_pair_indices")
        if self.pair_mode not in {"partner", "partner_residual"} and forced_partner_step is not None:
            raise ValueError("forced_partner_step requires partner routing")
        if forced_partner_step is not None and (
            type(forced_partner_step) is not int or not 0 <= forced_partner_step < self.num_steps
            or not torch.is_tensor(forced_partner_ids)
            or forced_partner_ids.shape != (x_f.shape[0],)
        ):
            raise ValueError("forced partner needs a valid step and one partner per sample")
        if forced_pair_step is not None and (
            type(forced_pair_step) is not int or not 0 <= forced_pair_step < self.num_steps
            or not torch.is_tensor(forced_pair_indices)
            or forced_pair_indices.shape != (x_f.shape[0], 2)
        ):
            raise ValueError("forced pair needs a valid step and two experts per sample")
        observed = m_f.bool().expand_as(x_f)
        if not bool(torch.all(torch.isfinite(x_f) | ~observed)):
            raise ValueError("Input observed values must be finite; mark missing values with m_f=0")
        original_mask = observed.to(x_f.dtype)
        x_input = torch.where(observed, x_f, torch.zeros_like(x_f))
        support = compute_observation_support(
            original_mask, self.temporal_kernel, self.spatial_kernel
        ).to(x_f.dtype)
        missing = 1 - original_mask
        support_missing = missing.repeat(1, len(SUPPORT_FEATURE_NAMES), 1, 1, 1)
        support_summary = torch.cat(
            [
                support.mean(dim=(2, 3, 4)),
                support.std(dim=(2, 3, 4), unbiased=False),
                support.amax(dim=(2, 3, 4)),
                self._missing_pool(support, support_missing),
            ],
            dim=1,
        )
        position = self._position(x_input)
        hidden = self.encoder(torch.cat([x_input, original_mask, support, position], dim=1))
        prediction = self.decoder(hidden)
        completion = torch.where(observed, x_input, prediction)
        change = torch.zeros_like(prediction)
        initial_hidden = hidden
        initial_prediction, initial_completion = prediction, completion
        pattern_summary = (
            observed_pattern_features(x_input, observed, self.router_value_std)
            if self.router_features == "grouped" else None
        )
        initial_router_features = self._router_features(
            hidden, completion, support_summary, change, missing, pattern_summary
        )
        hard_fraction, uniform_mix, sampling_temperature = self.routing_schedule()

        predictions, completions, changes = [], [], []
        candidate_predictions, candidate_completions, candidate_changes = [], [], []
        acceptance_history = []
        logits_history, probability_history, weight_history, path_history = [], [], [], []
        selected_history, pair_id_history, pair_logit_history, pair_prob_history = [], [], [], []
        importance_history, partner_score_history, primary_history, partner_history = [], [], [], []
        partner_rank_score_history, partner_fusion_weight_history = [], []
        partner_correction_abs_history = []
        fusion_second_history, fusion_shift_history = [], []
        previous_choice_history = []
        previous_candidate_prediction = initial_prediction
        for step in range(self.num_steps):
            forced_expert = self.fixed_expert_steps[step]
            forced_index = self.expert_names.index(forced_expert) if forced_expert is not None else None
            previous_choice = None
            if self.previous_expert_context:
                previous_choice = (F.one_hot(path_history[-1].detach(), self.num_experts).float()
                                   if step else hidden.new_zeros((hidden.shape[0], self.num_experts)))
                previous_choice_history.append(previous_choice)
            pair_bias = None
            if self.use_routed and self.routing_mode != "fixed" and forced_index is None:
                router_features = (
                    initial_router_features
                    if self.router_state == "initial"
                    else self._router_features(
                        hidden,
                        completion if self.completion_feedback else initial_completion,
                        support_summary,
                        change if self.completion_feedback else torch.zeros_like(change),
                        missing, pattern_summary,
                    )
                )
                router_features = self._add_router_input_noise(router_features, step)
                if self.global_route_weights:
                    logits = self.global_route_logits[step].to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0).expand(hidden.shape[0], -1)
                else:
                    with torch.autocast(device_type=hidden.device.type, enabled=False) if self.router_fp32 else nullcontext():
                        features = router_features.float() if self.router_fp32 else router_features
                        scored = (self.routers[step](features, previous_choice)
                                  if self.previous_expert_context and step else self.routers[step](features))
                        if self.pair_mode == "native":
                            logits = scored
                        else:
                            logits, pair_bias = scored
            elif forced_index is not None:
                logits = hidden.new_full((hidden.shape[0], self.num_experts), -20.0)
                logits[:, forced_index] = 20.0
            else:
                logits = hidden.new_zeros((hidden.shape[0], self.num_experts))
            # The partner route observes the first expert's proposal before
            # choosing a companion; all variants reuse this same U and shared update.
            expert_hidden = initial_hidden if self.expert_state == "initial" else hidden
            expert_completion = (initial_completion if self.expert_state == "initial" or
                                 not self.completion_feedback else completion)
            unified = self.state_norm(self.state_projection(torch.cat(
                [expert_hidden, expert_completion, original_mask, support, position], dim=1
            )))
            shared_update = (self.shared_scale_logits[step].sigmoid() * self.shared_expert(unified)
                             if self.use_shared else torch.zeros_like(hidden))
            primary_update = None
            partner_scores = None
            # Gumbel hard argmax is invariant to tau: its clean categorical
            # probabilities are softmax(logits). Tau controls the surrogate
            # gradient; only the soft-mixture variant tempers its actual weights.
            probability_logits = (
                logits / self.temperature
                if self.routing_mode in {"soft", "parallel"} else logits / sampling_temperature
            )
            probabilities = F.softmax(probability_logits.float(), dim=-1)
            if not self.use_routed:
                weights = torch.zeros_like(probabilities)
                paths = torch.full((hidden.shape[0],), -1, device=hidden.device, dtype=torch.long)
            elif forced_index is not None:
                paths = torch.full((hidden.shape[0],), forced_index, device=hidden.device, dtype=torch.long)
                weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)
            elif self.routing_mode == "fixed":
                expert_index = self.expert_names.index(self.fixed_path[step])
                paths = torch.full((hidden.shape[0],), expert_index, device=hidden.device, dtype=torch.long)
                weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)
            elif self.routing_mode in {"soft", "parallel"}:
                weights = probabilities
                paths = weights.argmax(dim=-1)
            elif self.pair_mode in {"partner", "partner_residual"}:
                native_top2 = logits.float().topk(2, dim=-1).indices
                primary = (native_top2[:, 0] if self.pair_mode == "partner_residual"
                           else logits.argmax(dim=-1))
                primary_update = self._dispatch(unified, primary, step)
                proposal_hidden = (hidden + shared_update +
                                   self.routed_scale_logits[step].sigmoid() * primary_update)
                proposal = self.decoder(proposal_hidden)
                proposal_change = torch.where(observed, torch.zeros_like(proposal),
                                              proposal - prediction)
                point_missing = missing.mean(dim=1, keepdim=True)
                partner_features = torch.cat((
                    router_features,
                    primary_update.mean(dim=(2, 3, 4)),
                    self._missing_pool(primary_update, point_missing),
                    proposal_change.mean(dim=(2, 3, 4)),
                    self._missing_pool(proposal_change, missing),
                ), dim=1)
                correction = self.partner_scorer(partner_features, primary)
                if self.pair_mode == "partner_residual":
                    primary_mask = F.one_hot(primary, self.num_experts).bool()
                    partner_correction_abs_history.append(
                        correction.detach().masked_fill(primary_mask, 0.).abs().sum(-1).div(self.num_experts - 1).mean()
                    )
                    # Zero-initialized correction preserves native Top-2,
                    # including torch.topk's tie rule.
                    partner_scores = logits.float() + correction.float()
                    candidate_scores = partner_scores.masked_fill(primary_mask, -torch.inf)
                    best = candidate_scores.max(dim=-1).values
                    native_second = native_top2[:, 1]
                    native_score = candidate_scores.gather(1, native_second[:, None]).squeeze(1)
                    partner = torch.where(native_score == best, native_second,
                                          candidate_scores.argmax(dim=-1))
                else:
                    partner_scores = correction
                    partner = partner_scores.argmax(dim=-1)
                if self.partner_aux_head_only:
                    rank_correction = self.partner_scorer(partner_features.detach(), primary)
                    partner_rank_score_history.append(logits.detach().float() + rank_correction.float())
                if forced_partner_step == step:
                    partner = forced_partner_ids.to(device=primary.device, dtype=torch.long)
                    if bool(((partner == primary) | (partner < 0) |
                             (partner >= self.num_experts)).any()):
                        raise ValueError("forced partner must differ from the primary expert")
                selected = torch.stack((primary, partner), dim=-1)
                if forced_pair_step == step:
                    selected = forced_pair_indices.to(device=primary.device, dtype=torch.long)
                if bool(((selected < 0) | (selected >= self.num_experts)).any()) or bool(
                    (selected[:, 0] == selected[:, 1]).any()
                ):
                    raise ValueError("forced pair must contain two distinct valid experts")
                fusion_logits = logits.float().gather(1, selected)
                if self.partner_fusion == "corrected" and forced_pair_step != step:
                    fusion_logits = torch.stack((
                        fusion_logits[:, 0], partner_scores.gather(1, selected[:, 1:]).squeeze(1)
                    ), dim=-1)
                within = F.softmax(fusion_logits / sampling_temperature, dim=-1)
                weights = torch.zeros_like(probabilities).scatter(1, selected, within)
                if self.partner_fusion == "corrected" and forced_pair_step != step:
                    partner_fusion_weight_history.append(within[:, 1].detach().float().mean())
                paths = weights.argmax(dim=-1)
                ordered = selected.sort(dim=-1).values
                first, second = ordered.unbind(dim=-1)
                pair_ids = first * (2 * self.num_experts - first - 1) // 2 + (second - first - 1)
                left, right = self.pair_indices.unbind(dim=-1)
                pair_logits = torch.where(
                    primary[:, None] == left[None], partner_scores[:, right],
                    torch.where(primary[:, None] == right[None], partner_scores[:, left],
                                partner_scores.new_full((len(primary), len(left)), -1e4)),
                )
                pair_probs = F.softmax(pair_logits.float() / sampling_temperature, dim=-1)
                partner_score_history.append(partner_scores)
                primary_history.append(primary)
                partner_history.append(partner)
            elif self.pair_mode != "native":
                dense_fraction = (1 - hard_fraction if step < self.pair_dense_warmup_steps else 0.)
                weights, pair_ids, pair_logits, pair_probs = self._pair_route(
                    logits, pair_bias, sampling_temperature, dense_fraction, uniform_mix
                )
                paths = weights.argmax(dim=-1)
            elif forced_pair_step == step:
                selected = forced_pair_indices.to(device=logits.device, dtype=torch.long)
                if bool(((selected < 0) | (selected >= self.num_experts)).any()) or bool(
                    (selected[:, 0] == selected[:, 1]).any()
                ):
                    raise ValueError("forced pair must contain two distinct valid experts")
                within = F.softmax(logits.float().gather(1, selected) /
                                   sampling_temperature, dim=-1)
                weights = torch.zeros_like(probabilities).scatter(1, selected, within)
                paths = weights.argmax(dim=-1)
            elif self.training:
                if self.top_k > 1:
                    # Top-K keeps only the selected experts and renormalizes
                    # their clean probabilities. This is a differentiable
                    # weighted fusion inside a discrete selected set.
                    sample_logits = (logits.float() if self.router_fp32 else logits) / sampling_temperature
                    selected = sample_logits.topk(self.top_k, dim=-1).indices
                    selection_mask = F.one_hot(selected, self.num_experts).any(dim=-2).to(probabilities.dtype)
                    weights = probabilities * selection_mask
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                else:
                    with torch.autocast(device_type=hidden.device.type, enabled=False) if self.router_fp32 else nullcontext():
                        sample_logits = (logits.float() if self.router_fp32 else logits) / sampling_temperature
                        if hard_fraction < 1:
                            soft = uniform_mix / self.num_experts + (1 - uniform_mix) * probabilities
                            if hard_fraction == 0:
                                weights = soft
                            else:
                                hard = F.gumbel_softmax(sample_logits, tau=self.temperature, hard=True, dim=-1)
                                weights = (1 - hard_fraction) * soft + hard_fraction * hard
                        else:
                            weights = F.gumbel_softmax(sample_logits, tau=self.temperature, hard=True, dim=-1)
                paths = weights.argmax(dim=-1)
            else:
                if self.top_k > 1:
                    selected = logits.topk(self.top_k, dim=-1).indices
                    selection_mask = F.one_hot(selected, self.num_experts).any(dim=-2).to(probabilities.dtype)
                    weights = probabilities * selection_mask
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    paths = weights.argmax(dim=-1)
                else:
                    paths = logits.argmax(dim=-1)
                    weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)

            if self.use_routed and self.routing_mode == "hard" and self.top_k == 2:
                if self.pair_mode == "native":
                    selected = weights.topk(2, dim=-1).indices.sort(dim=-1).values
                    first, second = selected.unbind(dim=-1)
                    pair_ids = first * (2 * self.num_experts - first - 1) // 2 + (second - first - 1)
                    individual = logits.float()
                    pair_logits = (individual[:, self.pair_indices[:, 0]] +
                                   individual[:, self.pair_indices[:, 1]])
                    pair_probs = F.softmax(pair_logits / sampling_temperature, dim=-1)
                else:
                    selected = self.pair_indices[pair_ids]
                selected_history.append(selected)
                pair_id_history.append(pair_ids)
                pair_logit_history.append(pair_logits)
                pair_prob_history.append(pair_probs)
                corrected_scores = (partner_scores if self.pair_mode == "partner_residual" and
                                    self.partner_fusion == "corrected" else None)
                importance_history.append(self._pair_importance(
                    logits, pair_probs, sampling_temperature, corrected_scores,
                    primary if corrected_scores is not None else None,
                ))
            else:
                importance_history.append(probabilities)

            precomputed_routed_update = None
            if self.fusion_mode != "original":
                active_ids = (forced_pair_indices.to(device=logits.device, dtype=torch.long)
                              if forced_pair_step == step else logits.float().topk(2, dim=-1).indices)
                precomputed_routed_update, second_mean, shift_mean = self._conditional_pair_update(
                    unified, active_ids, logits, router_features, missing, original_mask,
                    support, hidden, shared_update, completion, x_input,
                    step, sampling_temperature,
                )
                fusion_second_history.append(second_mean)
                fusion_shift_history.append(shift_mean)
            update = shared_update
            if self.use_routed:
                if precomputed_routed_update is not None:
                    routed_update = precomputed_routed_update
                elif self.routing_mode in {"soft", "parallel"}:
                    routed_update = torch.zeros_like(unified)
                    for expert_index, expert in enumerate(self.routed_experts(step)):
                        routed_update = routed_update + (
                            weights[:, expert_index, None, None, None, None] * expert(unified)
                        )
                elif self.routing_mode == "hard":
                    if (self.pair_mode in {"partner", "partner_residual"} and primary_update is not None and
                            forced_pair_step != step):
                        companion_update = self._dispatch(unified, partner, step)
                        primary_weight = weights.gather(1, primary[:, None]).flatten()
                        companion_weight = weights.gather(1, partner[:, None]).flatten()
                        routed_update = (
                            primary_update * primary_weight[:, None, None, None, None] +
                            companion_update * companion_weight[:, None, None, None, None]
                        )
                    else:
                        routed_update = self._dispatch_weighted(unified, weights, step)
                else:
                    routed_update = self._dispatch(unified, paths, step)
                update = update + self.routed_scale_logits[step].sigmoid() * routed_update
            hidden = hidden + update
            candidate_prediction = self.decoder(hidden)
            candidate_completion = torch.where(observed, x_input, candidate_prediction)
            candidate_change = torch.where(observed, torch.zeros_like(candidate_prediction),
                                           (candidate_prediction - previous_candidate_prediction).abs())
            old_completion = completion
            if self.acceptance_head is None:
                acceptance_weight = torch.ones_like(candidate_prediction)
                completion = candidate_completion
            else:
                proposal = candidate_completion - old_completion
                gate_features = torch.cat(
                    [hidden, old_completion, candidate_completion, proposal, original_mask, support], dim=1
                )
                gate_logits = self.acceptance_head(gate_features)
                if self.acceptance == "window":
                    gate_logits = gate_logits.mean(dim=(2, 3, 4), keepdim=True)
                acceptance_weight = torch.sigmoid(gate_logits)
                completion = torch.where(observed, x_input, old_completion + acceptance_weight * proposal)
            prediction = torch.where(observed, candidate_prediction, completion)
            change = torch.where(observed, torch.zeros_like(prediction),
                                 (completion - old_completion).abs())
            previous_candidate_prediction = candidate_prediction
            predictions.append(prediction)
            completions.append(completion)
            changes.append(change)
            candidate_predictions.append(candidate_prediction)
            candidate_completions.append(candidate_completion)
            candidate_changes.append(candidate_change)
            acceptance_history.append(acceptance_weight)
            logits_history.append(logits)
            probability_history.append(probabilities)
            weight_history.append(weights)
            path_history.append(paths)

        route_probabilities = torch.stack(probability_history, dim=1)
        route_importance = torch.stack(importance_history, dim=1)
        route_weights = torch.stack(weight_history, dim=1)
        effective_mode = ("soft" if self.routing_mode == "hard" and hard_fraction < 1
                          and (self.top_k == 1 or self.pair_dense_warmup_steps > 0)
                          else self.routing_mode)
        diagnostics = {
            "hard_fraction": hidden.new_tensor(hard_fraction),
            "sampling_temperature": hidden.new_tensor(sampling_temperature),
            "update_abs_mean": changes[-1].detach().mean(),
            "missing_fraction": missing.detach().mean(),
            "shared_residual_scale": self.shared_scale_logits.detach().sigmoid().mean(),
            "routed_residual_scale": self.routed_scale_logits.detach().sigmoid().mean(),
            "acceptance_mean": torch.stack(acceptance_history).detach().float().mean(),
            "pair_dense_warmup_fraction": hidden.new_tensor(
                1 - hard_fraction if self.pair_dense_warmup_steps else 0.
            ),
        }
        for step, step_change in enumerate(changes, start=1):
            if self.previous_expert_context and step > 1:
                diagnostics[f'step{step}_previous_expert_embedding_norm'] = self.routers[step - 1].previous_expert_embedding.detach().float().norm()
            step_logits = logits_history[step - 1].detach().float()
            diagnostics[f"step{step}_logit_abs_max"] = step_logits.abs().max()
            diagnostics[f"step{step}_logit_range"] = (step_logits.max(-1).values - step_logits.min(-1).values).mean()
            if self.num_experts > 1:
                top2 = step_logits.topk(2, dim=-1).values
                diagnostics[f"step{step}_top12_margin"] = (top2[:, 0] - top2[:, 1]).mean()
            diagnostics[f"step{step}_update_abs_mean"] = step_change.detach().float().mean()
            if self.fusion_mode != "original":
                diagnostics[f"step{step}_fusion_second_weight_mean"] = fusion_second_history[step - 1]
                diagnostics[f"step{step}_fusion_shift_abs_mean"] = fusion_shift_history[step - 1]
            if step <= self.pair_dense_warmup_steps:
                diagnostics[f"step{step}_dense_route_fraction"] = hidden.new_tensor(1 - hard_fraction)
            if self.partner_fusion == "corrected" and len(partner_fusion_weight_history) >= step:
                diagnostics[f"step{step}_partner_fusion_weight_mean"] = (
                    partner_fusion_weight_history[step - 1]
                )
            if self.pair_mode == "partner_residual" and len(partner_correction_abs_history) >= step:
                diagnostics[f"step{step}_partner_correction_abs_mean"] = (
                    partner_correction_abs_history[step - 1].detach().float()
                )
                ranked = step_logits.topk(3, dim=-1).values
                diagnostics[f"step{step}_native_partner_margin"] = (ranked[:, 1] - ranked[:, 2]).mean()
            if self.routing_mode == "hard" and self.top_k == 2:
                native_pair = step_logits.topk(2, dim=-1).indices.sort(dim=-1).values
                chosen_pair = selected_history[step - 1]
                diagnostics[f"step{step}_pair_vs_top2_disagreement_rate"] = (
                    (chosen_pair != native_pair).any(dim=-1).float().mean()
                )
        if self.use_routed and self.routing_mode != "fixed":
            diagnostic_probabilities = route_probabilities.detach().float()
            diagnostics["route_entropy"] = -(
                diagnostic_probabilities * diagnostic_probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            candidate_marginal = route_importance.detach().float()
            diagnostics["candidate_importance_entropy"] = -(
                candidate_marginal * candidate_marginal.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            diagnostics["candidate_importance_vs_selected_l1"] = (
                candidate_marginal - route_weights.detach().float()
            ).abs().sum(dim=-1).mean()
        return {
            "x_hat_main": prediction,
            "h_st_aux": hidden,
            "diagnostics": {"coe": diagnostics},
            "coe": {
                "predictions": predictions,
                "completions": completions,
                "candidate_predictions": candidate_predictions,
                "candidate_completions": candidate_completions,
                "candidate_changes": candidate_changes,
                "acceptance_weights": torch.stack(acceptance_history, dim=1),
                "selected_experts": (torch.stack(selected_history, dim=1) if selected_history else None),
                "pair_ids": (torch.stack(pair_id_history, dim=1) if pair_id_history else None),
                "pair_logits": (torch.stack(pair_logit_history, dim=1) if pair_logit_history else None),
                "pair_probs": (torch.stack(pair_prob_history, dim=1) if pair_prob_history else None),
                "initial_prediction": initial_prediction,
                "initial_completion": initial_completion,
                "changes": changes,
                "route_logits": torch.stack(logits_history, dim=1),
                "route_probs": route_probabilities,
                "route_importance": route_importance,
                "route_weights": route_weights,
                "partner_scores": (torch.stack(partner_score_history, dim=1)
                                   if partner_score_history else None),
                "partner_rank_scores": (torch.stack(partner_rank_score_history, dim=1)
                                        if partner_rank_score_history else None),
                "primary_ids": (torch.stack(primary_history, dim=1)
                                if primary_history else None),
                "partner_ids": (torch.stack(partner_history, dim=1)
                                if partner_history else None),
                "paths": torch.stack(path_history, dim=1),
                "paths_are_discrete": self.use_routed and effective_mode not in {"soft", "parallel"},
                "routing_mode": effective_mode,
                "configured_routing_mode": self.routing_mode,
                "router_state": self.router_state,
                "expert_state": self.expert_state,
                "pair_mode": self.pair_mode,
                "partner_fusion": self.partner_fusion,
                "partner_aux_head_only": self.partner_aux_head_only,
                "fusion_mode": self.fusion_mode,
                "pair_dense_warmup_steps": self.pair_dense_warmup_steps,
                "acceptance": self.acceptance,
                "expert_sharing": self.expert_sharing,
                "completion_feedback": self.completion_feedback,
                "previous_expert_context": self.previous_expert_context,
                "previous_expert_choices": (torch.stack(previous_choice_history, dim=1)
                                            if previous_choice_history else None),
                "use_shared": self.use_shared,
                "use_routed": self.use_routed,
                "expert_names": self.expert_names,
                "num_experts": self.num_experts,
                "num_steps": self.num_steps,
                "top_k": self.top_k,
                "support": support,
                "support_feature_names": SUPPORT_FEATURE_NAMES,
                "observation_mask": original_mask,
                "diagnostics": diagnostics,
            },
        }
