"""Single-scale, state-conditioned temporal/spatial Chain of Experts.

All tensors use ``[B, C, T, H, W]``.  The original observation mask stays
fixed throughout the chain; a decoded estimate never becomes an observation.
Hard training evaluates all configured experts for the straight-through routing
gradient, while hard evaluation dispatches whole windows to their chosen expert.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .coe_pattern_experts import (
    DilatedDirectionalExpert,
    JointSpatioTemporalExpert,
    TemporalAttentionExpert,
)


SUPPORTED_EXPERT_NAMES = ("T", "S", "TD", "SD", "TA", "ST")


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
    """

    requires_multiscale = False

    def __init__(
        self,
        c_in: int,
        dim: int = 64,
        num_steps: int = 2,
        routing_mode: str = "hard",
        fixed_path: Sequence[str] | None = None,
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
        unknown = set(self.expert_names) - set(SUPPORTED_EXPERT_NAMES)
        if unknown:
            raise ValueError(f"Unknown expert labels: {sorted(unknown)}")
        self.num_experts = len(self.expert_names)
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
        self.temporal_expert = (
            DirectionalExpert(dim, "temporal", self.temporal_kernel)
            if "T" in self.expert_names else None
        )
        self.spatial_expert = (
            DirectionalExpert(dim, "spatial", self.spatial_kernel)
            if "S" in self.expert_names else None
        )
        # Keep the legacy T/S module names for existing two-expert checkpoints.
        # Register only requested additions; no expert is replicated per step.
        self.pattern_experts = nn.ModuleDict()
        for name in self.expert_names:
            if name == "TD":
                self.pattern_experts[name] = DilatedDirectionalExpert(
                    dim, "temporal", self.temporal_kernel, temporal_dilation
                )
            elif name == "SD":
                self.pattern_experts[name] = DilatedDirectionalExpert(
                    dim, "spatial", self.spatial_kernel, spatial_dilation
                )
            elif name == "TA":
                self.pattern_experts[name] = TemporalAttentionExpert(dim, attention_heads)
            elif name == "ST":
                self.pattern_experts[name] = JointSpatioTemporalExpert(dim)
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
        self.routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(router_input_dim),
                    nn.Linear(router_input_dim, router_hidden_dim),
                    nn.GELU(),
                    nn.Linear(router_hidden_dim, self.num_experts),
                )
                for _ in range(num_steps)
            ]
        )
        residual_logit = math.log(float(residual_init) / (1 - float(residual_init)))
        self.shared_scale_logits = nn.Parameter(torch.full((num_steps,), residual_logit))
        self.routed_scale_logits = nn.Parameter(torch.full((num_steps,), residual_logit))

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
        )

    def routed_experts(self) -> tuple[nn.Module, ...]:
        """Return the registered operators in router-column order."""
        base = {"T": self.temporal_expert, "S": self.spatial_expert}
        return tuple(
            base[name] if name in base else self.pattern_experts[name]
            for name in self.expert_names
        )

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
    ) -> torch.Tensor:
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
            ],
            dim=1,
        )

    def _dispatch(self, unified: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """Evaluate only selected whole windows, keeping their full context."""
        result = torch.zeros_like(unified)
        for expert_index, expert in enumerate(self.routed_experts()):
            selected = torch.nonzero(paths == expert_index, as_tuple=False).flatten()
            if selected.numel():
                updates = expert(unified.index_select(0, selected))
                # Autocast can return FP16/BF16 expert outputs while the
                # normalized state/accumulator is FP32. index_copy requires
                # matching dtypes; the differentiable cast preserves gradients
                # and accumulates every expert in the state's precision.
                result = result.index_copy(0, selected, updates.to(result.dtype))
        return result

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
        initial_router_features = self._router_features(
            hidden, completion, support_summary, change, missing
        )

        predictions, completions, changes = [], [], []
        logits_history, probability_history, weight_history, path_history = [], [], [], []
        for step in range(self.num_steps):
            if self.use_routed and self.routing_mode != "fixed":
                router_features = (
                    initial_router_features
                    if self.router_state == "initial"
                    else self._router_features(hidden, completion, support_summary, change, missing)
                )
                logits = self.routers[step](router_features)
            else:
                logits = hidden.new_zeros((hidden.shape[0], self.num_experts))
            # Gumbel hard argmax is invariant to tau: its clean categorical
            # probabilities are softmax(logits). Tau controls the surrogate
            # gradient; only the soft-mixture variant tempers its actual weights.
            probability_logits = (
                logits / self.temperature
                if self.routing_mode in {"soft", "parallel"} else logits
            )
            probabilities = F.softmax(probability_logits.float(), dim=-1)
            if not self.use_routed:
                weights = torch.zeros_like(probabilities)
                paths = torch.full((hidden.shape[0],), -1, device=hidden.device, dtype=torch.long)
            elif self.routing_mode == "fixed":
                expert_index = self.expert_names.index(self.fixed_path[step])
                paths = torch.full((hidden.shape[0],), expert_index, device=hidden.device, dtype=torch.long)
                weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)
            elif self.routing_mode in {"soft", "parallel"}:
                weights = probabilities
                paths = weights.argmax(dim=-1)
            elif self.training:
                weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
                paths = weights.argmax(dim=-1)
            else:
                paths = logits.argmax(dim=-1)
                weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)

            # Recompute this same interface each round in both state variants.
            # Freezing the expert input leaves residual accumulation and router
            # state policy independent, and retains candidate computation counts.
            expert_hidden = initial_hidden if self.expert_state == "initial" else hidden
            expert_completion = initial_completion if self.expert_state == "initial" else completion
            unified = self.state_norm(self.state_projection(torch.cat(
                [expert_hidden, expert_completion, original_mask, support, position], dim=1
            )))
            update = torch.zeros_like(hidden)
            if self.use_shared:
                update = update + self.shared_scale_logits[step].sigmoid() * self.shared_expert(unified)
            if self.use_routed:
                if self.routing_mode in {"soft", "parallel"} or (self.routing_mode == "hard" and self.training):
                    routed_update = torch.zeros_like(unified)
                    for expert_index, expert in enumerate(self.routed_experts()):
                        routed_update = routed_update + (
                            weights[:, expert_index, None, None, None, None] * expert(unified)
                        )
                else:
                    routed_update = self._dispatch(unified, paths)
                update = update + self.routed_scale_logits[step].sigmoid() * routed_update
            hidden = hidden + update
            previous_prediction = prediction
            prediction = self.decoder(hidden)
            completion = torch.where(observed, x_input, prediction)
            change = torch.where(observed, torch.zeros_like(prediction), (prediction - previous_prediction).abs())
            predictions.append(prediction)
            completions.append(completion)
            changes.append(change)
            logits_history.append(logits)
            probability_history.append(probabilities)
            weight_history.append(weights)
            path_history.append(paths)

        route_probabilities = torch.stack(probability_history, dim=1)
        route_weights = torch.stack(weight_history, dim=1)
        diagnostics = {
            "update_abs_mean": changes[-1].detach().mean(),
            "missing_fraction": missing.detach().mean(),
            "shared_residual_scale": self.shared_scale_logits.detach().sigmoid().mean(),
            "routed_residual_scale": self.routed_scale_logits.detach().sigmoid().mean(),
        }
        for step, step_change in enumerate(changes, start=1):
            diagnostics[f"step{step}_update_abs_mean"] = step_change.detach().float().mean()
        if self.use_routed and self.routing_mode != "fixed":
            diagnostic_probabilities = route_probabilities.detach().float()
            diagnostics["route_entropy"] = -(
                diagnostic_probabilities * diagnostic_probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
        return {
            "x_hat_main": prediction,
            "h_st_aux": hidden,
            "diagnostics": {"coe": diagnostics},
            "coe": {
                "predictions": predictions,
                "completions": completions,
                "initial_prediction": initial_prediction,
                "initial_completion": initial_completion,
                "changes": changes,
                "route_logits": torch.stack(logits_history, dim=1),
                "route_probs": route_probabilities,
                "route_weights": route_weights,
                "paths": torch.stack(path_history, dim=1),
                "paths_are_discrete": self.use_routed and self.routing_mode not in {"soft", "parallel"},
                "routing_mode": self.routing_mode,
                "router_state": self.router_state,
                "expert_state": self.expert_state,
                "use_shared": self.use_shared,
                "use_routed": self.use_routed,
                "expert_names": self.expert_names,
                "num_experts": self.num_experts,
                "num_steps": self.num_steps,
                "support": support,
                "support_feature_names": SUPPORT_FEATURE_NAMES,
                "observation_mask": original_mask,
                "diagnostics": diagnostics,
            },
        }
