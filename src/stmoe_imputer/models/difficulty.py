from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


DIFFICULTY_STAT_NAMES = (
    "missing_rate",
    "observed_ratio",
    "temporal_gap_score",
    "spatial_block_score",
    "neighbor_density",
    "local_value_variance",
    "temporal_variance",
    "scale_reliability",
    "cross_scale_consistency",
)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(value).to(dtype=value.dtype)
    dims = tuple(range(1, value.ndim))
    return (value * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1.0)


def _observed_variance(x_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(x_obs).to(dtype=x_obs.dtype)
    dims = tuple(range(1, x_obs.ndim))
    count = expanded.sum(dim=dims).clamp_min(1.0)
    mean = (x_obs * expanded).sum(dim=dims) / count
    centered = (x_obs - mean.view(-1, 1, 1, 1, 1)) ** 2
    return (centered * expanded).sum(dim=dims) / count


def compute_raw_difficulty_stats(
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    reliability: torch.Tensor | None = None,
    cross_scale_reference: torch.Tensor | None = None,
    use_spatial_block: bool = True,
    use_cross_scale_consistency: bool = True,
) -> torch.Tensor:
    """Return nine finite, sample-level imputation difficulty descriptors."""
    if x_obs.ndim != 5 or mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(
            f"Expected x_obs [B,C,T,H,W] and mask [B,1,T,H,W], got "
            f"{tuple(x_obs.shape)} and {tuple(mask.shape)}"
        )
    batch_size, _, time_steps, height, width = x_obs.shape
    observed_ratio = mask.float().mean(dim=(1, 2, 3, 4))
    missing_rate = 1.0 - observed_ratio
    missing = 1.0 - mask.float()

    missing_per_t = missing.mean(dim=(1, 3, 4))
    if time_steps > 1:
        temporal_gap = (missing_per_t[:, 1:] * missing_per_t[:, :-1]).mean(dim=1)
    else:
        temporal_gap = missing_per_t[:, 0]

    if use_spatial_block and height >= 2 and width >= 2:
        kernel = min(4, height, width)
        missing_2d = missing.permute(0, 2, 1, 3, 4).reshape(
            batch_size * time_steps, 1, height, width
        )
        block = F.avg_pool2d(missing_2d, kernel_size=kernel, stride=1)
        spatial_block = block.amax(dim=(1, 2, 3)).reshape(batch_size, time_steps).mean(dim=1)
    else:
        spatial_block = torch.zeros_like(missing_rate)

    missing_count = missing.sum(dim=(1, 2, 3, 4))
    if height >= 3 and width >= 3:
        neighbor = F.avg_pool3d(
            mask.float(), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )
        neighbor_density = (neighbor * missing).sum(dim=(1, 2, 3, 4))
        neighbor_density = neighbor_density / missing_count.clamp_min(1.0)
        neighbor_density = torch.where(missing_count > 0, neighbor_density, observed_ratio)
    else:
        neighbor_density = observed_ratio

    local_variance = _observed_variance(x_obs, mask)

    if time_steps > 1:
        temporal_valid = mask[:, :, 1:] * mask[:, :, :-1]
        temporal_delta = (x_obs[:, :, 1:] - x_obs[:, :, :-1]) ** 2
        temporal_variance = _masked_mean(temporal_delta, temporal_valid)
    else:
        temporal_variance = torch.zeros_like(missing_rate)

    if reliability is None:
        scale_reliability = observed_ratio
    else:
        scale_reliability = reliability.float().mean(dim=(1, 2, 3, 4))

    if use_cross_scale_consistency and cross_scale_reference is not None:
        reference = cross_scale_reference
        if reference.shape[-3:] != x_obs.shape[-3:]:
            reference = F.interpolate(
                reference, size=x_obs.shape[-3:], mode="trilinear", align_corners=False
            )
        difference = _masked_mean((x_obs - reference).abs(), mask)
        magnitude = _masked_mean(x_obs.abs(), mask).clamp_min(1e-6)
        cross_scale_consistency = torch.exp(-difference / magnitude)
    else:
        cross_scale_consistency = torch.zeros_like(missing_rate)

    stats = torch.stack(
        (
            missing_rate,
            observed_ratio,
            temporal_gap,
            spatial_block,
            neighbor_density,
            local_variance,
            temporal_variance,
            scale_reliability,
            cross_scale_consistency,
        ),
        dim=1,
    )
    return torch.nan_to_num(stats, nan=0.0, posinf=1e4, neginf=-1e4)


class DifficultyDescriptor(nn.Module):
    """Encode explicit imputation difficulty statistics into a compact vector."""

    def __init__(
        self,
        out_dim: int = 16,
        hidden_dim: int = 32,
        zero_init: bool = True,
        dropout: float = 0.1,
        use_spatial_block: bool = True,
        use_cross_scale_consistency: bool = True,
    ) -> None:
        super().__init__()
        self.use_spatial_block = use_spatial_block
        self.use_cross_scale_consistency = use_cross_scale_consistency
        self.proj = nn.Sequential(
            nn.Linear(len(DIFFICULTY_STAT_NAMES), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        if zero_init:
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        h: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
        cross_scale_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del h  # Reserved for later feature-conditioned difficulty descriptors.
        raw_stats = compute_raw_difficulty_stats(
            x_obs=x_obs,
            mask=mask,
            reliability=reliability,
            cross_scale_reference=cross_scale_reference,
            use_spatial_block=self.use_spatial_block,
            use_cross_scale_consistency=self.use_cross_scale_consistency,
        )
        return self.proj(raw_stats), raw_stats


def aggregate_difficulty_score(raw_stats: torch.Tensor) -> torch.Tensor:
    """Build a bounded diagnostic score from the interpretable raw statistics."""
    missing = raw_stats[:, 0]
    temporal_gap = raw_stats[:, 2]
    spatial_block = raw_stats[:, 3]
    sparse_neighbors = 1.0 - raw_stats[:, 4].clamp(0.0, 1.0)
    unreliable = 1.0 - raw_stats[:, 7].clamp(0.0, 1.0)
    inconsistent = 1.0 - raw_stats[:, 8].clamp(0.0, 1.0)
    return torch.stack(
        (missing, temporal_gap, spatial_block, sparse_neighbors, unreliable, inconsistent),
        dim=1,
    ).mean(dim=1)
