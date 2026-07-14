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
    value_f = value.float()
    mask_f = mask.expand_as(value).float()
    dims = tuple(range(1, value.ndim))
    return (value_f * mask_f).sum(dim=dims) / mask_f.sum(dim=dims).clamp_min(1.0)


def _observed_variance(x_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x_f = x_obs.float()
    mask_f = mask.expand_as(x_obs).float()
    dims = tuple(range(1, x_obs.ndim))
    count = mask_f.sum(dim=dims).clamp_min(1.0)
    mean = (x_f * mask_f).sum(dim=dims) / count
    centered = (x_f - mean.view(-1, 1, 1, 1, 1)).square()
    return (centered * mask_f).sum(dim=dims) / count


def compute_raw_difficulty_stats(
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    reliability: torch.Tensor | None = None,
    cross_scale_reference: torch.Tensor | None = None,
    use_spatial_block: bool = True,
    use_cross_scale_consistency: bool = True,
) -> torch.Tensor:
    """Nine finite sample-level descriptors computed only from observations."""
    if x_obs.ndim != 5 or mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(
            f"Expected x_obs [B,C,T,H,W] and mask [B,1,T,H,W], got "
            f"{tuple(x_obs.shape)} and {tuple(mask.shape)}"
        )
    batch_size, _, time_steps, height, width = x_obs.shape
    mask_f = mask.float()
    observed_ratio = mask_f.mean(dim=(1, 2, 3, 4))
    missing_rate = 1.0 - observed_ratio
    missing = 1.0 - mask_f

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
        neighbor = F.avg_pool3d(mask_f, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        neighbor_density = (neighbor * missing).sum(dim=(1, 2, 3, 4))
        neighbor_density = neighbor_density / missing_count.clamp_min(1.0)
        neighbor_density = torch.where(missing_count > 0, neighbor_density, observed_ratio)
    else:
        neighbor_density = observed_ratio

    local_variance = _observed_variance(x_obs, mask)
    if time_steps > 1:
        temporal_valid = mask_f[:, :, 1:] * mask_f[:, :, :-1]
        temporal_delta = (x_obs.float()[:, :, 1:] - x_obs.float()[:, :, :-1]).square()
        temporal_variance = _masked_mean(temporal_delta, temporal_valid)
    else:
        temporal_variance = torch.zeros_like(missing_rate)

    if reliability is None:
        scale_reliability = observed_ratio
    else:
        scale_reliability = reliability.float().mean(dim=(1, 2, 3, 4))

    if use_cross_scale_consistency and cross_scale_reference is not None:
        reference = cross_scale_reference.float()
        if reference.shape[-3:] != x_obs.shape[-3:]:
            reference = F.interpolate(reference, size=x_obs.shape[-3:], mode="trilinear", align_corners=False)
        difference = _masked_mean((x_obs.float() - reference).abs(), mask)
        magnitude = _masked_mean(x_obs.float().abs(), mask).clamp_min(1e-6)
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


def aggregate_difficulty_score(raw_stats: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            raw_stats[:, 0],
            raw_stats[:, 2],
            raw_stats[:, 3],
            1.0 - raw_stats[:, 4].clamp(0.0, 1.0),
            1.0 - raw_stats[:, 7].clamp(0.0, 1.0),
            1.0 - raw_stats[:, 8].clamp(0.0, 1.0),
        ),
        dim=1,
    ).mean(dim=1)


class DifficultyConditionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        out_dim: int = 32,
        dropout: float = 0.1,
        enabled: bool = True,
        use_spatial_block: bool = True,
        use_cross_scale_consistency: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.enabled = enabled
        self.use_spatial_block = use_spatial_block
        self.use_cross_scale_consistency = use_cross_scale_consistency
        self.proj = nn.Sequential(
            nn.Linear(len(DIFFICULTY_STAT_NAMES) * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _condition_ready(stats: torch.Tensor) -> torch.Tensor:
        normalized = stats.clone()
        normalized[:, 5:7] = torch.log1p(normalized[:, 5:7].clamp_min(0.0))
        return normalized

    def forward(
        self,
        x_f: torch.Tensor,
        m_f: torch.Tensor,
        x_m: torch.Tensor,
        m_m: torch.Tensor,
        x_c: torch.Tensor,
        m_c: torch.Tensor,
        r_m: torch.Tensor | None = None,
        r_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ref_f = 0.5 * (
            F.interpolate(x_m, size=x_f.shape[-3:], mode="trilinear", align_corners=False)
            + F.interpolate(x_c, size=x_f.shape[-3:], mode="trilinear", align_corners=False)
        )
        ref_m = F.interpolate(x_c, size=x_m.shape[-3:], mode="trilinear", align_corners=False)
        ref_c = F.interpolate(x_m, size=x_c.shape[-3:], mode="trilinear", align_corners=False)
        raw_f = compute_raw_difficulty_stats(
            x_f, m_f, cross_scale_reference=ref_f,
            use_spatial_block=self.use_spatial_block,
            use_cross_scale_consistency=self.use_cross_scale_consistency,
        )
        raw_m = compute_raw_difficulty_stats(
            x_m, m_m, reliability=r_m, cross_scale_reference=ref_m,
            use_spatial_block=self.use_spatial_block,
            use_cross_scale_consistency=self.use_cross_scale_consistency,
        )
        raw_c = compute_raw_difficulty_stats(
            x_c, m_c, reliability=r_c, cross_scale_reference=ref_c,
            use_spatial_block=self.use_spatial_block,
            use_cross_scale_consistency=self.use_cross_scale_consistency,
        )
        if self.enabled:
            condition = self.proj(torch.cat([
                self._condition_ready(raw_f),
                self._condition_ready(raw_m),
                self._condition_ready(raw_c),
            ], dim=-1))
        else:
            condition = torch.zeros(
                x_f.shape[0], self.out_dim, device=x_f.device, dtype=x_f.dtype
            )
        return condition.to(dtype=x_f.dtype), {
            "raw_f": raw_f,
            "raw_m": raw_m,
            "raw_c": raw_c,
            "score_f": aggregate_difficulty_score(raw_f),
            "score_m": aggregate_difficulty_score(raw_m),
            "score_c": aggregate_difficulty_score(raw_c),
        }
