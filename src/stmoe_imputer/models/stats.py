from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_observation_stats(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute [missing_rate, observed_ratio, temporal_missing_score,
    spatial_missing_score, aggregation_reliability] for each sample.
    """
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask [B,1,T,H,W], got {tuple(mask.shape)}")
    b, _, t, h, w = mask.shape
    observed_ratio = mask.mean(dim=(1, 2, 3, 4)).view(b, 1)
    missing_rate = 1.0 - observed_ratio

    obs_per_t = mask.mean(dim=(1, 3, 4))
    temporal_missing_score = (obs_per_t < 0.1).float().mean(dim=1, keepdim=True)

    missing = 1.0 - mask
    missing_2d = missing.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    if h >= 4 and w >= 4:
        block = F.avg_pool2d(missing_2d, kernel_size=4, stride=1)
        spatial_missing_score = block.amax(dim=(1, 2, 3)).reshape(b, t).mean(dim=1, keepdim=True)
    else:
        spatial_missing_score = missing_rate

    aggregation_reliability = observed_ratio
    return torch.cat(
        [
            missing_rate,
            observed_ratio,
            temporal_missing_score,
            spatial_missing_score,
            aggregation_reliability,
        ],
        dim=1,
    )
