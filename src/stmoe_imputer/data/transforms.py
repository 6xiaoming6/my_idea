from __future__ import annotations

import torch
import torch.nn.functional as F


def to_bcthw(x: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    """Convert [B,C,T,H,W] or [B,T,H,W,C] to float [B,C,T,H,W]."""
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")
    if channels is not None and x.shape[1] == channels:
        return x.float().contiguous()
    if channels is not None and x.shape[-1] == channels:
        return x.permute(0, 4, 1, 2, 3).float().contiguous()
    if x.shape[1] <= 8 and (x.shape[-1] > 8 or x.shape[1] <= x.shape[-1]):
        return x.float().contiguous()
    if x.shape[-1] <= 8:
        return x.permute(0, 4, 1, 2, 3).float().contiguous()
    raise ValueError(
        "Cannot infer tensor layout. Expected [B,C,T,H,W] or [B,T,H,W,C], "
        f"got {tuple(x.shape)}."
    )


def masked_pool2d_spatial(
    x: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int = 2,
    mode: str = "avg",
    eps: float = 1e-6,
    return_reliability: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spatial masked pooling for [B,C,T,H,W] values and [B,1,T,H,W] masks.

    The returned value is built only from observed cells. ``mode="avg"`` keeps
    scales numerically close; ``mode="sum"`` is useful for later ablations.
    """
    squeeze_batch = False
    if x.ndim == 4 and mask.ndim == 4:
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze_batch = True
    if x.ndim != 5 or mask.ndim != 5:
        raise ValueError(f"Expected 5D tensors, got x={tuple(x.shape)}, mask={tuple(mask.shape)}")
    if mask.shape[1] != 1:
        raise ValueError(f"Mask channel must be 1, got {mask.shape[1]}")
    if x.shape[0] != mask.shape[0] or x.shape[2:] != mask.shape[2:]:
        raise ValueError(f"Shape mismatch: x={tuple(x.shape)}, mask={tuple(mask.shape)}")
    if kernel_size <= 1:
        x_same, mask_same = x * mask, mask
        rel_same = mask.to(dtype=x.dtype)
        if squeeze_batch:
            x_same = x_same.squeeze(0)
            mask_same = mask_same.squeeze(0)
            rel_same = rel_same.squeeze(0)
        if return_reliability:
            return x_same, mask_same, rel_same
        return x_same, mask_same

    b, c, t, h, w = x.shape
    if h % kernel_size != 0 or w % kernel_size != 0:
        raise ValueError(f"H/W must be divisible by kernel_size={kernel_size}, got H={h}, W={w}")

    x_2d = (x * mask).permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    m_2d = mask.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    area = float(kernel_size * kernel_size)
    x_sum = F.avg_pool2d(x_2d, kernel_size=kernel_size, stride=kernel_size) * area
    m_sum = F.avg_pool2d(m_2d, kernel_size=kernel_size, stride=kernel_size) * area

    if mode == "avg":
        x_down = x_sum / (m_sum + eps)
    elif mode == "sum":
        x_down = x_sum
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")
    m_down = (m_sum > 0).to(mask.dtype)
    r_down = (m_sum / area).clamp(0.0, 1.0).to(dtype=x.dtype)
    x_down = x_down * m_down

    h2, w2 = x_down.shape[-2:]
    x_down = x_down.reshape(b, t, c, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
    m_down = m_down.reshape(b, t, 1, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
    r_down = r_down.reshape(b, t, 1, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
    if squeeze_batch:
        x_down = x_down.squeeze(0)
        m_down = m_down.squeeze(0)
        r_down = r_down.squeeze(0)
    if return_reliability:
        return x_down, m_down, r_down
    return x_down, m_down


def observation_moment_pool2d_spatial(
    x: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int = 2,
    mode: str = "avg",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a path-consistent content/evidence state from fine observations.

    The seven evidence channels are bounded, dimensionless low-order moments:

    ``coverage, relative_value_variance, centroid_y, centroid_x,``
    ``spread_y, spread_x, covariance_yx``.

    Unlike repeated masked means, the content mean is always computed from the
    fine-level observed sum and count.  The evidence is not claimed to preserve
    the complete mask distribution; it deliberately retains only interpretable
    low-order observation moments.
    """
    squeeze_batch = False
    if x.ndim == 4 and mask.ndim == 4:
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze_batch = True
    if x.ndim != 5 or mask.ndim != 5:
        raise ValueError(f"Expected 5D tensors, got x={tuple(x.shape)}, mask={tuple(mask.shape)}")
    if mask.shape[1] != 1:
        raise ValueError(f"Mask channel must be 1, got {mask.shape[1]}")
    if x.shape[0] != mask.shape[0] or x.shape[2:] != mask.shape[2:]:
        raise ValueError(f"Shape mismatch: x={tuple(x.shape)}, mask={tuple(mask.shape)}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")

    b, c, t, h, w = x.shape
    if h % kernel_size != 0 or w % kernel_size != 0:
        raise ValueError(f"H/W must be divisible by kernel_size={kernel_size}, got H={h}, W={w}")

    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    m_2d = mask.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w).to(dtype=x.dtype)
    area = float(kernel_size * kernel_size)

    def block_sum(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(value, kernel_size=kernel_size, stride=kernel_size) * area

    count = block_sum(m_2d)
    observed_sum = block_sum(x_2d * m_2d)
    observed_square_sum = block_sum(x_2d.square() * m_2d)
    safe_count = count.clamp_min(eps)
    mean = observed_sum / safe_count
    second_moment = observed_square_sum / safe_count
    variance = (second_moment - mean.square()).clamp_min(0.0)

    if mode == "avg":
        content = mean
    elif mode == "sum":
        content = observed_sum
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")

    # Coordinates are local to every pooling cell.  Their direct construction
    # avoids the coordinate-frame error caused by naively adding local moments
    # from successive hierarchy levels.
    coordinate = torch.linspace(-1.0, 1.0, kernel_size, device=x.device, dtype=x.dtype)
    local_y, local_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    repeat_h = h // kernel_size
    repeat_w = w // kernel_size
    grid_y = local_y.repeat(repeat_h, repeat_w).view(1, 1, h, w)
    grid_x = local_x.repeat(repeat_h, repeat_w).view(1, 1, h, w)

    centroid_y = block_sum(m_2d * grid_y) / safe_count
    centroid_x = block_sum(m_2d * grid_x) / safe_count
    raw_y2 = block_sum(m_2d * grid_y.square()) / safe_count
    raw_x2 = block_sum(m_2d * grid_x.square()) / safe_count
    raw_yx = block_sum(m_2d * grid_y * grid_x) / safe_count
    spread_y = (raw_y2 - centroid_y.square()).clamp(0.0, 1.0)
    spread_x = (raw_x2 - centroid_x.square()).clamp(0.0, 1.0)
    covariance_yx = (raw_yx - centroid_y * centroid_x).clamp(-1.0, 1.0)

    mean_second = second_moment.mean(dim=1, keepdim=True)
    relative_variance = (
        variance.mean(dim=1, keepdim=True) / mean_second.clamp_min(eps)
    ).clamp(0.0, 1.0)
    coverage = (count / area).clamp(0.0, 1.0)
    valid = (count > 0).to(dtype=x.dtype)
    evidence = torch.cat(
        (
            coverage,
            relative_variance,
            centroid_y,
            centroid_x,
            spread_y,
            spread_x,
            covariance_yx,
        ),
        dim=1,
    )
    evidence = evidence * valid
    content = content * valid

    h2, w2 = content.shape[-2:]

    def restore(value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[1]
        restored = value.reshape(b, t, channels, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
        return restored.squeeze(0) if squeeze_batch else restored

    return restore(content), restore(valid), restore(coverage), restore(evidence)


def ensure_observed(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "x_f_obs" not in sample:
        sample["x_f_obs"] = sample["x_f_gt"] * sample["m_f"]
    return sample


def ensure_multiscale(
    sample: dict[str, torch.Tensor],
    fine_to_mid: int = 2,
    fine_to_coarse: int = 4,
    pooling_mode: str = "avg",
    pyramid_mode: str = "legacy",
) -> dict[str, torch.Tensor]:
    """Create mid/coarse scales from observed values, never from full ground truth."""
    sample = ensure_observed(sample)
    if pyramid_mode not in {
        "legacy",
        "observation_moment",
        "dual_observation_moment",
    }:
        raise ValueError(
            "pyramid_mode must be 'legacy', 'observation_moment', or "
            "'dual_observation_moment', "
            f"got {pyramid_mode!r}"
        )
    if pyramid_mode == "dual_observation_moment":
        # Keep the exact V14 hierarchical path as the safe anchor.  The
        # measure path is returned under separate keys so a downstream model
        # can decide how much of its correction to accept instead of replacing
        # the legacy representation unconditionally.
        legacy_mid, legacy_mid_mask, legacy_mid_reliability = masked_pool2d_spatial(
            sample["x_f_obs"],
            sample["m_f"],
            kernel_size=fine_to_mid,
            mode=pooling_mode,
            return_reliability=True,
        )
        ratio = max(1, fine_to_coarse // fine_to_mid)
        legacy_coarse, legacy_coarse_mask, legacy_coarse_reliability = (
            masked_pool2d_spatial(
                legacy_mid,
                legacy_mid_mask,
                kernel_size=ratio,
                mode=pooling_mode,
                return_reliability=True,
            )
        )
        measure_mid, measure_mid_mask, measure_mid_reliability, mid_evidence = (
            observation_moment_pool2d_spatial(
                sample["x_f_obs"],
                sample["m_f"],
                kernel_size=fine_to_mid,
                mode=pooling_mode,
            )
        )
        (
            measure_coarse,
            measure_coarse_mask,
            measure_coarse_reliability,
            coarse_evidence,
        ) = observation_moment_pool2d_spatial(
            sample["x_f_obs"],
            sample["m_f"],
            kernel_size=fine_to_coarse,
            mode=pooling_mode,
        )
        sample.update({
            "x_m_obs": legacy_mid,
            "m_m": legacy_mid_mask,
            "r_m": legacy_mid_reliability,
            "x_c_obs": legacy_coarse,
            "m_c": legacy_coarse_mask,
            "r_c": legacy_coarse_reliability,
            "x_m_measure": measure_mid,
            "m_m_measure": measure_mid_mask,
            "r_m_measure": measure_mid_reliability,
            "x_c_measure": measure_coarse,
            "m_c_measure": measure_coarse_mask,
            "r_c_measure": measure_coarse_reliability,
            "e_m": mid_evidence,
            "e_c": coarse_evidence,
        })
        zeros = torch.zeros_like(sample["m_f"], dtype=sample["x_f_obs"].dtype)
        sample["e_f"] = torch.cat(
            (sample["m_f"].to(dtype=zeros.dtype), *(zeros for _ in range(6))), dim=-4
        )
        return sample
    if pyramid_mode == "observation_moment":
        sample["x_m_obs"], sample["m_m"], sample["r_m"], sample["e_m"] = (
            observation_moment_pool2d_spatial(
                sample["x_f_obs"],
                sample["m_f"],
                kernel_size=fine_to_mid,
                mode=pooling_mode,
            )
        )
        sample["x_c_obs"], sample["m_c"], sample["r_c"], sample["e_c"] = (
            observation_moment_pool2d_spatial(
                sample["x_f_obs"],
                sample["m_f"],
                kernel_size=fine_to_coarse,
                mode=pooling_mode,
            )
        )
        # Fine evidence uses the same channel contract.  At unit scale there is
        # no within-cell geometry or variance, only observed support.
        zeros = torch.zeros_like(sample["m_f"], dtype=sample["x_f_obs"].dtype)
        sample["e_f"] = torch.cat(
            (sample["m_f"].to(dtype=zeros.dtype), *(zeros for _ in range(6))), dim=-4
        )
        return sample
    if "x_m_obs" not in sample or "m_m" not in sample:
        sample["x_m_obs"], sample["m_m"], sample["r_m"] = masked_pool2d_spatial(
            sample["x_f_obs"],
            sample["m_f"],
            kernel_size=fine_to_mid,
            mode=pooling_mode,
            return_reliability=True,
        )
    elif "r_m" not in sample:
        sample["r_m"] = sample["m_m"].float()
    if "x_c_obs" not in sample or "m_c" not in sample:
        ratio = max(1, fine_to_coarse // fine_to_mid)
        sample["x_c_obs"], sample["m_c"], sample["r_c"] = masked_pool2d_spatial(
            sample["x_m_obs"],
            sample["m_m"],
            kernel_size=ratio,
            mode=pooling_mode,
            return_reliability=True,
        )
    elif "r_c" not in sample:
        sample["r_c"] = sample["m_c"].float()
    return sample
