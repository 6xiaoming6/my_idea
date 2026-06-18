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


def ensure_observed(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "x_f_obs" not in sample:
        sample["x_f_obs"] = sample["x_f_gt"] * sample["m_f"]
    return sample


def ensure_multiscale(
    sample: dict[str, torch.Tensor],
    fine_to_mid: int = 2,
    fine_to_coarse: int = 4,
    pooling_mode: str = "avg",
) -> dict[str, torch.Tensor]:
    """Create mid/coarse scales from observed values, never from full ground truth."""
    sample = ensure_observed(sample)
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
