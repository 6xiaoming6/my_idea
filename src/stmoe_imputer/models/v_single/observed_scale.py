from __future__ import annotations

import torch


def masked_channel_rms(
    value: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Return detached observed-value RMS with shape ``[B, C, 1, 1, 1]``."""
    if value.ndim != 5 or mask.ndim != 5:
        raise ValueError(
            f"value and mask must be 5-D tensors, got {value.ndim}-D and {mask.ndim}-D"
        )
    if value.shape[0] != mask.shape[0] or value.shape[2:] != mask.shape[2:]:
        raise ValueError(
            "value and mask must have matching batch and spatiotemporal dimensions, "
            f"got {tuple(value.shape)} and {tuple(mask.shape)}"
        )
    if mask.shape[1] != 1:
        raise ValueError(
            f"mask channel dimension must be 1, got {mask.shape[1]}"
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    value_f = value.detach().float()
    mask_f = mask.detach().float().expand_as(value_f)
    dims = (2, 3, 4)
    count = mask_f.sum(dim=dims, keepdim=True)
    rms_observed = (
        (value_f.square() * mask_f).sum(dim=dims, keepdim=True)
        / count.clamp_min(1.0)
    ).sqrt()

    # ``value`` is the observed input. This fallback remains target-free and
    # keeps completely empty samples finite.
    rms_fallback = value_f.square().mean(dim=dims, keepdim=True).sqrt()
    rms = torch.where(count > 0, rms_observed, rms_fallback)
    return rms.clamp_min(float(eps)).to(dtype=value.dtype).detach()
