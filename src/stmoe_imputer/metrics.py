from __future__ import annotations

import torch

from .losses import expand_mask_as


@torch.no_grad()
def masked_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    missing = expand_mask_as(1.0 - mask, pred)
    denom = missing.sum().clamp_min(1.0)
    diff = (pred - target) * missing
    mae = diff.abs().sum() / denom
    rmse = torch.sqrt((diff.pow(2).sum() / denom).clamp_min(0.0))
    mape = ((diff.abs() / target.abs().clamp_min(eps)) * missing).sum() / denom
    return {"mae": mae, "rmse": rmse, "mape": mape}


def tensor_dict_to_float(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}
