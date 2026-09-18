from __future__ import annotations

import torch

from .losses import supervision_mask


class MaskedMetricAccumulator:
    """Accumulate exact dataset-level metrics over missing entries.

    Averaging per-batch MAE/RMSE gives every batch the same weight and is not
    exact when batch sizes or missing counts differ.  This accumulator keeps
    numerators and denominators and computes each metric once over the complete
    split.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.absolute_percentage_error = 0.0
        self.absolute_target = 0.0
        self.count = 0.0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> None:
        # Keep reductions in float32 on GPU (consumer GPUs have very low FP64
        # throughput), then accumulate batch totals in Python float64.
        selected = supervision_mask(target, mask, target_mask)
        diff_abs = (pred[selected].float() - target[selected].float()).abs()
        target_abs = target[selected].float().abs()
        self.absolute_error += float(diff_abs.sum().double().cpu())
        self.squared_error += float(diff_abs.square().sum().double().cpu())
        self.absolute_percentage_error += float(
            (diff_abs / target_abs.clamp_min(self.eps)).sum().double().cpu()
        )
        self.absolute_target += float(target_abs.sum().double().cpu())
        self.count += float(selected.sum().double().cpu())

    def compute(self) -> dict[str, float]:
        denom = max(self.count, 1.0)
        return {
            "mae": self.absolute_error / denom,
            "rmse": (self.squared_error / denom) ** 0.5,
            "mape": self.absolute_percentage_error / denom,
            "wape": self.absolute_error / max(self.absolute_target, self.eps),
            "metric_missing_count": self.count,
        }


@torch.no_grad()
def masked_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    target_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    selected = supervision_mask(target, mask, target_mask)
    denom = selected.sum().clamp_min(1.0)
    diff = pred[selected].float() - target[selected].float()
    mae = diff.abs().sum() / denom
    rmse = torch.sqrt((diff.pow(2).sum() / denom).clamp_min(0.0))
    mape = (diff.abs() / target[selected].float().abs().clamp_min(eps)).sum() / denom
    return {"mae": mae, "rmse": rmse, "mape": mape}


def tensor_dict_to_float(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}
