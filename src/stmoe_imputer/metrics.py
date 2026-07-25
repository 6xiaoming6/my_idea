from __future__ import annotations

import math

import torch

from .losses import expand_mask_as


@torch.no_grad()
def masked_metric_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return sufficient statistics for exact dataset-level masked metrics."""
    missing = expand_mask_as(1.0 - mask, pred)
    difference = pred - target
    absolute_error = difference.abs() * missing
    squared_error = difference.square() * missing
    absolute_percentage_error = (
        difference.abs() / target.abs().clamp_min(eps)
    ) * missing
    return {
        "absolute_error_sum": absolute_error.sum(dtype=torch.float64),
        "squared_error_sum": squared_error.sum(dtype=torch.float64),
        "absolute_percentage_error_sum": absolute_percentage_error.sum(
            dtype=torch.float64
        ),
        "count": missing.sum(dtype=torch.float64),
    }


@torch.no_grad()
def masked_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    sums = masked_metric_sums(pred, target, mask, eps=eps)
    denom = sums["count"].clamp_min(1.0)
    mae = sums["absolute_error_sum"] / denom
    rmse = torch.sqrt(
        (sums["squared_error_sum"] / denom).clamp_min(0.0)
    )
    mape = sums["absolute_percentage_error_sum"] / denom
    return {"mae": mae, "rmse": rmse, "mape": mape}


class MaskedMetricAccumulator:
    """Accumulate exact global metrics across unequal batch sizes/mask counts."""

    def __init__(self) -> None:
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.absolute_percentage_error_sum = 0.0
        self.count = 0.0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor]:
        sums = masked_metric_sums(pred, target, mask, eps=eps)
        self.absolute_error_sum += float(
            sums["absolute_error_sum"].detach().cpu()
        )
        self.squared_error_sum += float(
            sums["squared_error_sum"].detach().cpu()
        )
        self.absolute_percentage_error_sum += float(
            sums["absolute_percentage_error_sum"].detach().cpu()
        )
        self.count += float(sums["count"].detach().cpu())
        denominator = sums["count"].clamp_min(1.0)
        return {
            "mae": sums["absolute_error_sum"] / denominator,
            "rmse": torch.sqrt(
                (sums["squared_error_sum"] / denominator).clamp_min(0.0)
            ),
            "mape": sums["absolute_percentage_error_sum"] / denominator,
        }

    def compute(self) -> dict[str, float]:
        denominator = max(self.count, 1.0)
        return {
            "mae": self.absolute_error_sum / denominator,
            "rmse": math.sqrt(
                max(self.squared_error_sum / denominator, 0.0)
            ),
            "mape": self.absolute_percentage_error_sum / denominator,
        }


def tensor_dict_to_float(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}
