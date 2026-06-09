from __future__ import annotations

import torch
import torch.nn.functional as F

from .data.transforms import masked_pool2d_spatial


def expand_mask_as(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mask.expand(target.shape[0], target.shape[1], *target.shape[2:])


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    missing = expand_mask_as(1.0 - mask, pred)
    denom = missing.sum().clamp_min(1.0)
    pred_m = pred * missing
    target_m = target * missing
    if loss_type == "l1":
        loss = F.l1_loss(pred_m, target_m, reduction="sum")
    elif loss_type == "mse":
        loss = F.mse_loss(pred_m, target_m, reduction="sum")
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred_m, target_m, reduction="sum")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return loss / denom


def observed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    observed = expand_mask_as(mask, pred)
    denom = observed.sum().clamp_min(1.0)
    if loss_type == "l1":
        loss = F.l1_loss(pred * observed, target * observed, reduction="sum")
    elif loss_type == "mse":
        loss = F.mse_loss(pred * observed, target * observed, reduction="sum")
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred * observed, target * observed, reduction="sum")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return loss / denom


def cross_scale_loss(
    x_hat_main: torch.Tensor,
    x_m_obs: torch.Tensor,
    m_m: torch.Tensor,
    x_c_obs: torch.Tensor,
    m_c: torch.Tensor,
    fine_to_mid: int = 2,
    fine_to_coarse: int = 4,
    pooling_mode: str = "avg",
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    ones_f = torch.ones(
        x_hat_main.shape[0],
        1,
        x_hat_main.shape[2],
        x_hat_main.shape[3],
        x_hat_main.shape[4],
        device=x_hat_main.device,
        dtype=x_hat_main.dtype,
    )
    x_hat_m, _ = masked_pool2d_spatial(
        x_hat_main, ones_f, kernel_size=fine_to_mid, mode=pooling_mode
    )
    ones_m = torch.ones_like(m_m)
    mid_to_coarse = max(1, fine_to_coarse // fine_to_mid)
    x_hat_c, _ = masked_pool2d_spatial(
        x_hat_m, ones_m, kernel_size=mid_to_coarse, mode=pooling_mode
    )
    return observed_loss(x_hat_m, x_m_obs, m_m, loss_type) + observed_loss(
        x_hat_c, x_c_obs, m_c, loss_type
    )


def gate_balance_loss(gates: dict[str, torch.Tensor]) -> torch.Tensor:
    gate_all = torch.cat([gates["fine"], gates["mid"], gates["coarse"]], dim=0)
    usage = gate_all.mean(dim=0)
    target = torch.ones_like(usage) / gate_all.shape[1]
    return ((usage - target) ** 2).sum()


def compute_main_stage_loss(
    outputs: dict,
    batch: dict[str, torch.Tensor],
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg["loss"]
    loss_type = loss_cfg.get("type", "smooth_l1")
    x_f_gt = batch["x_f_gt"]
    m_f = batch["m_f"]
    l_main = masked_loss(outputs["x_hat_main"], x_f_gt, m_f, loss_type=loss_type)
    l_final = masked_loss(outputs["x_hat_final"], x_f_gt, m_f, loss_type=loss_type)
    scale_cfg = cfg["data"]["scales"]
    l_cross = cross_scale_loss(
        outputs["x_hat_main"],
        batch["x_m_obs"],
        batch["m_m"],
        batch["x_c_obs"],
        batch["m_c"],
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        loss_type=loss_type,
    )
    l_balance = gate_balance_loss(outputs["gates"])
    loss = l_main + loss_cfg.get("lambda_cross", 0.1) * l_cross
    loss = loss + loss_cfg.get("lambda_balance", 0.01) * l_balance
    if loss_cfg.get("lambda_final", 0.0) > 0:
        loss = loss + loss_cfg["lambda_final"] * l_final
    return loss, {
        "loss": loss.detach(),
        "l_final": l_final.detach(),
        "l_main": l_main.detach(),
        "l_cross": l_cross.detach(),
        "l_balance": l_balance.detach(),
    }
