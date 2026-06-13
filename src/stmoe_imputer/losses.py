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
    scale_mode: str = "fine_mid_coarse",
) -> torch.Tensor:
    if scale_mode == "fine":
        return _empty_loss_like(x_hat_main)

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
    loss = observed_loss(x_hat_m, x_m_obs, m_m, loss_type)
    if scale_mode == "fine_mid":
        return loss
    if scale_mode != "fine_mid_coarse":
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    ones_m = torch.ones_like(m_m)
    mid_to_coarse = max(1, fine_to_coarse // fine_to_mid)
    x_hat_c, _ = masked_pool2d_spatial(
        x_hat_m, ones_m, kernel_size=mid_to_coarse, mode=pooling_mode
    )
    return loss + observed_loss(x_hat_c, x_c_obs, m_c, loss_type)


def _empty_loss_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def gate_balance_loss(gates: dict[str, torch.Tensor]) -> torch.Tensor:
    gate_all = torch.cat([gates["fine"], gates["mid"], gates["coarse"]], dim=0)
    usage = gate_all.mean(dim=0)
    target = torch.ones_like(usage) / gate_all.shape[1]
    return ((usage - target) ** 2).sum()


def moe_balance_loss(
    gates: dict[str, torch.Tensor],
    selected_masks: dict[str, torch.Tensor] | None,
    use_load_balance: bool = True,
    scale_names: tuple[str, ...] = ("fine", "mid", "coarse"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_all = torch.cat([gates[name] for name in scale_names], dim=0)
    num_experts = gate_all.shape[1]

    importance = gate_all.mean(dim=0)
    target_importance = torch.ones_like(importance) / num_experts
    l_importance = ((importance - target_importance) ** 2).sum()

    if not use_load_balance or selected_masks is None:
        l_load = _empty_loss_like(l_importance)
    else:
        mask_all = torch.cat([selected_masks[name] for name in scale_names], dim=0)
        load = mask_all.mean(dim=0)
        target_load = torch.ones_like(load) * load.mean().detach()
        l_load = ((load - target_load) ** 2).sum()

    return l_importance + l_load, l_importance, l_load


def fusion_entropy_loss(fusion_gate: torch.Tensor) -> torch.Tensor:
    entropy = -(fusion_gate * fusion_gate.clamp_min(1e-8).log()).sum(dim=1).mean()
    return entropy


def categorical_entropy_loss(gate: torch.Tensor) -> torch.Tensor:
    return -(gate * gate.clamp_min(1e-8).log()).sum(dim=1).mean()


def complementary_loss(h_shared: torch.Tensor, h_route: torch.Tensor) -> torch.Tensor:
    h_shared_norm = F.normalize(h_shared.flatten(2), dim=1)
    h_route_norm = F.normalize(h_route.flatten(2), dim=1)
    cos = (h_shared_norm * h_route_norm).sum(dim=1)
    return (cos ** 2).mean()


def compute_main_stage_loss(
    outputs: dict,
    batch: dict[str, torch.Tensor],
    cfg: dict,
    epoch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg["loss"]
    loss_type = loss_cfg.get("type", "smooth_l1")
    x_f_gt = batch["x_f_gt"]
    m_f = batch["m_f"]
    l_main = masked_loss(outputs["x_hat_main"], x_f_gt, m_f, loss_type=loss_type)
    l_final = masked_loss(outputs["x_hat_final"], x_f_gt, m_f, loss_type=loss_type)
    is_full = (
        cfg["model"]["main"].get("use_shared_branch", True)
        and cfg["model"]["main"].get("use_routed_branch", True)
    )
    scale_cfg = cfg["data"]["scales"]
    scale_mode = outputs.get(
        "scale_mode",
        cfg["model"]["main"].get("scale_mode", cfg["model"].get("scale_mode", "fine_mid_coarse")),
    )
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
        scale_mode=scale_mode,
    )
    routing_mode = outputs.get("routing_mode", "topk")
    if scale_mode == "fine":
        balance_scales = ("fine",)
    elif scale_mode == "fine_mid":
        balance_scales = ("fine", "mid")
    elif scale_mode == "fine_mid_coarse":
        balance_scales = ("fine", "mid", "coarse")
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    use_routed_branch = cfg["model"]["main"].get("use_routed_branch", True)
    use_router = cfg["model"]["main"].get("use_router", True)
    use_load_balance = use_routed_branch and use_router and routing_mode != "dense"
    l_balance, l_importance_balance, l_load_balance = moe_balance_loss(
        outputs["gates"],
        outputs.get("selected_masks"),
        use_load_balance=use_load_balance,
        scale_names=balance_scales,
    )
    if not use_routed_branch or (routing_mode == "dense" and not use_router):
        l_balance = _empty_loss_like(l_balance)
        l_importance_balance = _empty_loss_like(l_importance_balance)
        l_load_balance = _empty_loss_like(l_load_balance)

    l_fusion_entropy = _empty_loss_like(l_balance)
    if loss_cfg.get("lambda_fusion_entropy", 0.0) != 0:
        route_gate = outputs["gates"].get("route_fusion_32")
        if route_gate is not None:
            l_fusion_entropy = fusion_entropy_loss(route_gate)
    l_branch_entropy = _empty_loss_like(l_balance)
    if loss_cfg.get("lambda_branch_entropy", 0.0) != 0:
        branch_gate = outputs["gates"].get("branch_gate")
        if branch_gate is not None:
            l_branch_entropy = categorical_entropy_loss(branch_gate)

    l_shared_aux = _empty_loss_like(l_main)
    x_hat_shared = outputs.get("x_hat_shared")
    if is_full and x_hat_shared is not None and cfg["model"]["main"].get("enable_branch_aux", True):
        l_shared_aux = masked_loss(x_hat_shared, x_f_gt, m_f, loss_type=loss_type)

    l_route_aux = _empty_loss_like(l_main)
    x_hat_route = outputs.get("x_hat_route")
    if is_full and x_hat_route is not None and cfg["model"]["main"].get("enable_branch_aux", True):
        l_route_aux = masked_loss(x_hat_route, x_f_gt, m_f, loss_type=loss_type)

    l_complementary = _empty_loss_like(l_main)
    features = outputs.get("features", {})
    h_shared = features.get("h_shared") if isinstance(features, dict) else None
    h_route = features.get("h_route_proj") if isinstance(features, dict) else None
    if (
        is_full
        and cfg["model"]["main"].get("enable_complementary_loss", True)
        and h_shared is not None
        and h_route is not None
    ):
        l_complementary = complementary_loss(h_shared, h_route)

    warmup_epochs = max(1, cfg.get("train", {}).get("aux_loss_warmup_epochs", 1))
    warmup_factor = 1.0 if epoch is None else min(1.0, max(0.0, epoch / warmup_epochs))

    loss = l_main + loss_cfg.get("lambda_cross", 0.1) * warmup_factor * l_cross
    balance_weight = loss_cfg.get("lambda_balance", 0.01)
    importance_weight = loss_cfg.get("lambda_importance_balance", balance_weight)
    load_weight = loss_cfg.get("lambda_load_balance", balance_weight)
    loss = loss + importance_weight * warmup_factor * l_importance_balance
    loss = loss + load_weight * warmup_factor * l_load_balance
    loss = loss + loss_cfg.get("lambda_fusion_entropy", 0.0) * l_fusion_entropy
    loss = loss + loss_cfg.get("lambda_branch_entropy", 0.0) * l_branch_entropy
    loss = loss + loss_cfg.get("lambda_shared_aux", 0.0) * l_shared_aux
    loss = loss + loss_cfg.get("lambda_route_aux", 0.0) * l_route_aux
    loss = loss + loss_cfg.get("lambda_complementary", 0.0) * l_complementary
    if loss_cfg.get("lambda_final", 0.0) > 0:
        loss = loss + loss_cfg["lambda_final"] * l_final
    return loss, {
        "loss": loss.detach(),
        "l_final": l_final.detach(),
        "l_main": l_main.detach(),
        "l_cross": l_cross.detach(),
        "l_balance": l_balance.detach(),
        "l_importance_balance": l_importance_balance.detach(),
        "l_load_balance": l_load_balance.detach(),
        "l_fusion_entropy": l_fusion_entropy.detach(),
        "l_branch_entropy": l_branch_entropy.detach(),
        "l_shared_aux": l_shared_aux.detach(),
        "l_route_aux": l_route_aux.detach(),
        "l_complementary": l_complementary.detach(),
        "aux_loss_warmup": torch.as_tensor(warmup_factor, device=l_main.device).detach(),
    }
