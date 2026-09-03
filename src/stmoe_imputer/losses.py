from __future__ import annotations

import math

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


def multi_resolution_supervision_loss(
    outputs: dict,
    x_f_gt: torch.Tensor,
    m_m: torch.Tensor,
    m_c: torch.Tensor,
    fine_to_mid: int,
    fine_to_coarse: int,
    pooling_mode: str,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_hat_mid = outputs.get("x_hat_mid")
    x_hat_coarse = outputs.get("x_hat_coarse")
    if x_hat_mid is None and x_hat_coarse is None:
        empty = _empty_loss_like(x_f_gt)
        return empty, empty
    ones_f = torch.ones(
        x_f_gt.shape[0], 1, *x_f_gt.shape[2:],
        device=x_f_gt.device, dtype=x_f_gt.dtype,
    )
    x_m_gt, m_m_gt = masked_pool2d_spatial(
        x_f_gt, ones_f, kernel_size=fine_to_mid, mode=pooling_mode
    )
    l_mid = (
        masked_loss(x_hat_mid, x_m_gt, m_m, loss_type)
        if x_hat_mid is not None
        else _empty_loss_like(x_f_gt)
    )
    if x_hat_coarse is None:
        return l_mid, _empty_loss_like(x_f_gt)
    mid_to_coarse = max(1, fine_to_coarse // fine_to_mid)
    x_c_gt, _ = masked_pool2d_spatial(
        x_m_gt, m_m_gt, kernel_size=mid_to_coarse, mode=pooling_mode
    )
    return l_mid, masked_loss(x_hat_coarse, x_c_gt, m_c, loss_type)


def _missing_absolute_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    missing = expand_mask_as(1.0 - mask, pred)
    error = (pred - target).abs() * missing
    denom = missing.sum().clamp_min(1.0)
    return error, denom


def v14_regret_loss(
    pred_final: torch.Tensor,
    pred_base: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    final_error, denom = _missing_absolute_error(pred_final, target, mask)
    base_error, _ = _missing_absolute_error(pred_base.detach(), target, mask)
    regret = F.relu(final_error - base_error).sum() / denom
    missing = expand_mask_as(1.0 - mask, pred_final)
    violation = ((final_error > base_error).to(pred_final.dtype) * missing).sum() / denom
    return regret, violation


def v14_sample_rmse_regret_loss(
    pred_final: torch.Tensor,
    pred_base: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize sample-level RMSE regressions relative to detached V14 base."""
    missing = expand_mask_as(1.0 - mask, pred_final)
    reduce_dims = tuple(range(1, pred_final.ndim))
    denom = missing.sum(dim=reduce_dims).clamp_min(1.0)
    final_mse = ((pred_final - target).square() * missing).sum(dim=reduce_dims) / denom
    base_mse = (
        ((pred_base.detach() - target).square() * missing).sum(dim=reduce_dims) / denom
    )
    # Adding eps before sqrt keeps the derivative finite when the zero-initialized
    # V14 residual makes final_mse and base_mse exactly zero/equal.
    final_rmse = (final_mse.clamp_min(0.0) + eps).sqrt()
    base_rmse = (base_mse.clamp_min(0.0) + eps).sqrt()
    regret = F.relu(final_rmse - base_rmse)
    return regret.mean(), (regret > 0).to(pred_final.dtype).mean()


def v14_delta_scale_loss(
    delta: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularize raw residual scale relative to each sample's observed RMS."""
    reduce_dims = tuple(range(1, delta.ndim))
    observed_mask = expand_mask_as(mask, observed)
    observed_count = observed_mask.sum(dim=reduce_dims).clamp_min(1.0)
    observed_rms = (
        (
            (observed.square() * observed_mask).sum(dim=reduce_dims)
            / observed_count
        ).clamp_min(0.0)
        + eps
    ).sqrt()
    # The residual head is intentionally zero-initialized.  sqrt(x + eps)
    # avoids an infinite derivative at its first optimization step.
    delta_rms = (delta.square().mean(dim=reduce_dims).clamp_min(0.0) + eps).sqrt()
    ratio = delta_rms / observed_rms.clamp_min(eps)
    return torch.log1p(ratio).square().mean(), ratio.mean()


def _empty_loss_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def v14_stage_aux_schedule_scale(
    cfg: dict,
    epoch: int | None,
) -> float:
    """Return the single-stage V14 mid/coarse supervision schedule scale.

    The default ``constant`` mode exactly preserves legacy V14.  ``cosine``
    keeps the original supervision during an optional initial fraction and
    then decays it smoothly to ``final_scale`` without freezing parameters or
    creating a second training stage.
    """
    schedule = cfg.get("loss", {}).get("v14_stage_aux_schedule", {})
    if schedule is None:
        schedule = {}
    if not isinstance(schedule, dict):
        raise ValueError("loss.v14_stage_aux_schedule must be an object")
    mode = str(schedule.get("mode", "constant"))
    if mode == "constant":
        return 1.0
    if mode != "cosine":
        raise ValueError(
            "loss.v14_stage_aux_schedule.mode must be 'constant' or 'cosine', "
            f"got {mode!r}"
        )
    final_scale = float(schedule.get("final_scale", 0.0))
    start_fraction = float(schedule.get("start_fraction", 0.0))
    if not 0.0 <= final_scale <= 1.0:
        raise ValueError("v14 stage auxiliary final_scale must be in [0,1]")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("v14 stage auxiliary start_fraction must be in [0,1)")
    max_epochs = max(1, int(cfg.get("train", {}).get("epochs", 1)))
    current_epoch = max_epochs if epoch is None else min(max(1, int(epoch)), max_epochs)
    if max_epochs == 1:
        progress = 0.0
    else:
        progress = (current_epoch - 1) / (max_epochs - 1)
    if progress <= start_fraction:
        return 1.0
    decay_progress = (progress - start_fraction) / (1.0 - start_fraction)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return final_scale + (1.0 - final_scale) * cosine


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
    load_balance_mode: str = "legacy_hard",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute soft-importance and optional hard-load-aware routing losses.

    ``legacy_hard`` exactly preserves the original V14 implementation.  Its
    hard Top-K variance is useful as a diagnostic, but the discrete mask has no
    gradient to the router.  ``switch_topk`` uses the detached, normalized
    hard load to weight differentiable mean gate probabilities.  This gives
    overloaded experts a training signal without changing Top-K inference.
    """
    if load_balance_mode not in {"legacy_hard", "switch_topk"}:
        raise ValueError(
            "loss.load_balance_mode must be 'legacy_hard' or 'switch_topk', "
            f"got {load_balance_mode!r}"
        )
    if not scale_names:
        raise ValueError("scale_names must contain at least one active scale")

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
        if load_balance_mode == "legacy_hard":
            target_load = torch.ones_like(load) * load.mean().detach()
            l_load = ((load - target_load) ** 2).sum()
        else:
            load_distribution = (
                load.detach() / load.detach().sum().clamp_min(eps)
            )
            l_load = num_experts * (importance * load_distribution).sum()

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
    v14_rmse_regret_weight = float(loss_cfg.get("lambda_v14_rmse_regret", 0.0))
    v14_delta_scale_weight = float(loss_cfg.get("lambda_v14_delta_scale", 0.0))
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
    x_hat_base = outputs.get("x_hat_base")
    x_hat_ctf = outputs.get("x_hat_ctf")
    if x_hat_base is not None:
        l_v14_base = masked_loss(x_hat_base, x_f_gt, m_f, loss_type=loss_type)
        l_v14_mid, l_v14_coarse = multi_resolution_supervision_loss(
            outputs,
            x_f_gt,
            batch["m_m"],
            batch["m_c"],
            fine_to_mid=scale_cfg["fine_to_mid"],
            fine_to_coarse=scale_cfg["fine_to_coarse"],
            pooling_mode=scale_cfg.get("pooling_mode", "avg"),
            loss_type=loss_type,
        )
        l_v14_regret, v14_violation_rate = v14_regret_loss(
            outputs["x_hat_main"], x_hat_base, x_f_gt, m_f
        )
        v14_diagnostics = outputs.get("diagnostics", {}).get("v14", {})
        alpha_final = v14_diagnostics.get("alpha_final")
        l_v14_gate = alpha_final.mean() if torch.is_tensor(alpha_final) else _empty_loss_like(l_main)
        if v14_rmse_regret_weight != 0.0:
            l_v14_rmse_regret, v14_rmse_violation_rate = (
                v14_sample_rmse_regret_loss(
                    outputs["x_hat_main"], x_hat_base, x_f_gt, m_f
                )
            )
        else:
            l_v14_rmse_regret = _empty_loss_like(l_main)
            v14_rmse_violation_rate = _empty_loss_like(l_main)
        delta_ctf = outputs.get("features", {}).get("delta_ctf")
        if v14_delta_scale_weight != 0.0 and torch.is_tensor(delta_ctf):
            l_v14_delta_scale, v14_delta_scale_ratio = v14_delta_scale_loss(
                delta_ctf, batch["x_f_obs"], m_f
            )
        else:
            l_v14_delta_scale = _empty_loss_like(l_main)
            v14_delta_scale_ratio = _empty_loss_like(l_main)
        base_error, base_denom = _missing_absolute_error(x_hat_base, x_f_gt, m_f)
        v14_base_hidden_mae = base_error.sum() / base_denom
        final_error, final_denom = _missing_absolute_error(outputs["x_hat_main"], x_f_gt, m_f)
        v14_final_hidden_mae = final_error.sum() / final_denom
        if x_hat_ctf is not None:
            ctf_error, ctf_denom = _missing_absolute_error(x_hat_ctf, x_f_gt, m_f)
            v14_ctf_hidden_mae = ctf_error.sum() / ctf_denom
        else:
            v14_ctf_hidden_mae = _empty_loss_like(l_main)
    else:
        l_v14_base = _empty_loss_like(l_main)
        l_v14_mid = _empty_loss_like(l_main)
        l_v14_coarse = _empty_loss_like(l_main)
        l_v14_regret = _empty_loss_like(l_main)
        l_v14_gate = _empty_loss_like(l_main)
        l_v14_rmse_regret = _empty_loss_like(l_main)
        l_v14_delta_scale = _empty_loss_like(l_main)
        v14_violation_rate = _empty_loss_like(l_main)
        v14_rmse_violation_rate = _empty_loss_like(l_main)
        v14_delta_scale_ratio = _empty_loss_like(l_main)
        v14_base_hidden_mae = _empty_loss_like(l_main)
        v14_ctf_hidden_mae = _empty_loss_like(l_main)
        v14_final_hidden_mae = _empty_loss_like(l_main)
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
    load_balance_mode = str(loss_cfg.get("load_balance_mode", "legacy_hard"))
    l_balance, l_importance_balance, l_load_balance = moe_balance_loss(
        outputs["gates"],
        outputs.get("selected_masks"),
        use_load_balance=use_load_balance,
        scale_names=balance_scales,
        load_balance_mode=load_balance_mode,
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
    v14_cfg = cfg.get("model", {}).get("v14", {})
    loss = loss + loss_cfg.get("lambda_v14_base", v14_cfg.get("lambda_base", 0.0)) * l_v14_base
    v14_stage_aux_scale = v14_stage_aux_schedule_scale(cfg, epoch)
    v14_mid_weight = loss_cfg.get(
        "lambda_v14_mid", v14_cfg.get("lambda_mid", 0.0)
    )
    v14_coarse_weight = loss_cfg.get(
        "lambda_v14_coarse", v14_cfg.get("lambda_coarse", 0.0)
    )
    loss = loss + v14_stage_aux_scale * (
        v14_mid_weight * l_v14_mid + v14_coarse_weight * l_v14_coarse
    )
    loss = loss + loss_cfg.get("lambda_v14_regret", v14_cfg.get("lambda_regret", 0.0)) * l_v14_regret
    loss = loss + loss_cfg.get("lambda_v14_gate", v14_cfg.get("lambda_gate", 0.0)) * l_v14_gate
    loss = loss + v14_rmse_regret_weight * l_v14_rmse_regret
    loss = loss + v14_delta_scale_weight * l_v14_delta_scale
    v20_probe = outputs.get("v20_probe")
    if outputs.get("v20_enabled", False) and isinstance(v20_probe, dict):
        l_v20_probe = v20_probe.get("probe_loss")
        if not torch.is_tensor(l_v20_probe):
            l_v20_probe = _empty_loss_like(l_main)
        loss = loss + float(loss_cfg.get("lambda_v20_probe", 0.05)) * l_v20_probe
    else:
        l_v20_probe = _empty_loss_like(l_main)
    if loss_cfg.get("lambda_final", 0.0) > 0:
        loss = loss + loss_cfg["lambda_final"] * l_final
    loss_logs = {
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
    if x_hat_base is not None:
        loss_logs.update({
            "l_v14_base": l_v14_base.detach(),
            "l_v14_mid": l_v14_mid.detach(),
            "l_v14_coarse": l_v14_coarse.detach(),
            "l_v14_regret": l_v14_regret.detach(),
            "l_v14_gate": l_v14_gate.detach(),
            "l_v14_rmse_regret": l_v14_rmse_regret.detach(),
            "l_v14_delta_scale": l_v14_delta_scale.detach(),
            "v14_base_hidden_mae": v14_base_hidden_mae.detach(),
            "v14_ctf_hidden_mae": v14_ctf_hidden_mae.detach(),
            "v14_final_hidden_mae": v14_final_hidden_mae.detach(),
            "v14_non_regression_violation_rate": v14_violation_rate.detach(),
            "v14_rmse_non_regression_violation_rate": v14_rmse_violation_rate.detach(),
            "v14_delta_scale_ratio": v14_delta_scale_ratio.detach(),
            "v14_stage_aux_scale": torch.as_tensor(
                v14_stage_aux_scale, device=l_main.device
            ).detach(),
        })
    if outputs.get("v20_enabled", False):
        loss_logs["l_v20_probe"] = l_v20_probe.detach()
    return loss, loss_logs
