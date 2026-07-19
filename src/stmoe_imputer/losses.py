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


def masked_mean_per_sample(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average ``value`` per sample over positions selected by an explicit mask."""
    expanded = expand_mask_as(mask, value).to(dtype=value.dtype)
    numerator = (value * expanded).flatten(1).sum(dim=1)
    denominator = expanded.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


@torch.no_grad()
def oracle_alpha_grid(
    x_base: torch.Tensor,
    delta_candidate: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    alpha_grid: list[float] | tuple[float, ...] | torch.Tensor,
) -> torch.Tensor:
    """Select the per-sample residual scale with lowest hidden Smooth-L1 loss."""
    if not torch.is_tensor(alpha_grid):
        values = tuple(float(value) for value in alpha_grid)
        if len(values) < 2 or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("oracle alpha grid must contain at least two values in [0,1]")
        alpha_grid = values
    grid = torch.as_tensor(
        alpha_grid,
        device=x_base.device,
        dtype=x_base.dtype,
    ).flatten()
    if grid.numel() < 2:
        raise ValueError("oracle alpha grid must contain at least two values")
    missing = expand_mask_as(1.0 - observed_mask, x_base).to(dtype=x_base.dtype)
    denominator = missing.flatten(1).sum(dim=1).clamp_min(1.0)
    scores = []
    for alpha in grid:
        prediction = x_base.detach() + alpha * delta_candidate.detach()
        error = F.smooth_l1_loss(
            prediction,
            target.detach(),
            reduction="none",
        )
        scores.append((error * missing).flatten(1).sum(dim=1) / denominator)
    return grid[torch.stack(scores, dim=1).argmin(dim=1)]


def _pearson(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    value = value.float().flatten()
    target = target.float().flatten()
    value = value - value.mean()
    target = target - target.mean()
    denominator = value.square().sum().sqrt() * target.square().sum().sqrt()
    if value.numel() < 2:
        return value.sum() * 0.0
    correlation = (value * target).sum() / denominator.clamp_min(1e-12)
    return torch.where(denominator > 1e-12, correlation, correlation * 0.0)


def _rank(value: torch.Tensor) -> torch.Tensor:
    value = value.float().flatten()
    # Spearman uses average ranks for ties; alpha_star intentionally contains
    # many grid-value ties, so ordinal ranks would give a misleading correlation.
    _, inverse, counts = torch.unique(
        value,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    counts_f = counts.to(dtype=torch.float32)
    average_rank = counts_f.cumsum(dim=0) - (counts_f + 1.0) / 2.0
    return average_rank[inverse]


def _spearman(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _pearson(_rank(value), _rank(target))


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
    teacher_outputs: dict | None = None,
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
    x_hat_base = outputs.get("x_hat_base")
    x_hat_ctf = outputs.get("x_hat_ctf")
    is_v14 = bool(outputs.get("v14_enabled", False))
    if is_v14 and x_hat_base is not None:
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
        v14_violation_rate = _empty_loss_like(l_main)
        v14_base_hidden_mae = _empty_loss_like(l_main)
        v14_ctf_hidden_mae = _empty_loss_like(l_main)
        v14_final_hidden_mae = _empty_loss_like(l_main)

    effective_delta = outputs.get("delta_effective")
    is_v15 = bool(outputs.get("v15_enabled", False))
    l_v15_base = _empty_loss_like(l_main)
    l_v15_delta = _empty_loss_like(l_main)
    l_v15_safe = _empty_loss_like(l_main)
    v15_base_hidden_mae = _empty_loss_like(l_main)
    v15_final_hidden_mae = _empty_loss_like(l_main)
    v15_final_vs_base_improvement = _empty_loss_like(l_main)
    v15_sample_violation_rate = _empty_loss_like(l_main)
    if is_v15 and x_hat_base is not None and effective_delta is not None:
        l_v15_base = masked_loss(x_hat_base, x_f_gt, m_f, loss_type=loss_type)
        target_delta = x_f_gt - x_hat_base.detach()
        l_v15_delta = masked_loss(
            effective_delta, target_delta, m_f, loss_type=loss_type
        )

        missing = expand_mask_as(1.0 - m_f, x_hat_base)
        base_per_sample = masked_mean_per_sample(
            (x_hat_base.detach() - x_f_gt).abs(), missing
        )
        final_per_sample = masked_mean_per_sample(
            (outputs["x_hat_main"] - x_f_gt).abs(), missing
        )
        sample_regret = F.relu(final_per_sample - base_per_sample)
        l_v15_safe = sample_regret.mean()
        v15_base_hidden_mae = base_per_sample.mean()
        v15_final_hidden_mae = final_per_sample.mean()
        v15_final_vs_base_improvement = (
            v15_base_hidden_mae - v15_final_hidden_mae
        )
        v15_sample_violation_rate = (
            final_per_sample > base_per_sample
        ).to(dtype=l_main.dtype).mean()

    x_hat_candidate = outputs.get("x_hat_candidate")
    accept_gate = outputs.get("accept_gate")
    accept_logit = outputs.get("accept_logit")
    is_v15_1 = bool(outputs.get("v15_1_enabled", False))
    l_v15_1_base = _empty_loss_like(l_main)
    l_v15_1_candidate = _empty_loss_like(l_main)
    l_v15_1_accept = _empty_loss_like(l_main)
    l_v15_1_safe = _empty_loss_like(l_main)
    v15_1_base_hidden_mae = _empty_loss_like(l_main)
    v15_1_candidate_hidden_mae = _empty_loss_like(l_main)
    v15_1_final_hidden_mae = _empty_loss_like(l_main)
    v15_1_candidate_vs_base_gain = _empty_loss_like(l_main)
    v15_1_final_vs_base_gain = _empty_loss_like(l_main)
    v15_1_candidate_violation_rate = _empty_loss_like(l_main)
    v15_1_final_violation_rate = _empty_loss_like(l_main)
    v15_1_accept_target_mean = _empty_loss_like(l_main)
    v15_1_accept_positive_rate = _empty_loss_like(l_main)
    v15_1_accept_negative_rate = _empty_loss_like(l_main)
    v15_1_accept_uncertain_rate = _empty_loss_like(l_main)
    v15_1_accept_accuracy = _empty_loss_like(l_main)
    if (
        is_v15_1
        and x_hat_base is not None
        and x_hat_candidate is not None
        and accept_gate is not None
        and accept_logit is not None
    ):
        l_v15_1_base = masked_loss(x_hat_base, x_f_gt, m_f, loss_type=loss_type)
        l_v15_1_candidate = masked_loss(
            x_hat_candidate, x_f_gt, m_f, loss_type=loss_type
        )

        missing = expand_mask_as(1.0 - m_f, x_hat_base)
        base_per_sample = masked_mean_per_sample(
            (x_hat_base.detach() - x_f_gt).abs(), missing
        )
        candidate_per_sample = masked_mean_per_sample(
            (x_hat_candidate.detach() - x_f_gt).abs(), missing
        )
        final_per_sample = masked_mean_per_sample(
            (outputs["x_hat_main"] - x_f_gt).abs(), missing
        )
        relative_gain = (
            base_per_sample - candidate_per_sample
        ) / base_per_sample.clamp_min(1e-6)
        gain_margin = float(
            cfg.get("model", {}).get("v15_1", {}).get("accept_gain_margin", 0.002)
        )
        positive = relative_gain > gain_margin
        negative = relative_gain < -gain_margin
        uncertain = ~(positive | negative)
        target_accept = torch.full_like(relative_gain, 0.5)
        target_accept = torch.where(
            positive, torch.full_like(target_accept, 0.95), target_accept
        )
        target_accept = torch.where(
            negative, torch.full_like(target_accept, 0.05), target_accept
        )

        accept_prediction = accept_gate.flatten(1).mean(dim=1).clamp(1e-6, 1.0 - 1e-6)
        accept_logit_per_sample = accept_logit.flatten(1).mean(dim=1)
        l_v15_1_accept = F.binary_cross_entropy_with_logits(
            accept_logit_per_sample, target_accept.detach()
        )
        sample_regret = F.relu(final_per_sample - base_per_sample)
        l_v15_1_safe = sample_regret.mean()

        v15_1_base_hidden_mae = base_per_sample.mean()
        v15_1_candidate_hidden_mae = candidate_per_sample.mean()
        v15_1_final_hidden_mae = final_per_sample.mean()
        v15_1_candidate_vs_base_gain = (
            v15_1_base_hidden_mae - v15_1_candidate_hidden_mae
        )
        v15_1_final_vs_base_gain = (
            v15_1_base_hidden_mae - v15_1_final_hidden_mae
        )
        v15_1_candidate_violation_rate = (
            candidate_per_sample > base_per_sample
        ).to(dtype=l_main.dtype).mean()
        v15_1_final_violation_rate = (
            final_per_sample > base_per_sample
        ).to(dtype=l_main.dtype).mean()
        v15_1_accept_target_mean = target_accept.mean()
        v15_1_accept_positive_rate = positive.to(dtype=l_main.dtype).mean()
        v15_1_accept_negative_rate = negative.to(dtype=l_main.dtype).mean()
        v15_1_accept_uncertain_rate = uncertain.to(dtype=l_main.dtype).mean()
        decided = positive | negative
        predicted_positive = accept_prediction >= 0.5
        correct = (predicted_positive & positive) | (~predicted_positive & negative)
        v15_1_accept_accuracy = (
            correct.to(dtype=l_main.dtype).sum()
            / decided.to(dtype=l_main.dtype).sum().clamp_min(1.0)
        )
    is_v16 = bool(outputs.get("v16_enabled", False))
    v16_cfg = cfg.get("model", {}).get("v16", {})
    empty_v16 = _empty_loss_like(l_main)
    l_v16_base_gt = empty_v16
    l_v16_base_teacher = empty_v16
    l_v16_anchor = empty_v16
    l_v16_candidate = empty_v16
    l_v16_calibration = empty_v16
    l_v16_safe = empty_v16
    v16_logs: dict[str, torch.Tensor] = {}
    if is_v16:
        v16_base = outputs["x_hat_base"]
        v16_candidate = outputs["x_hat_candidate"]
        v16_final = outputs["x_hat_main"]
        delta_candidate = outputs["delta_candidate"]
        alpha_pred = outputs["residual_alpha"].flatten(1).mean(dim=1)
        l_v16_base_gt = masked_loss(
            v16_base, x_f_gt, m_f, loss_type=loss_type
        )
        teacher_prediction = None
        if teacher_outputs is not None:
            teacher_prediction = teacher_outputs["x_hat_main"].detach()
            l_v16_base_teacher = masked_loss(
                v16_base,
                teacher_prediction,
                m_f,
                loss_type=loss_type,
            )
        teacher_inside = float(v16_cfg.get("lambda_teacher_inside", 0.5))
        l_v16_anchor = l_v16_base_gt + teacher_inside * l_v16_base_teacher
        l_v16_candidate = masked_loss(
            v16_candidate, x_f_gt, m_f, loss_type=loss_type
        )

        alpha_grid = v16_cfg.get(
            "oracle_alpha_grid",
            (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
        )
        alpha_star = oracle_alpha_grid(
            x_base=v16_base,
            delta_candidate=delta_candidate,
            target=x_f_gt,
            observed_mask=m_f,
            alpha_grid=alpha_grid,
        )
        supervision = outputs.get(
            "calibration_supervision",
            v16_cfg.get("calibration_supervision", "oracle"),
        )
        calibration_mode = outputs.get(
            "calibration_mode", v16_cfg.get("calibration_mode", "learned")
        )
        if calibration_mode == "fixed_one":
            l_v16_calibration = _empty_loss_like(l_main)
        elif supervision == "binary":
            missing = expand_mask_as(1.0 - m_f, v16_base)
            base_for_binary = masked_mean_per_sample(
                (v16_base.detach() - x_f_gt).abs(), missing
            )
            candidate_for_binary = masked_mean_per_sample(
                (v16_candidate.detach() - x_f_gt).abs(), missing
            )
            relative_gain = (
                base_for_binary - candidate_for_binary
            ) / base_for_binary.clamp_min(1e-6)
            margin = float(v16_cfg.get("binary_gain_margin", 0.002))
            target_accept = torch.full_like(relative_gain, 0.5)
            target_accept = torch.where(
                relative_gain > margin,
                torch.full_like(target_accept, 0.95),
                target_accept,
            )
            target_accept = torch.where(
                relative_gain < -margin,
                torch.full_like(target_accept, 0.05),
                target_accept,
            )
            logits = outputs["residual_alpha_logit"].flatten(1).mean(dim=1)
            l_v16_calibration = F.binary_cross_entropy_with_logits(
                logits, target_accept.detach()
            )
        else:
            l_v16_calibration = F.smooth_l1_loss(alpha_pred, alpha_star.detach())

        missing = expand_mask_as(1.0 - m_f, v16_base)
        base_sample = masked_mean_per_sample(
            (v16_base.detach() - x_f_gt).abs(), missing
        )
        candidate_sample = masked_mean_per_sample(
            (v16_candidate.detach() - x_f_gt).abs(), missing
        )
        final_sample = masked_mean_per_sample(
            (v16_final - x_f_gt).abs(), missing
        )
        oracle_prediction = (
            v16_base.detach()
            + alpha_star.view(-1, 1, 1, 1, 1) * delta_candidate.detach()
        )
        oracle_sample = masked_mean_per_sample(
            (oracle_prediction - x_f_gt).abs(), missing
        )
        l_v16_safe = F.relu(final_sample - base_sample).mean()

        if teacher_prediction is not None:
            teacher_sample = masked_mean_per_sample(
                (teacher_prediction - x_f_gt).abs(), missing
            )
            teacher_hidden_mae = teacher_sample.mean()
            base_vs_teacher_gap = base_sample.mean() - teacher_hidden_mae
            final_vs_teacher_gain = teacher_hidden_mae - final_sample.mean()
            teacher_available = torch.ones_like(l_main)
        else:
            teacher_hidden_mae = _empty_loss_like(l_main)
            base_vs_teacher_gap = _empty_loss_like(l_main)
            final_vs_teacher_gain = _empty_loss_like(l_main)
            teacher_available = _empty_loss_like(l_main)
        base_hidden_mae = base_sample.mean()
        candidate_hidden_mae = candidate_sample.mean()
        final_hidden_mae = final_sample.mean()
        oracle_hidden_mae = oracle_sample.mean()
        alpha_error = alpha_pred - alpha_star
        v16_logs = {
            "v16_teacher_hidden_mae": teacher_hidden_mae,
            "v16_student_base_hidden_mae": base_hidden_mae,
            "v16_candidate_hidden_mae": candidate_hidden_mae,
            "v16_final_hidden_mae": final_hidden_mae,
            "v16_teacher_available": teacher_available,
            "v16_student_base_vs_teacher_gap": base_vs_teacher_gap,
            "v16_final_vs_teacher_gain": final_vs_teacher_gain,
            "v16_final_vs_base_gain": base_hidden_mae - final_hidden_mae,
            "v16_candidate_vs_base_gain": base_hidden_mae - candidate_hidden_mae,
            "v16_alpha_pred_mean": alpha_pred.mean(),
            "v16_alpha_pred_std": alpha_pred.std(unbiased=False),
            "v16_alpha_pred_min": alpha_pred.min(),
            "v16_alpha_pred_max": alpha_pred.max(),
            "v16_alpha_star_mean": alpha_star.mean(),
            "v16_alpha_star_std": alpha_star.std(unbiased=False),
            "v16_alpha_star_min": alpha_star.min(),
            "v16_alpha_star_max": alpha_star.max(),
            "v16_alpha_absolute_error": alpha_error.abs().mean(),
            "v16_alpha_rmse": alpha_error.square().mean().sqrt(),
            "v16_alpha_pearson": _pearson(alpha_pred.detach(), alpha_star),
            "v16_alpha_spearman": _spearman(alpha_pred.detach(), alpha_star),
            "v16_alpha_zero_target_rate": (alpha_star == 0).to(l_main.dtype).mean(),
            "v16_alpha_full_target_rate": (alpha_star == 1).to(l_main.dtype).mean(),
            "v16_alpha_middle_target_rate": (
                (alpha_star > 0) & (alpha_star < 1)
            ).to(l_main.dtype).mean(),
            "v16_oracle_hidden_mae": oracle_hidden_mae,
            "v16_calibration_regret": final_hidden_mae - oracle_hidden_mae,
            "v16_final_violation_rate": (
                final_sample > base_sample
            ).to(l_main.dtype).mean(),
        }
        condition = outputs.get("calibration_condition")
        if torch.is_tensor(condition):
            correlation_indices = {
                "missing_rate": 0,
                "scale_weight_f": 5,
                "scale_weight_m": 6,
                "scale_weight_c": 7,
                "candidate_relative_rms": 8,
            }
            if condition.shape[1] >= 12:
                correlation_indices.update({
                    "branch_disagreement": 9,
                    "observed_base_mae": 10,
                    "observed_gain": 11,
                })
            for name, index in correlation_indices.items():
                v16_logs[f"v16_alpha_corr_{name}"] = _pearson(
                    condition[:, index].detach(), alpha_star
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
    v14_cfg = cfg.get("model", {}).get("v14", {})
    loss = loss + loss_cfg.get("lambda_v14_base", v14_cfg.get("lambda_base", 0.0)) * l_v14_base
    loss = loss + loss_cfg.get("lambda_v14_mid", v14_cfg.get("lambda_mid", 0.0)) * l_v14_mid
    loss = loss + loss_cfg.get("lambda_v14_coarse", v14_cfg.get("lambda_coarse", 0.0)) * l_v14_coarse
    loss = loss + loss_cfg.get("lambda_v14_regret", v14_cfg.get("lambda_regret", 0.0)) * l_v14_regret
    loss = loss + loss_cfg.get("lambda_v14_gate", v14_cfg.get("lambda_gate", 0.0)) * l_v14_gate
    v15_cfg = cfg.get("model", {}).get("v15", {})
    loss = loss + loss_cfg.get(
        "lambda_v15_base", v15_cfg.get("lambda_base", 0.0)
    ) * l_v15_base
    loss = loss + loss_cfg.get(
        "lambda_v15_delta", v15_cfg.get("lambda_delta", 0.0)
    ) * l_v15_delta
    loss = loss + loss_cfg.get(
        "lambda_v15_safe", v15_cfg.get("lambda_safe", 0.0)
    ) * l_v15_safe
    v15_1_cfg = cfg.get("model", {}).get("v15_1", {})
    loss = loss + loss_cfg.get(
        "lambda_v15_1_base", v15_1_cfg.get("lambda_base", 0.0)
    ) * l_v15_1_base
    loss = loss + loss_cfg.get(
        "lambda_v15_1_candidate", v15_1_cfg.get("lambda_candidate", 0.0)
    ) * l_v15_1_candidate
    loss = loss + loss_cfg.get(
        "lambda_v15_1_accept", v15_1_cfg.get("lambda_accept", 0.0)
    ) * l_v15_1_accept
    loss = loss + loss_cfg.get(
        "lambda_v15_1_safe", v15_1_cfg.get("lambda_safe", 0.0)
    ) * l_v15_1_safe
    if is_v16:
        loss = loss + loss_cfg.get(
            "lambda_v16_anchor", v16_cfg.get("lambda_anchor", 0.30)
        ) * l_v16_anchor
        loss = loss + loss_cfg.get(
            "lambda_v16_candidate", v16_cfg.get("lambda_candidate", 0.05)
        ) * l_v16_candidate
        loss = loss + loss_cfg.get(
            "lambda_v16_calibration", v16_cfg.get("lambda_calibration", 0.10)
        ) * l_v16_calibration
        loss = loss + loss_cfg.get(
            "lambda_v16_safe", v16_cfg.get("lambda_safe", 0.10)
        ) * l_v16_safe
    if loss_cfg.get("lambda_final", 0.0) > 0:
        loss = loss + loss_cfg["lambda_final"] * l_final
    # During the proposer warm-up the backbone and calibrator are frozen and the
    # candidate objective is intentionally the sole optimization target.
    if is_v16 and outputs.get("v16_stage") == "warmup":
        loss = l_v16_candidate
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
    if is_v14 and x_hat_base is not None:
        loss_logs.update({
            "l_v14_base": l_v14_base.detach(),
            "l_v14_mid": l_v14_mid.detach(),
            "l_v14_coarse": l_v14_coarse.detach(),
            "l_v14_regret": l_v14_regret.detach(),
            "l_v14_gate": l_v14_gate.detach(),
            "v14_base_hidden_mae": v14_base_hidden_mae.detach(),
            "v14_ctf_hidden_mae": v14_ctf_hidden_mae.detach(),
            "v14_final_hidden_mae": v14_final_hidden_mae.detach(),
            "v14_non_regression_violation_rate": v14_violation_rate.detach(),
        })
    if is_v15 and x_hat_base is not None and effective_delta is not None:
        loss_logs.update({
            "l_v15_base": l_v15_base.detach(),
            "l_v15_delta": l_v15_delta.detach(),
            "l_v15_safe": l_v15_safe.detach(),
            "v15_base_hidden_mae": v15_base_hidden_mae.detach(),
            "v15_final_hidden_mae": v15_final_hidden_mae.detach(),
            "v15_final_vs_base_improvement": v15_final_vs_base_improvement.detach(),
            "v15_sample_non_regression_violation_rate": v15_sample_violation_rate.detach(),
        })
    if is_v15_1 and x_hat_base is not None and x_hat_candidate is not None:
        loss_logs.update({
            "l_v15_1_base": l_v15_1_base.detach(),
            "l_v15_1_candidate": l_v15_1_candidate.detach(),
            "l_v15_1_accept": l_v15_1_accept.detach(),
            "l_v15_1_safe": l_v15_1_safe.detach(),
            "v15_1_base_hidden_mae": v15_1_base_hidden_mae.detach(),
            "v15_1_candidate_hidden_mae": v15_1_candidate_hidden_mae.detach(),
            "v15_1_final_hidden_mae": v15_1_final_hidden_mae.detach(),
            "v15_1_candidate_vs_base_gain": v15_1_candidate_vs_base_gain.detach(),
            "v15_1_final_vs_base_gain": v15_1_final_vs_base_gain.detach(),
            "v15_1_candidate_violation_rate": v15_1_candidate_violation_rate.detach(),
            "v15_1_final_violation_rate": v15_1_final_violation_rate.detach(),
            "v15_1_accept_target_mean": v15_1_accept_target_mean.detach(),
            "v15_1_accept_positive_rate": v15_1_accept_positive_rate.detach(),
            "v15_1_accept_negative_rate": v15_1_accept_negative_rate.detach(),
            "v15_1_accept_uncertain_rate": v15_1_accept_uncertain_rate.detach(),
            "v15_1_accept_accuracy": v15_1_accept_accuracy.detach(),
        })
    if is_v16:
        loss_logs.update({
            "l_v16_base_gt": l_v16_base_gt.detach(),
            "l_v16_base_teacher": l_v16_base_teacher.detach(),
            "l_v16_anchor": l_v16_anchor.detach(),
            "l_v16_candidate": l_v16_candidate.detach(),
            "l_v16_calibration": l_v16_calibration.detach(),
            "l_v16_safe": l_v16_safe.detach(),
            "v16_warmup_active": torch.as_tensor(
                float(outputs.get("v16_stage") == "warmup"),
                device=l_main.device,
            ),
            **{key: value.detach() for key, value in v16_logs.items()},
        })
    return loss, loss_logs
