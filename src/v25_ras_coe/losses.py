"""Target-only training objectives for selective repair."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from stmoe_imputer.losses import compute_coe_loss, supervision_mask


def _point_errors(old: torch.Tensor, candidate: torch.Tensor,
                  target: torch.Tensor, selected: torch.Tensor):
    """FP32 MAE averaged only over supervised variables at each point."""
    q = selected.float()
    count = q.sum(dim=1, keepdim=True)
    safe_target = torch.where(selected, target.float(), 0.0)
    old_values = torch.where(selected, old.float(), 0.0)
    candidate_values = torch.where(selected, candidate.float(), 0.0)
    old_error = ((old_values - safe_target).abs() * q).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
    candidate_error = ((candidate_values - safe_target).abs() * q).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
    return old_error, candidate_error, count > 0


def repair_oracle_labels(old: torch.Tensor, candidate: torch.Tensor,
                         target: torch.Tensor, selected: torch.Tensor,
                         margin: float = 0.0):
    """Detached improvement labels; ambiguous and unsupervised points are excluded."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("repair_accept_margin must be finite and nonnegative")
    with torch.no_grad():
        old_error, candidate_error, supervised = _point_errors(old, candidate, target, selected)
        improvement = old_error - candidate_error
        positive = supervised & (improvement > margin)
        negative = supervised & (improvement < -margin)
    return positive, negative, old_error.detach(), candidate_error.detach()


def _balanced_bce(logits: torch.Tensor, positive: torch.Tensor,
                  negative: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if bool(positive.any()) and bool(negative.any()):
        return 0.5 * (
            F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive])) +
            F.binary_cross_entropy_with_logits(logits[negative], torch.zeros_like(logits[negative]))
        )
    valid = positive | negative
    if bool(valid.any()):
        return F.binary_cross_entropy_with_logits(logits[valid], positive[valid].float())
    return logits.sum() * 0.0


def compute_repair_acceptance_loss(outputs: dict, batch: dict, cfg: dict):
    coe = outputs["coe"]
    logits = coe["repair_acceptance_aux_logits"]
    target = batch["x_f_gt"]
    selected = supervision_mask(target, batch["m_f"], batch.get("target_mask"))
    margin = float(cfg.get("loss", {}).get("repair_accept_margin", 0.0))
    rounds = logits.shape[1]
    if rounds != len(coe["candidate_completions"]) or rounds != len(coe["completions"]):
        raise ValueError("Repair acceptance logits and completion histories must have the same number of rounds")
    old = coe["initial_completion"]
    losses = []
    positive_counts = []
    valid_counts = []
    for index in range(rounds):
        candidate = coe["candidate_completions"][index]
        positive, negative, _, _ = repair_oracle_labels(old, candidate, target, selected, margin)
        losses.append(_balanced_bce(logits[:, index], positive, negative))
        positive_counts.append(positive.sum().detach())
        valid_counts.append((positive | negative).sum().detach())
        old = coe["completions"][index]
    zero = logits.sum() * 0.0
    loss = torch.stack(losses).mean() if losses else zero
    positive_total = torch.stack(positive_counts).sum() if positive_counts else zero.detach()
    valid_total = torch.stack(valid_counts).sum() if valid_counts else zero.detach()
    return loss, {"accept_valid_points": valid_total,
                  "accept_positive_rate": positive_total.float() / valid_total.clamp_min(1).float()}


def _monotonic_loss(outputs: dict, batch: dict, tolerance: float) -> torch.Tensor:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("repair_monotonic_tolerance must be finite and nonnegative")
    coe = outputs["coe"]
    target = batch["x_f_gt"]
    selected = supervision_mask(target, batch["m_f"], batch.get("target_mask"))
    old = coe["initial_completion"]
    losses = []
    for accepted in coe["completions"]:
        old_error, accepted_error, valid = _point_errors(old, accepted, target, selected)
        regret = F.relu(accepted_error - old_error - tolerance)
        losses.append(regret[valid].mean() if bool(valid.any()) else accepted.sum() * 0.0)
        old = accepted
    return torch.stack(losses).mean() if losses else outputs["x_hat_main"].sum() * 0.0


def compute_v25_loss(outputs: dict, batch: dict, cfg: dict,
                     epoch: int | None = None):
    """Preserve the v24 task/balance loss, adding supervised repair only for A3–A5."""
    total, logs = compute_coe_loss(outputs, batch, cfg)
    loss_cfg = cfg.get("loss", {})
    accept_weight = float(loss_cfg.get("lambda_coe_accept", 0.0))
    monotonic_weight = float(loss_cfg.get("lambda_coe_monotonic", 0.0))
    if not math.isfinite(accept_weight) or accept_weight < 0:
        raise ValueError("lambda_coe_accept must be finite and nonnegative")
    if not math.isfinite(monotonic_weight) or monotonic_weight < 0:
        raise ValueError("lambda_coe_monotonic must be finite and nonnegative")
    active = cfg["model"].get("coe", {}).get("repair_acceptance", "none") == "latent_point"
    zero = total * 0.0
    if active and accept_weight:
        accept, accept_logs = compute_repair_acceptance_loss(outputs, batch, cfg)
    else:
        accept, accept_logs = zero, {"accept_valid_points": zero.detach(),
                                     "accept_positive_rate": zero.detach()}
    monotonic = (_monotonic_loss(outputs, batch,
                    float(loss_cfg.get("repair_monotonic_tolerance", 0.0)))
                 if active and monotonic_weight else zero)
    total = total + accept_weight * accept + monotonic_weight * monotonic
    logs.update({"loss": total.detach(), "l_coe_accept": accept.detach(),
                 "l_coe_accept_weighted": (accept_weight * accept).detach(),
                 "l_coe_monotonic": monotonic.detach(),
                 "l_coe_monotonic_weighted": (monotonic_weight * monotonic).detach(),
                 **accept_logs})
    return total, logs
