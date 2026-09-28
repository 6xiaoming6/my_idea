"""Sparse partner supervision and label-only validation Oracle diagnostics.

Candidate labels are computed only from hidden training/validation targets. They
never become router inputs, and test data is never used for Oracle selection.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .losses import supervision_mask
from .utils.device import move_batch_to_device


def _hidden_error(prediction: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["x_f_gt"]
    selected = supervision_mask(target, batch["m_f"], batch.get("target_mask"))
    errors = torch.where(selected, (prediction.float() - target.float()).abs(), 0.)
    return errors.flatten(1).sum(1), selected.flatten(1).sum(1)


def _candidate_ids(primary: torch.Tensor, native: torch.Tensor, chosen: torch.Tensor,
                   experts: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for anchor, second, selected in zip(primary.tolist(), native.tolist(), chosen.tolist()):
        ids = [second]
        if selected not in ids:
            ids.append(selected)
        remaining = [idx for idx in range(experts) if idx != anchor and idx not in ids]
        while len(ids) < 3:
            offset = int(torch.randint(len(remaining), (1,), generator=generator))
            ids.append(remaining.pop(offset))
        result.append(ids)
    return torch.tensor(result, device=primary.device, dtype=torch.long)


def partner_candidate_loss(raw_model: torch.nn.Module, batch: dict, outputs: dict,
                           cfg: dict, epoch: int, batch_index: int) -> tuple[torch.Tensor, dict]:
    """Train the partner scorer from three no-grad full-suffix candidate trials."""
    coe = outputs["coe"]
    scores = coe["partner_scores"]
    if scores is None:
        raise ValueError("Partner candidate loss requires partner scores")
    steps = scores.shape[1]
    # Use an independent deterministic stream; candidate exploration must not
    # perturb model, data-loader, or mask-generator random states.
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    step = (epoch + batch_index - 1) % steps
    primary = coe["primary_ids"][:, step].detach().cpu()
    individual = coe["route_logits"][:, step].detach().float()
    native = individual.scatter(1, coe["primary_ids"][:, step, None], -torch.inf).argmax(1).cpu()
    chosen = coe["partner_ids"][:, step].detach().cpu()
    ids = _candidate_ids(primary, native, chosen, scores.shape[-1],
                         int(cfg["seed"]) + 100003 * epoch + 211 * batch_index + 17 * rank)
    ids = ids.to(scores.device)
    selected_scores = scores[:, step].gather(1, ids)
    trials = []
    with torch.no_grad():
        for candidate in ids.unbind(dim=1):
            trial = raw_model({**batch, "forced_partner_step": step,
                               "forced_partner_ids": candidate})
            errors, counts = _hidden_error(trial["x_hat_final"], batch)
            trials.append(errors / counts.clamp_min(1))
    errors = torch.stack(trials, dim=1)
    # Center by sample and scale by candidate spread. The floor prevents very
    # small early differences from producing an overconfident teacher.
    centered = errors - errors.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(
        float(cfg["train"]["partner_probe"].get("min_scale", 0.5))
    )
    teacher_logits = -centered / scale
    # The native second expert is first in ids and breaks exact early ties.
    teacher_logits[:, 0] += 1e-4
    targets = F.softmax(teacher_logits.detach(), dim=-1)
    valid = counts > 0
    if valid.any():
        loss = -(targets[valid] * F.log_softmax(selected_scores[valid].float(), dim=-1)).sum(-1).mean()
    else:
        loss = selected_scores.sum() * 0.
    return loss, {
        "partner_probe_forward_calls": 3.,
        "partner_probe_step": float(step + 1),
        "partner_probe_teacher_gap": float((errors.max(1).values - errors.min(1).values).mean()),
    }


@torch.no_grad()
def last_step_oracle(raw_model: torch.nn.Module, loader, device: torch.device,
                     cfg: dict, max_samples: int) -> dict[str, float]:
    """Enumerate 15 last-round pairs on validation data; never use for routing."""
    raw_model.eval()
    backbone = raw_model.main_branch
    if backbone.top_k != 2 or backbone.num_steps != 3:
        raise ValueError("Last-step Oracle requires a three-round Top-2 CoE")
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    local_limit = math.ceil(max_samples / world)
    total = torch.zeros(9, device=device, dtype=torch.float64)
    seen = 0
    amp = cfg["train"].get("amp", True) and device.type == "cuda"
    for batch in loader:
        if seen >= local_limit:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            natural = raw_model(batch)
            logits = natural["coe"]["route_logits"][:, -1].float()
            native_ranked = logits.topk(2, dim=-1).indices
            primary = native_ranked[:, 0]
            top2 = native_ranked.sort(dim=-1).values
            pairs = backbone.pair_indices
            candidate_errors = []
            counts = None
            for pair in pairs:
                pair_batch = pair.to(device).expand(logits.shape[0], -1)
                trial = raw_model({**batch, "forced_pair_step": backbone.num_steps - 1,
                                   "forced_pair_indices": pair_batch})
                errors, counts = _hidden_error(trial["x_hat_final"], batch)
                candidate_errors.append(errors)
        errors = torch.stack(candidate_errors, dim=1)
        pair_id = (pairs[None].to(device) == top2[:, None]).all(-1).long().argmax(-1)
        native_error = errors.gather(1, pair_id[:, None]).squeeze(1)
        anchor_mask = (pairs[None].to(device) == primary[:, None, None]).any(-1)
        anchor_error = errors.masked_fill(~anchor_mask, torch.inf).min(1).values
        all_error = errors.min(1).values
        valid = counts > 0
        if valid.any():
            total[0] += native_error[valid].sum().double()
            total[1] += anchor_error[valid].sum().double()
            total[2] += all_error[valid].sum().double()
            total[3] += counts[valid].sum().double()
            total[4] += valid.sum().double()
            total[5] += ((native_error - anchor_error) / counts.clamp_min(1) > 0.1)[valid].sum().double()
            total[6] += ((native_error - all_error) / counts.clamp_min(1) > 0.1)[valid].sum().double()
            total[7] += ((anchor_error - all_error) / counts.clamp_min(1) < 1e-6)[valid].sum().double()
        total[8] += logits.shape[0]
        seen += logits.shape[0]
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if total[3] <= 0:
        raise ValueError("Oracle validation subset has no finite hidden labels")
    return {
        "status": "label_only_diagnostic_not_deployable",
        "step": float(backbone.num_steps),
        "evaluated_samples": float(total[8]),
        "supervised_samples": float(total[4]),
        "supervised_targets": float(total[3]),
        "top2_mae": float(total[0] / total[3]),
        "anchor_best_mae": float(total[1] / total[3]),
        "all_best_mae": float(total[2] / total[3]),
        "anchor_improvable_fraction_gt_0p1": float(total[5] / total[4]),
        "all_improvable_fraction_gt_0p1": float(total[6] / total[4]),
        "anchor_retains_all_best_fraction": float(total[7] / total[4]),
        "candidate_forward_calls_per_sample": float(len(pairs)),
    }
