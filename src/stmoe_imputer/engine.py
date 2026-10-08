from __future__ import annotations

from collections import defaultdict
import math
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from .losses import compute_main_stage_loss, supervision_mask
from .metrics import MaskedMetricAccumulator, masked_metrics
from .data.diverse_masks import ALL_FAMILIES as FAMILIES
from .routing_metrics import CoERoutingMetricAccumulator, RoutingMetricAccumulator, active_routing_scales
from .models.registry import resolve_architecture
from .partner_study import partner_candidate_loss
from .utils.device import move_batch_to_device



def _loss_gradient_alignment(task: torch.Tensor, auxiliary: torch.Tensor,
                             parameters: list[torch.nn.Parameter]) -> tuple[float, float, float]:
    """Norms and cosine before the losses are combined; zero cosine is undefined."""
    if not parameters:
        return 0., 0., 0.
    task_grads = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
    task_sq = auxiliary.new_zeros((), dtype=torch.float32)
    aux_sq = task_sq.clone()
    dot = task_sq.clone()
    for left, right in zip(task_grads, aux_grads):
        if left is not None:
            left = left.detach().float()
            task_sq = task_sq + left.square().sum()
        if right is not None:
            right = right.detach().float()
            aux_sq = aux_sq + right.square().sum()
        if left is not None and right is not None:
            dot = dot + (left * right).sum()
    task_norm, aux_norm = task_sq.sqrt(), aux_sq.sqrt()
    cosine = dot / (task_norm * aux_norm).clamp_min(1e-30)
    return float(task_norm.cpu()), float(aux_norm.cpu()), float(cosine.cpu())


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    base_lr = train_cfg["lr_main"]
    aux_lr = train_cfg.get("lr_aux", base_lr)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    gate_lr_mult = train_cfg.get("gate_lr_mult", 1.0)
    scalar_lr_mult = train_cfg.get("scalar_lr_mult", 2.0)
    v14_lr = train_cfg.get("lr_v14", base_lr)

    grouped: dict[str, dict] = {
        "main": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "router": {"params": [], "lr": train_cfg.get("lr_router", base_lr), "weight_decay": 0.0},
        "gate": {"params": [], "lr": base_lr * gate_lr_mult, "weight_decay": 0.0},
        "scalar": {"params": [], "lr": base_lr * scalar_lr_mult, "weight_decay": 0.0},
        "no_decay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "v14": {"params": [], "lr": v14_lr, "weight_decay": weight_decay},
        "v14_no_decay": {"params": [], "lr": v14_lr, "weight_decay": 0.0},
        "other": {"params": [], "lr": aux_lr, "weight_decay": weight_decay},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_l = name.lower()
        is_v14_new = any(
            token in name_l
            for token in (
                "main_branch.condition_encoder",
                "main_branch.controller",
                "main_branch.refiner",
                "main_branch.local_residual_gate",
            )
        )
        if "lr_router" in train_cfg and name_l.startswith("main_branch.routers."):
            grouped["router"]["params"].append(param)
        elif is_v14_new and (name_l.endswith(".bias") or "norm" in name_l):
            grouped["v14_no_decay"]["params"].append(param)
        elif is_v14_new:
            grouped["v14"]["params"].append(param)
        elif any(
            token in name_l
            for token in (
                "route_gamma",
                "shared_gamma",
                "shared_input_adapter.beta",
                "controller.mid_bias",
                "controller.fine_bias",
                "controller.final_bias",
            )
        ):
            grouped["scalar"]["params"].append(param)
        elif "scale_gate" in name_l or "branch_gate" in name_l:
            grouped["gate"]["params"].append(param)
        elif name_l.endswith(".bias") or "norm" in name_l or "embedding" in name_l or "scale_embed" in name_l:
            grouped["no_decay"]["params"].append(param)
        elif name.startswith("main_branch."):
            grouped["main"]["params"].append(param)
        else:
            grouped["other"]["params"].append(param)

    groups = [
        {"name": name, **group}
        for name, group in grouped.items()
        if group["params"]
    ]
    return torch.optim.AdamW(groups)


class WarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        warmup_epochs: int = 5,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.max_epochs = max(1, max_epochs)
        self.warmup_epochs = max(0, warmup_epochs)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch + 1
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        cosine_epochs = max(1, self.max_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - self.warmup_epochs) / cosine_epochs))
        cosine_factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


class EndpointCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Cosine decay that stays at eta_min after the configured final epoch."""

    def _get_closed_form_lr(self):
        progress = min(self.last_epoch, self.T_max) / self.T_max
        return [self.eta_min + (base_lr - self.eta_min) *
                (1.0 + math.cos(math.pi * progress)) / 2.0 for base_lr in self.base_lrs]

    def get_lr(self):
        return self._get_closed_form_lr()


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.LRScheduler | None:
    sched_cfg = cfg["train"].get("scheduler", {})
    sched_type = sched_cfg.get("type", "none")
    if sched_type == "none":
        return None
    if sched_type == "cosine":
        total_epochs = int(sched_cfg.get("total_epochs", cfg["train"]["epochs"]))
        # step() runs after each epoch: N training epochs use scheduler steps 0..N-1.
        # Opt-in keeps historical experiment schedules unchanged.
        scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
        if sched_cfg.get("reach_min_at_last_epoch", False):
            total_epochs = max(1, total_epochs - 1)
            scheduler_class = EndpointCosineAnnealingLR
        return scheduler_class(
            optimizer,
            T_max=total_epochs,
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    if sched_type == "warmup_cosine":
        return WarmupCosineLR(
            optimizer,
            max_epochs=sched_cfg.get("total_epochs", cfg["train"]["epochs"]),
            warmup_epochs=sched_cfg.get("warmup_epochs", 5),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    raise ValueError(f"Unknown scheduler type: {sched_type}")


def _distributed_merge(logs, exact_metrics, active_exact_metrics, routing_metrics, coe_quality, counters):
    if not (dist.is_available() and dist.is_initialized()):
        return logs, exact_metrics, active_exact_metrics, routing_metrics, coe_quality, counters
    payload = (dict(logs), exact_metrics, active_exact_metrics, routing_metrics, coe_quality, counters)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    merged_logs: dict[str, list[float]] = defaultdict(list)
    merged_exact = {key: MaskedMetricAccumulator() for key in exact_metrics}
    merged_active = set()
    merged_routing = _routing_accumulator_for_merge(routing_metrics)
    merged_quality = _CoEQualityMetrics() if coe_quality is not None else None
    merged_counters = defaultdict(float)
    for part_logs, part_exact, part_active, part_routing, part_quality, part_counters in gathered:
        for key, values in part_logs.items():
            merged_logs[key].extend(values)
        for key, metric in part_exact.items():
            merged_exact[key].merge(metric)
        merged_active.update(part_active)
        if merged_routing is not None and part_routing is not None:
            merged_routing.merge(part_routing)
        if merged_quality is not None and part_quality is not None:
            merged_quality.merge(part_quality)
        for key, value in part_counters.items():
            merged_counters[key] += value
    return merged_logs, merged_exact, merged_active, merged_routing, merged_quality, merged_counters


def _routing_accumulator_for_merge(routing_metrics):
    if isinstance(routing_metrics, CoERoutingMetricAccumulator):
        return CoERoutingMetricAccumulator()
    if isinstance(routing_metrics, RoutingMetricAccumulator):
        return RoutingMetricAccumulator(routing_metrics.scale_names)
    return None


def _mean_logs(accumulator: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(values) / max(1, len(values)) for key, values in accumulator.items()}


def _zero_pair() -> list[float]:
    return [0.0, 0.0]


class _CoEQualityMetrics:
    """Exact per-family errors and missing-point acceptance deltas."""

    def __init__(self) -> None:
        self.family = {name: MaskedMetricAccumulator() for name in FAMILIES}
        self.acceptance: dict[str, list[float]] = defaultdict(_zero_pair)
        self.coordination = None

    @torch.no_grad()
    def update(self, outputs: dict, batch: dict) -> None:
        target, mask = batch["x_f_gt"], batch["m_f"]
        if "coordination" in outputs:
            if self.coordination is None:
                from .coordination import CoordinationMetrics
                self.coordination = CoordinationMetrics()
            self.coordination.update(outputs, batch)
        family_ids = batch.get("mask_family")
        if family_ids is not None:
            for family_id in family_ids.unique().tolist():
                index = family_ids == family_id
                self.family[FAMILIES[int(family_id)]].update(
                    outputs["x_hat_final"][index], target[index], mask[index],
                    target_mask=(batch["target_mask"][index] if "target_mask" in batch else None),
                )
        coe = outputs.get("coe", {})
        candidates = coe.get("candidate_completions", ())
        completions = coe.get("completions", ())
        old = coe.get("initial_completion")
        if old is None or not candidates or len(candidates) != len(completions):
            return
        selected = supervision_mask(target, mask, batch.get("target_mask"))
        count = float(selected.sum().item())
        if not count:
            return
        for step, (candidate, accepted) in enumerate(zip(candidates, completions), start=1):
            old_error = (old[selected].float() - target[selected].float()).abs()
            for label, value in (("candidate", candidate), ("accepted", accepted)):
                change = (value[selected].float() - target[selected].float()).abs() - old_error
                for direction, delta in (("harm", change.clamp_min(0)),
                                         ("benefit", (-change).clamp_min(0))):
                    totals = self.acceptance[f"coe_step{step}_{label}_{direction}"]
                    totals[0] += float(delta.sum().double().cpu())
                    totals[1] += count
            old = accepted

    def merge(self, other: "_CoEQualityMetrics") -> None:
        if other.coordination is not None:
            if self.coordination is None:
                from .coordination import CoordinationMetrics
                self.coordination = CoordinationMetrics()
            self.coordination.merge(other.coordination)
        for name, metric in other.family.items():
            self.family[name].merge(metric)
        for name, values in other.acceptance.items():
            self.acceptance[name][0] += values[0]
            self.acceptance[name][1] += values[1]

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, metric in self.family.items():
            if metric.count:
                result[f"coe_family_{name}_absolute_error"] = metric.absolute_error
                result[f"coe_family_{name}_squared_error"] = metric.squared_error
                result[f"coe_family_{name}_count"] = metric.count
                result[f"coe_family_{name}_mae"] = metric.absolute_error / metric.count
                result[f"coe_family_{name}_rmse"] = (metric.squared_error / metric.count) ** 0.5
        if self.coordination is not None:
            result.update(self.coordination.compute())
        for key, (total, count) in self.acceptance.items():
            result[key] = total / count
        return result


def _append_model_diagnostics(logs: dict[str, list[float]], outputs: dict) -> None:
    coe = outputs.get("diagnostics", {}).get("coe", {})
    for key, value in coe.items():
        if torch.is_tensor(value):
            logs[f"coe_{key}"].append(float(value.detach().float().mean().cpu()))
    scale_gate = outputs.get("gates", {}).get("scale_gate")
    if scale_gate is not None:
        labels = ("f", "m", "c")
        for idx, label in enumerate(labels):
            values = scale_gate[:, idx]
            logs[f"scale_gate_{label}_mean"].append(float(values.mean().detach().cpu()))
            logs[f"scale_gate_{label}_std"].append(float(values.std(unbiased=False).detach().cpu()))

    scale_evidence = outputs.get("gates", {}).get("scale_evidence")
    if scale_evidence is not None:
        for idx, label in enumerate(("f", "m", "c")):
            values = scale_evidence[:, idx]
            logs[f"scale_evidence_{label}_mean"].append(
                float(values.mean().detach().cpu())
            )

    route_gamma = outputs.get("route_gamma")
    if route_gamma is not None and torch.is_tensor(route_gamma):
        gamma_value = float(route_gamma.detach().cpu())
        logs["route_gamma"].append(gamma_value)
        logs["route_alpha"].append(gamma_value)

    branch_gate = outputs.get("gates", {}).get("branch_gate")
    if branch_gate is not None:
        shared = branch_gate[:, 0]
        route = branch_gate[:, 1]
        logs["branch_gate_shared_mean"].append(float(shared.mean().detach().cpu()))
        logs["branch_gate_route_mean"].append(float(route.mean().detach().cpu()))
        logs["branch_gate_shared_std"].append(float(shared.std(unbiased=False).detach().cpu()))
        logs["branch_gate_route_std"].append(float(route.std(unbiased=False).detach().cpu()))

    diagnostics = outputs.get("diagnostics", {})
    beta = diagnostics.get("shared_input_beta") if isinstance(diagnostics, dict) else None
    if beta is not None and torch.is_tensor(beta):
        logs["shared_input_beta_f"].append(float(beta[0].detach().cpu()))
        logs["shared_input_beta_m"].append(float(beta[1].detach().cpu()))
        logs["shared_input_beta_c"].append(float(beta[2].detach().cpu()))

    v14 = diagnostics.get("v14") if isinstance(diagnostics, dict) else None
    if isinstance(v14, dict):
        for key, value in v14.items():
            if value is None or not torch.is_tensor(value):
                continue
            value_f = value.detach().float()
            summary_key = f"v14_{key}" if key.endswith("_mean") else f"v14_{key}_mean"
            logs[summary_key].append(float(value_f.mean().cpu()))
            if key in {"alpha_mid", "alpha_fine", "alpha_final"}:
                logs[f"v14_{key}_std"].append(float(value_f.std(unbiased=False).cpu()))
                logs[f"v14_{key}_min"].append(float(value_f.min().cpu()))
                logs[f"v14_{key}_max"].append(float(value_f.max().cpu()))
            if key == "local_gate_modulation":
                logs["v14_local_gate_modulation_std"].append(
                    float(value_f.std(unbiased=False).cpu())
                )
                logs["v14_local_gate_modulation_min"].append(
                    float(value_f.min().cpu())
                )
                logs["v14_local_gate_modulation_max"].append(
                    float(value_f.max().cpu())
                )

    if isinstance(v14, dict):
        for scale in ("fine", "mid", "coarse"):
            gate = outputs.get("gates", {}).get(scale)
            if gate is None or not torch.is_tensor(gate):
                continue
            gate_f = gate.detach().float()
            entropy = -(gate_f * gate_f.clamp_min(1e-8).log()).sum(dim=1)
            logs[f"expert_entropy_{scale}"].append(float(entropy.mean().cpu()))
            selected = outputs.get("selected_masks", {}).get(scale)
            if selected is not None and torch.is_tensor(selected):
                usage = selected.detach().float().mean(dim=0)
                for index, value in enumerate(usage):
                    logs[f"expert_usage_{scale}_{index}"].append(float(value.cpu()))

    features = outputs.get("features", {})
    h_shared = features.get("h_shared") if isinstance(features, dict) else None
    h_route_proj = features.get("h_route_proj") if isinstance(features, dict) else None
    if h_shared is not None and h_route_proj is not None:
        shared_norm = h_shared.detach().float().square().mean().sqrt()
        route_norm = h_route_proj.detach().float().square().mean().sqrt()
        logs["effective_shared_norm"].append(float(shared_norm.cpu()))
        logs["effective_route_norm"].append(float(route_norm.cpu()))
        logs["effective_route_ratio"].append(float((route_norm / shared_norm.clamp_min(1e-6)).cpu()))


def _append_lr_logs(logs: dict[str, list[float]], optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        name = group.get("name", "group")
        logs[f"lr_group_{name}"].append(float(group["lr"]))


def _routing_accumulator(cfg: dict) -> RoutingMetricAccumulator | CoERoutingMetricAccumulator | None:
    if resolve_architecture(cfg) == "v24_ts_coe":
        return CoERoutingMetricAccumulator()
    main_cfg = cfg["model"]["main"]
    if (
        not main_cfg.get("use_routed_branch", True)
        or not main_cfg.get("use_router", True)
        or main_cfg.get("routing_mode", "topk") == "dense"
    ):
        return None
    scale_mode = main_cfg.get(
        "scale_mode", cfg["model"].get("scale_mode", "fine_mid_coarse")
    )
    return RoutingMetricAccumulator(active_routing_scales(scale_mode))


def _update_routing_metrics(accumulator, outputs: dict) -> None:
    if isinstance(accumulator, CoERoutingMetricAccumulator):
        accumulator.update(outputs["coe"])
    elif accumulator is not None:
        accumulator.update(outputs.get("gates", {}), outputs.get("selected_masks"))


def build_grad_scaler(device: torch.device, cfg: dict) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device.type, enabled=cfg["train"].get("amp", True) and device.type == "cuda"
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    show_progress: bool = True,
    total_epochs: int | None = None,
    training_context=None,
) -> dict[str, float]:
    model.train()
    if hasattr(getattr(loader, 'dataset', None), 'set_epoch'):
        loader.dataset.set_epoch(epoch)
    core_model = model.module if hasattr(model, "module") else model
    if hasattr(core_model.main_branch, "set_routing_epoch"):
        core_model.main_branch.set_routing_epoch(epoch)
    is_coe = resolve_architecture(cfg) == "v24_ts_coe"
    logs: dict[str, list[float]] = defaultdict(list)
    exact_metrics = {
        "": MaskedMetricAccumulator(),
        "_shared_aux": MaskedMetricAccumulator(),
        "_route_aux": MaskedMetricAccumulator(),
    }
    active_exact_metrics = {""}
    routing_metrics = _routing_accumulator(cfg)
    coe_quality = _CoEQualityMetrics() if is_coe else None
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    if scaler is None:
        # Compatibility for callers that manage only model/optimizer: retain
        # scale and growth tracker across epochs on that optimizer as well.
        scaler = getattr(optimizer, "_stmoe_grad_scaler", None)
        if scaler is None:
            scaler = build_grad_scaler(device, cfg)
            optimizer._stmoe_grad_scaler = scaler
    scale_start = scaler.get_scale()
    optimizer_steps = seen_samples = skipped_empty_batches = skipped_amp_steps = 0
    partner_probe_calls = 0
    partner_cfg = cfg["train"].get("partner_probe", {})
    partner_mode = is_coe and cfg["model"].get("coe", {}).get("pair_mode") in {"partner", "partner_residual"}
    if partner_mode and (type(partner_cfg.get("interval_batches")) is not int or
                         partner_cfg["interval_batches"] < 1 or
                         float(partner_cfg.get("weight", 0)) <= 0):
        raise ValueError("Partner routing requires positive train.partner_probe interval_batches and weight")
    alignment_every = partner_cfg.get("grad_diagnostic_interval_batches", 0) if partner_mode else 0
    if type(alignment_every) is not int or alignment_every < 0:
        raise ValueError("train.partner_probe.grad_diagnostic_interval_batches must be nonnegative")
    total_epochs = int(total_epochs or cfg["train"]["epochs"])
    progress = tqdm(loader, desc=f"train epoch {epoch}/{total_epochs}", leave=True, disable=not show_progress)
    for batch_index, batch in enumerate(progress):
        if training_context is not None:
            training_context.record_batch(batch)
        batch = move_batch_to_device(batch, device)
        if is_coe and not (dist.is_available() and dist.is_initialized()) and not bool(supervision_mask(
            batch["x_f_gt"], batch["m_f"], batch.get("target_mask")
        ).any()):
            skipped_empty_batches += 1
            continue
        seen_samples += int(batch["x_f_gt"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        weighted_ranking = None
        four_direction = cfg["model"].get("coe", {}).get("four_direction", {}).get("enabled", False)
        if four_direction:
            core_model.main_branch.prepare_training_batch(batch)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
            loss, loss_dict = compute_main_stage_loss(outputs, batch, cfg, epoch=epoch)
            if training_context is not None:
                extra, extra_logs = training_context.extra_loss(core_model, batch, outputs)
                loss = loss + extra
                loss_dict.update(extra_logs)
                loss_dict["loss"] = loss.detach()
            if four_direction and "four_probe" in outputs:
                auxiliary, probe_logs = core_model.main_branch.candidate_loss(batch, outputs)
                loss = loss + 0.1 * auxiliary
                loss_dict["loss"] = loss.detach()
                logs["l_four_candidate"].append(float(auxiliary.detach()))
                for key, value in probe_logs.items(): logs[key].append(value)
            task_loss = outputs["coe"].get("_loss_terms", {}).get("main", loss) if partner_mode else loss
            if partner_mode:
                # Keep DDP's partner-head hooks active on non-probe batches too.
                loss = loss + outputs["coe"]["partner_scores"].sum() * 0.
                if batch_index % partner_cfg["interval_batches"] == 0:
                    ranking, probe_logs = partner_candidate_loss(
                        core_model, batch, outputs, cfg, epoch, batch_index,
                    )
                    weighted_ranking = float(partner_cfg["weight"]) * ranking
                    loss = loss + weighted_ranking
                    logs["l_partner"].append(float(ranking.detach().cpu()))
                    for key, value in probe_logs.items():
                        logs[key].append(value)
                    partner_probe_calls += int(probe_logs["partner_probe_forward_calls"])
                loss_dict["loss"] = loss.detach()
        if (weighted_ranking is not None and alignment_every and
                ((epoch - 1) * len(loader) + batch_index + 1) % alignment_every == 0):
            backbone = core_model.main_branch
            decoder_params = list(backbone.decoder.parameters())
            primary_ids = torch.unique(outputs["coe"]["primary_ids"].detach()).tolist()
            primary_params = [param for expert_id in primary_ids
                              for param in backbone.routed_experts()[expert_id].parameters()]
            for group, parameters in (("decoder", decoder_params), ("primary_experts", primary_params)):
                task_norm, rank_norm, cosine = _loss_gradient_alignment(
                    task_loss, weighted_ranking, parameters,
                )
                logs[f"coe_partner_{group}_task_grad_norm"].append(task_norm)
                logs[f"coe_partner_{group}_rank_grad_norm"].append(rank_norm)
                logs[f"coe_partner_{group}_grad_cosine"].append(cosine)
        diagnostic_every = cfg["train"].get("router_grad_diagnostic_every", 0)
        if is_coe and diagnostic_every and batch_index % diagnostic_every == 0:
            parameters = [p for router in core_model.main_branch.routers for p in router.parameters()]
            for name, term in outputs["coe"].get("_loss_terms", {}).items():
                if not term.requires_grad:
                    continue
                grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                squares = [g.detach().float().square().sum() for g in grads if g is not None]
                logs[f"coe_router_{name}_grad_norm"].append(float(torch.stack(squares).sum().sqrt()) if squares else 0.)
        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip_norm")
        if grad_clip or is_coe:
            scaler.unscale_(optimizer)
        if is_coe:
            for step, router in enumerate(core_model.main_branch.routers):
                gradients = [p.grad.detach().float().square().sum()
                             for p in router.parameters() if p.grad is not None]
                if gradients:
                    logs[f"coe_step{step + 1}_router_grad_norm"].append(
                        float(torch.stack(gradients).sum().sqrt().cpu())
                    )
            expert_steps = (range(core_model.main_branch.num_steps)
                            if getattr(core_model.main_branch, "expert_sharing", "shared") == "per_step"
                            else range(1))
            for expert_step in expert_steps:
                for name, expert in zip(
                    core_model.main_branch.expert_names, core_model.main_branch.routed_experts(expert_step)
                ):
                    gradients = [p.grad.detach().float().square().sum()
                                 for p in expert.parameters() if p.grad is not None]
                    if gradients:
                        norm = float(torch.stack(gradients).sum().sqrt().cpu())
                        prefix = (f"coe_step{expert_step + 1}_expert_{name}" if
                                  getattr(core_model.main_branch, "expert_sharing", "shared") == "per_step" else
                                  f"coe_expert_{name}")
                        logs[f"{prefix}_grad_norm"].append(norm)
                        logs[f"{prefix}_finite_nonzero_gradient_batch_fraction"].append(
                            float(0 < norm < float("inf")))
            partner_head = getattr(core_model.main_branch, "partner_scorer", None)
            if partner_head is not None:
                gradients = [p.grad.detach().float().square().sum()
                             for p in partner_head.parameters() if p.grad is not None]
                logs["coe_partner_scorer_grad_norm"].append(
                    float(torch.stack(gradients).sum().sqrt().cpu()) if gradients else 0.
                )
            acceptance_head = getattr(core_model.main_branch, "acceptance_head", None)
            if acceptance_head is not None:
                gradients = [p.grad.detach().float().square().sum()
                             for p in acceptance_head.parameters() if p.grad is not None]
                if gradients:
                    logs["coe_acceptance_head_grad_norm"].append(
                        float(torch.stack(gradients).sum().sqrt().cpu()))
        if training_context is not None:
            for group in ('adapter','feedback','coverage'):
                gs=[p.grad.detach().float().square().sum() for n,p in core_model.main_branch.named_parameters()
                    if n.startswith(group) and p.grad is not None]
                if gs:logs[f'w_{group}_grad_norm'].append(float(torch.stack(gs).sum().sqrt()))
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            skipped_amp_steps += 1
        else:
            optimizer_steps += 1
            if training_context is not None:
                training_context.after_update(core_model)

        metrics = masked_metrics(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
        exact_metrics[""].update(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
        if outputs.get("x_hat_shared") is not None:
            shared_metrics = masked_metrics(outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
            metrics.update({f"{key}_shared_aux": value for key, value in shared_metrics.items()})
            exact_metrics["_shared_aux"].update(
                outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask")
            )
            active_exact_metrics.add("_shared_aux")
        if outputs.get("x_hat_route") is not None:
            route_metrics = masked_metrics(outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
            metrics.update({f"{key}_route_aux": value for key, value in route_metrics.items()})
            exact_metrics["_route_aux"].update(
                outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask")
            )
            active_exact_metrics.add("_route_aux")
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs)
        if coe_quality is not None:
            coe_quality.update(outputs, batch)
        if is_coe and 'mask_family' in batch:
            # Labels are introduced after forward and used only for diagnostics.
            outputs['coe']['mask_family'] = batch['mask_family'].repeat_interleave(outputs['coe'].get('routing_region_count', 1))
        _update_routing_metrics(routing_metrics, outputs)
        _append_lr_logs(logs, optimizer)
        progress.set_postfix(loss=logs["loss"][-1], mae=logs["mae"][-1], rmse=logs["rmse"][-1])
    logs, exact_metrics, active_exact_metrics, routing_metrics, coe_quality, counters = _distributed_merge(
        logs, exact_metrics, active_exact_metrics, routing_metrics, coe_quality,
        {"optimizer_steps": optimizer_steps, "seen_samples": seen_samples,
         "skipped_empty_batches": skipped_empty_batches, "skipped_amp_steps": skipped_amp_steps,
         "partner_probe_calls": partner_probe_calls},
    )
    seen_samples = counters["seen_samples"]
    skipped_empty_batches = counters["skipped_empty_batches"]
    skipped_amp_steps = counters["skipped_amp_steps"]
    result = _mean_logs(logs)
    if is_coe:
        result.update({
            "train_optimizer_steps": float(optimizer_steps),
            "train_seen_samples": float(seen_samples),
            "train_skipped_empty_batches": float(skipped_empty_batches),
            "train_skipped_amp_steps": float(skipped_amp_steps),
            "train_amp_scale_start": float(scale_start),
            "train_amp_scale_end": float(scaler.get_scale()),
            "train_partner_probe_forward_calls": float(counters["partner_probe_calls"]),
        })
    if is_coe and exact_metrics[""].count == 0:
        raise ValueError("TS-CoE training has no finite hidden supervision targets")
    for suffix in active_exact_metrics:
        for key, value in exact_metrics[suffix].compute().items():
            result[f"{key}{suffix}"] = value
    if coe_quality is not None:
        result.update(coe_quality.compute())
    if routing_metrics is not None:
        result.update(routing_metrics.compute())
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    desc: str = "eval",
    epoch: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    logs: dict[str, list[float]] = defaultdict(list)
    exact_metrics = {
        "": MaskedMetricAccumulator(),
        "_shared_aux": MaskedMetricAccumulator(),
        "_route_aux": MaskedMetricAccumulator(),
    }
    active_exact_metrics = {""}
    routing_metrics = _routing_accumulator(cfg)
    coe_quality = _CoEQualityMetrics() if resolve_architecture(cfg) == "v24_ts_coe" else None
    measure_forward = bool(cfg.get("train", {}).get("measure_forward_latency", False))
    counters = {"forward_seconds": 0.0, "forward_batches": 0.0, "forward_samples": 0.0}
    for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        batch = move_batch_to_device(batch, device)
        if measure_forward:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
        outputs = model(batch)
        if measure_forward:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            counters["forward_seconds"] += time.perf_counter() - forward_start
            counters["forward_batches"] += 1
            counters["forward_samples"] += batch["x_f_gt"].shape[0]
        _, loss_dict = compute_main_stage_loss(outputs, batch, cfg, epoch=epoch)
        metrics = masked_metrics(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
        exact_metrics[""].update(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
        if outputs.get("x_hat_shared") is not None:
            shared_metrics = masked_metrics(outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
            metrics.update({f"{key}_shared_aux": value for key, value in shared_metrics.items()})
            exact_metrics["_shared_aux"].update(
                outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask")
            )
            active_exact_metrics.add("_shared_aux")
        if outputs.get("x_hat_route") is not None:
            route_metrics = masked_metrics(outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask"))
            metrics.update({f"{key}_route_aux": value for key, value in route_metrics.items()})
            exact_metrics["_route_aux"].update(
                outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"], target_mask=batch.get("target_mask")
            )
            active_exact_metrics.add("_route_aux")
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs)
        if coe_quality is not None:
            coe_quality.update(outputs, batch)
        if resolve_architecture(cfg) == "v24_ts_coe" and 'mask_family' in batch:
            outputs['coe']['mask_family'] = batch['mask_family'].repeat_interleave(outputs['coe'].get('routing_region_count', 1))
        _update_routing_metrics(routing_metrics, outputs)
    logs, exact_metrics, active_exact_metrics, routing_metrics, coe_quality, counters = _distributed_merge(
        logs, exact_metrics, active_exact_metrics, routing_metrics, coe_quality, counters,
    )
    result = _mean_logs(logs)
    if measure_forward and counters["forward_batches"]:
        result["forward_ms_per_batch_per_rank"] = 1000.0 * counters["forward_seconds"] / counters["forward_batches"]
        result["forward_ms_per_sample_per_rank"] = 1000.0 * counters["forward_seconds"] / counters["forward_samples"]
    if resolve_architecture(cfg) == "v24_ts_coe" and exact_metrics[""].count == 0:
        raise ValueError("TS-CoE evaluation has no finite hidden supervision targets")
    for suffix in active_exact_metrics:
        for key, value in exact_metrics[suffix].compute().items():
            result[f"{key}{suffix}"] = value
    if coe_quality is not None:
        result.update(coe_quality.compute())
    if routing_metrics is not None:
        result.update(routing_metrics.compute())
    return result
