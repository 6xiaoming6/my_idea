from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from .losses import compute_main_stage_loss, supervision_mask
from .metrics import MaskedMetricAccumulator, masked_metrics
from .routing_metrics import CoERoutingMetricAccumulator, RoutingMetricAccumulator, active_routing_scales
from .models.registry import resolve_architecture
from .utils.device import move_batch_to_device


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


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.LRScheduler | None:
    sched_cfg = cfg["train"].get("scheduler", {})
    sched_type = sched_cfg.get("type", "none")
    if sched_type == "none":
        return None
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("total_epochs", cfg["train"]["epochs"]),
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


def _mean_logs(accumulator: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(values) / max(1, len(values)) for key, values in accumulator.items()}


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
) -> dict[str, float]:
    model.train()
    if hasattr(getattr(loader, 'dataset', None), 'set_epoch'):
        loader.dataset.set_epoch(epoch)
    if hasattr(model.main_branch, "set_routing_epoch"):
        model.main_branch.set_routing_epoch(epoch)
    is_coe = resolve_architecture(cfg) == "v24_ts_coe"
    logs: dict[str, list[float]] = defaultdict(list)
    exact_metrics = {
        "": MaskedMetricAccumulator(),
        "_shared_aux": MaskedMetricAccumulator(),
        "_route_aux": MaskedMetricAccumulator(),
    }
    active_exact_metrics = {""}
    routing_metrics = _routing_accumulator(cfg)
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
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=True)
    for batch_index, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        if is_coe and not bool(supervision_mask(
            batch["x_f_gt"], batch["m_f"], batch.get("target_mask")
        ).any()):
            skipped_empty_batches += 1
            continue
        seen_samples += int(batch["x_f_gt"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
            loss, loss_dict = compute_main_stage_loss(outputs, batch, cfg, epoch=epoch)
        diagnostic_every = cfg["train"].get("router_grad_diagnostic_every", 0)
        if is_coe and diagnostic_every and batch_index % diagnostic_every == 0:
            parameters = [p for router in model.main_branch.routers for p in router.parameters()]
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
            for step, router in enumerate(model.main_branch.routers):
                gradients = [p.grad.detach().float().square().sum()
                             for p in router.parameters() if p.grad is not None]
                if gradients:
                    logs[f"coe_step{step + 1}_router_grad_norm"].append(
                        float(torch.stack(gradients).sum().sqrt().cpu())
                    )
            for name, expert in zip(
                model.main_branch.expert_names, model.main_branch.routed_experts()
            ):
                gradients = [p.grad.detach().float().square().sum()
                             for p in expert.parameters() if p.grad is not None]
                if gradients:
                    norm = float(torch.stack(gradients).sum().sqrt().cpu())
                    logs[f"coe_expert_{name}_grad_norm"].append(norm)
                    logs[f"coe_expert_{name}_finite_nonzero_gradient_batch_fraction"].append(float(0 < norm < float("inf")))
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            skipped_amp_steps += 1
        else:
            optimizer_steps += 1

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
        if is_coe and 'mask_family' in batch:
            # Labels are introduced after forward and used only for diagnostics.
            outputs['coe']['mask_family'] = batch['mask_family']
        _update_routing_metrics(routing_metrics, outputs)
        _append_lr_logs(logs, optimizer)
        progress.set_postfix(loss=logs["loss"][-1], mae=logs["mae"][-1], rmse=logs["rmse"][-1])
    result = _mean_logs(logs)
    if is_coe:
        result.update({
            "train_optimizer_steps": float(optimizer_steps),
            "train_seen_samples": float(seen_samples),
            "train_skipped_empty_batches": float(skipped_empty_batches),
            "train_skipped_amp_steps": float(skipped_amp_steps),
            "train_amp_scale_start": float(scale_start),
            "train_amp_scale_end": float(scaler.get_scale()),
        })
    if is_coe and exact_metrics[""].count == 0:
        raise ValueError("TS-CoE training has no finite hidden supervision targets")
    for suffix in active_exact_metrics:
        for key, value in exact_metrics[suffix].compute().items():
            result[f"{key}{suffix}"] = value
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
    for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
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
        _update_routing_metrics(routing_metrics, outputs)
    result = _mean_logs(logs)
    if resolve_architecture(cfg) == "v24_ts_coe" and exact_metrics[""].count == 0:
        raise ValueError("TS-CoE evaluation has no finite hidden supervision targets")
    for suffix in active_exact_metrics:
        for key, value in exact_metrics[suffix].compute().items():
            result[f"{key}{suffix}"] = value
    if routing_metrics is not None:
        result.update(routing_metrics.compute())
    return result
