from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from .losses import compute_main_stage_loss
from .metrics import masked_metrics
from .utils.device import move_batch_to_device


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    base_lr = train_cfg["lr_main"]
    aux_lr = train_cfg.get("lr_aux", base_lr)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    gate_lr_mult = train_cfg.get("gate_lr_mult", 1.0)
    scalar_lr_mult = train_cfg.get("scalar_lr_mult", 2.0)

    grouped: dict[str, dict] = {
        "main": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "gate": {"params": [], "lr": base_lr * gate_lr_mult, "weight_decay": 0.0},
        "scalar": {"params": [], "lr": base_lr * scalar_lr_mult, "weight_decay": 0.0},
        "no_decay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "other": {"params": [], "lr": aux_lr, "weight_decay": weight_decay},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_l = name.lower()
        if any(token in name_l for token in ("route_gamma", "shared_gamma", "shared_input_adapter.beta")):
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
            T_max=cfg["train"]["epochs"],
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    if sched_type == "warmup_cosine":
        return WarmupCosineLR(
            optimizer,
            max_epochs=cfg["train"]["epochs"],
            warmup_epochs=sched_cfg.get("warmup_epochs", 5),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    raise ValueError(f"Unknown scheduler type: {sched_type}")


def _mean_logs(accumulator: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(values) / max(1, len(values)) for key, values in accumulator.items()}


def _append_model_diagnostics(logs: dict[str, list[float]], outputs: dict) -> None:
    scale_gate = outputs.get("gates", {}).get("scale_gate")
    if scale_gate is not None:
        labels = ("f", "m", "c")
        for idx, label in enumerate(labels):
            values = scale_gate[:, idx]
            logs[f"scale_gate_{label}_mean"].append(float(values.mean().detach().cpu()))
            logs[f"scale_gate_{label}_std"].append(float(values.std(unbiased=False).detach().cpu()))

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

    features = outputs.get("features", {})
    h_shared = features.get("h_shared") if isinstance(features, dict) else None
    h_route_proj = features.get("h_route_proj") if isinstance(features, dict) else None
    if h_shared is not None and h_route_proj is not None:
        shared_norm = h_shared.detach().pow(2).mean().sqrt()
        route_norm = h_route_proj.detach().pow(2).mean().sqrt()
        logs["effective_shared_norm"].append(float(shared_norm.cpu()))
        logs["effective_route_norm"].append(float(route_norm.cpu()))
        logs["effective_route_ratio"].append(float((route_norm / shared_norm.clamp_min(1e-6)).cpu()))


def _append_lr_logs(logs: dict[str, list[float]], optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        name = group.get("name", "group")
        logs[f"lr_group_{name}"].append(float(group["lr"]))


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    epoch: int,
) -> dict[str, float]:
    model.train()
    logs: dict[str, list[float]] = defaultdict(list)
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
            loss, loss_dict = compute_main_stage_loss(outputs, batch, cfg, epoch=epoch)
        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip_norm")
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        metrics = masked_metrics(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"])
        if outputs.get("x_hat_shared") is not None:
            shared_metrics = masked_metrics(outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"])
            metrics.update({f"{key}_shared_aux": value for key, value in shared_metrics.items()})
        if outputs.get("x_hat_route") is not None:
            route_metrics = masked_metrics(outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"])
            metrics.update({f"{key}_route_aux": value for key, value in route_metrics.items()})
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs)
        _append_lr_logs(logs, optimizer)
        progress.set_postfix(loss=logs["loss"][-1], mae=logs["mae"][-1], rmse=logs["rmse"][-1])
    return _mean_logs(logs)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    desc: str = "eval",
    epoch: int | None = None,
) -> dict[str, float]:
    model.eval()
    logs: dict[str, list[float]] = defaultdict(list)
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        _, loss_dict = compute_main_stage_loss(outputs, batch, cfg, epoch=epoch)
        metrics = masked_metrics(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"])
        if outputs.get("x_hat_shared") is not None:
            shared_metrics = masked_metrics(outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"])
            metrics.update({f"{key}_shared_aux": value for key, value in shared_metrics.items()})
        if outputs.get("x_hat_route") is not None:
            route_metrics = masked_metrics(outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"])
            metrics.update({f"{key}_route_aux": value for key, value in route_metrics.items()})
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs)
    return _mean_logs(logs)
