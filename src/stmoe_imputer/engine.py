from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from .losses import compute_main_stage_loss
from .metrics import masked_metrics
from .utils.device import move_batch_to_device


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    groups = []
    main_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith("main_branch.")
    ]
    other_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("main_branch.")
    ]
    if main_params:
        groups.append({"params": main_params, "lr": train_cfg["lr_main"]})
    if other_params:
        groups.append({"params": other_params, "lr": train_cfg.get("lr_aux", train_cfg["lr_main"])})
    return torch.optim.AdamW(groups, weight_decay=train_cfg.get("weight_decay", 0.0))


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
        logs["effective_route_ratio"].append(gamma_value)


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
            loss, loss_dict = compute_main_stage_loss(outputs, batch, cfg)
        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip_norm")
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        metrics = masked_metrics(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"])
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs)
        progress.set_postfix(loss=logs["loss"][-1], mae=logs["mae"][-1], rmse=logs["rmse"][-1])
    return _mean_logs(logs)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()
    logs: dict[str, list[float]] = defaultdict(list)
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        _, loss_dict = compute_main_stage_loss(outputs, batch, cfg)
        metrics = masked_metrics(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"])
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs)
    return _mean_logs(logs)
