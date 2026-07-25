from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from .losses import compute_main_stage_loss
from .metrics import MaskedMetricAccumulator
from .utils.device import move_batch_to_device


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    base_lr = train_cfg["lr_main"]
    aux_lr = train_cfg.get("lr_aux", base_lr)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    model_cfg = cfg.get("model", {})
    architecture = str(model_cfg.get("architecture", "")).lower()
    version = str(model_cfg.get("version", "")).lower()
    is_v18 = architecture == "v18_base_anchored_residual_moe" or version.startswith("v18")
    gate_lr_mult = train_cfg.get("gate_lr_mult", 1.0)
    scalar_lr_mult = train_cfg.get("scalar_lr_mult", 2.0)
    v14_lr = train_cfg.get("lr_v14", base_lr)

    grouped: dict[str, dict] = {
        "main": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "gate": {"params": [], "lr": base_lr * gate_lr_mult, "weight_decay": 0.0},
        "scalar": {"params": [], "lr": base_lr * scalar_lr_mult, "weight_decay": 0.0},
        "no_decay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "v14": {"params": [], "lr": v14_lr, "weight_decay": weight_decay},
        "v14_no_decay": {"params": [], "lr": v14_lr, "weight_decay": 0.0},
        "v18_refiner": {
            "params": [],
            "lr": train_cfg.get("lr_v18_refiner", base_lr),
            "weight_decay": weight_decay,
        },
        "v18_controller": {
            "params": [],
            "lr": train_cfg.get("lr_v18_controller", base_lr * 0.5),
            "weight_decay": weight_decay,
        },
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
            )
        )
        is_v18_refiner = is_v18 and "residual_pyramid" in name_l
        is_v18_controller = is_v18 and ".controller." in name_l
        if is_v18_refiner:
            grouped["v18_refiner"]["params"].append(param)
        elif is_v18_controller:
            grouped["v18_controller"]["params"].append(param)
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
    batch_weights = accumulator.get("__batch_size__", [])
    result: dict[str, float] = {}
    for key, values in accumulator.items():
        if key == "__batch_size__":
            continue
        if batch_weights and len(values) == len(batch_weights):
            denominator = max(sum(batch_weights), 1.0)
            result[key] = sum(
                value * weight
                for value, weight in zip(values, batch_weights)
            ) / denominator
        else:
            result[key] = sum(values) / max(1, len(values))
    return result


class _RunningDistribution:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = float("inf")
        self.maximum = -float("inf")
        self.high_count = 0
        self.low_count = 0

    def update(
        self,
        values: torch.Tensor,
        *,
        high_threshold: float,
        low_threshold: float,
    ) -> None:
        values_f = values.detach().double().flatten().cpu()
        if values_f.numel() == 0:
            return
        self.count += values_f.numel()
        self.total += float(values_f.sum())
        self.total_square += float(values_f.square().sum())
        self.minimum = min(self.minimum, float(values_f.min()))
        self.maximum = max(self.maximum, float(values_f.max()))
        self.high_count += int((values_f > high_threshold).sum())
        self.low_count += int((values_f < low_threshold).sum())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": variance ** 0.5,
            "min": self.minimum,
            "max": self.maximum,
            "saturation_high": self.high_count / self.count,
            "saturation_low": self.low_count / self.count,
        }


class _V18EpochDiagnostics:
    def __init__(self) -> None:
        self.rho = {
            "c": _RunningDistribution(),
            "m": _RunningDistribution(),
            "f": _RunningDistribution(),
        }
        self.sample_violations = 0
        self.sample_count = 0
        self.point_violations = 0
        self.point_count = 0
        self.bound_violations = 0
        self.bound_count = 0
        self.bound_max_ratio = 0.0
        self.expert_importance_sum: dict[str, torch.Tensor] = {}
        self.expert_importance_count: dict[str, int] = {}

    @torch.no_grad()
    def update(self, outputs: dict, batch: dict, cfg: dict) -> None:
        if not bool(outputs.get("v18_enabled", False)):
            return
        diagnostics = outputs.get("diagnostics", {}).get("v18", {})
        v18_cfg = cfg.get("model", {}).get("v18", {})
        maxima = {
            "c": float(v18_cfg.get("rho_coarse_max", 0.15)),
            "m": float(v18_cfg.get("rho_mid_max", 0.15)),
            "f": float(v18_cfg.get("rho_fine_max", 0.20)),
        }
        for suffix, maximum in maxima.items():
            values = diagnostics.get(f"rho_{suffix}")
            if torch.is_tensor(values):
                self.rho[suffix].update(
                    values,
                    high_threshold=0.95 * maximum,
                    low_threshold=0.05 * maximum,
                )

        base = outputs.get("x_hat_base")
        final = outputs.get("x_hat_main")
        target = batch.get("x_f_gt")
        mask = batch.get("m_f")
        if all(torch.is_tensor(value) for value in (base, final, target, mask)):
            missing = (1.0 - mask.float()).expand_as(final)
            base_error = (base.detach() - target).abs() * missing
            final_error = (final.detach() - target).abs() * missing
            self.point_violations += int(
                ((final_error > base_error) * missing.bool()).sum().cpu()
            )
            self.point_count += int(missing.sum().cpu())

            missing_per_sample = missing.flatten(1).sum(dim=1).clamp_min(1.0)
            base_sample = base_error.flatten(1).sum(dim=1) / missing_per_sample
            final_sample = final_error.flatten(1).sum(dim=1) / missing_per_sample
            self.sample_violations += int((final_sample > base_sample).sum().cpu())
            self.sample_count += final.shape[0]

        features = outputs.get("features", {})
        residual = features.get("effective_residual")
        scale = features.get("observed_scale_f")
        if torch.is_tensor(residual) and torch.is_tensor(scale):
            bound = maxima["f"] * scale.detach().float()
            ratio = (
                residual.detach().float().abs()
                / bound.expand_as(residual).clamp_min(1e-12)
            )
            self.bound_violations += int((ratio > 1.0 + 1e-6).sum().cpu())
            self.bound_count += ratio.numel()
            self.bound_max_ratio = max(
                self.bound_max_ratio, float(ratio.max().cpu())
            )

        for scale_name in ("fine", "mid", "coarse"):
            gate = outputs.get("gates", {}).get(scale_name)
            if not torch.is_tensor(gate):
                continue
            importance = gate.detach().float().sum(dim=0).cpu()
            if scale_name not in self.expert_importance_sum:
                self.expert_importance_sum[scale_name] = torch.zeros_like(
                    importance
                )
                self.expert_importance_count[scale_name] = 0
            self.expert_importance_sum[scale_name] += importance
            self.expert_importance_count[scale_name] += gate.shape[0]

    def compute(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for suffix, distribution in self.rho.items():
            for statistic, value in distribution.compute().items():
                values[f"v18_rho_{suffix}_{statistic}"] = value
        if self.sample_count:
            values["v18_sample_violation_rate"] = (
                self.sample_violations / self.sample_count
            )
        if self.point_count:
            values["v18_point_violation_rate"] = (
                self.point_violations / self.point_count
            )
        if self.bound_count:
            values["v18_residual_bound_violation_rate"] = (
                self.bound_violations / self.bound_count
            )
            values["v18_residual_bound_max_ratio"] = self.bound_max_ratio
        for scale_name, total in self.expert_importance_sum.items():
            count = max(self.expert_importance_count[scale_name], 1)
            for index, value in enumerate(total / count):
                values[
                    f"expert_importance_{scale_name}_{index}"
                ] = float(value)
        return values


def _update_v18_metric_accumulators(
    accumulators: dict[str, MaskedMetricAccumulator],
    outputs: dict,
    batch: dict,
) -> None:
    if not bool(outputs.get("v18_enabled", False)):
        return
    target = batch["x_f_gt"]
    mask = batch["m_f"]
    for name, prediction in (
        ("v18_base_hidden", outputs.get("x_hat_base")),
        ("v18_probe_hidden", outputs.get("x_hat_probe")),
    ):
        if torch.is_tensor(prediction):
            accumulators.setdefault(
                name, MaskedMetricAccumulator()
            ).update(prediction, target, mask)

    observed_target = batch.get("x_f_obs", target)
    observed_as_missing_mask = 1.0 - mask
    for name, prediction in (
        ("v18_base_observed", outputs.get("x_hat_base")),
        ("v18_final_observed", outputs.get("x_hat_main")),
    ):
        if torch.is_tensor(prediction):
            accumulators.setdefault(
                name, MaskedMetricAccumulator()
            ).update(
                prediction,
                observed_target,
                observed_as_missing_mask,
            )


def _finalize_logs(
    logs: dict[str, list[float]],
    metric_accumulators: dict[str, MaskedMetricAccumulator],
    v18_diagnostics: _V18EpochDiagnostics,
) -> dict[str, float]:
    result = _mean_logs(logs)
    for suffix, accumulator in metric_accumulators.items():
        for metric, value in accumulator.compute().items():
            if suffix.startswith("v18_"):
                key = f"{suffix}_{metric}"
            else:
                key = metric if not suffix else f"{metric}_{suffix}"
            result[key] = value
    if "v18_base_hidden_mae" in result:
        result["v18_final_hidden_mae"] = result["mae"]
        result["v18_final_hidden_rmse"] = result["rmse"]
        result["v18_final_base_mae_gain"] = (
            result["v18_base_hidden_mae"]
            - result["v18_final_hidden_mae"]
        )
        result["v18_final_base_mae_gain_ratio"] = (
            result["v18_final_base_mae_gain"]
            / max(result["v18_base_hidden_mae"], 1e-6)
        )
        result["v18_final_base_rmse_gain"] = (
            result["v18_base_hidden_rmse"]
            - result["v18_final_hidden_rmse"]
        )
    result.update(v18_diagnostics.compute())
    return result


def _append_model_diagnostics(
    logs: dict[str, list[float]],
    outputs: dict,
    cfg: dict | None = None,
) -> None:
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

    v18 = diagnostics.get("v18") if isinstance(diagnostics, dict) else None
    if isinstance(v18, dict):
        # Direction/residual summaries are emitted below under their canonical
        # names.  Skipping them here avoids duplicate ``*_mean`` fields in
        # every train/validation/test record.
        detailed_v18_keys = {
            "direction_c_rms",
            "direction_m_rms",
            "direction_f_rms",
            "direction_f_abs_q95",
            "effective_residual_rms",
            "effective_residual_q95",
            "effective_residual_scale_ratio",
        }
        for key, value in v18.items():
            if (
                key in detailed_v18_keys
                or value is None
                or not torch.is_tensor(value)
            ):
                continue
            value_f = value.detach().float()
            summary_key = f"v18_{key}" if key.endswith("_mean") else f"v18_{key}_mean"
            logs[summary_key].append(float(value_f.mean().cpu()))
            if key in {"rho_c", "rho_m", "rho_f"}:
                logs[f"v18_{key}_std"].append(
                    float(value_f.std(unbiased=False).cpu())
                )
                logs[f"v18_{key}_min"].append(float(value_f.min().cpu()))
                logs[f"v18_{key}_max"].append(float(value_f.max().cpu()))

        v18_cfg = (cfg or {}).get("model", {}).get("v18", {})
        for suffix, default_max in (("c", 0.15), ("m", 0.15), ("f", 0.20)):
            rho = v18.get(f"rho_{suffix}")
            if rho is None or not torch.is_tensor(rho):
                continue
            rho_f = rho.detach().float()
            rho_max = float(
                v18_cfg.get(
                    {
                        "c": "rho_coarse_max",
                        "m": "rho_mid_max",
                        "f": "rho_fine_max",
                    }[suffix],
                    default_max,
                )
            )
            logs[f"v18_rho_{suffix}_saturation_high"].append(
                float((rho_f > 0.95 * rho_max).float().mean().cpu())
            )
            logs[f"v18_rho_{suffix}_saturation_low"].append(
                float((rho_f < 0.05 * rho_max).float().mean().cpu())
            )

        features = outputs.get("features", {})
        if isinstance(features, dict):
            for scale in ("c", "m", "f"):
                direction = features.get(f"direction_{scale}")
                if direction is None or not torch.is_tensor(direction):
                    continue
                direction_f = direction.detach().float()
                direction_rms_values = v18.get(f"direction_{scale}_rms")
                direction_rms = (
                    direction_rms_values.detach().float().mean()
                    if torch.is_tensor(direction_rms_values)
                    else direction_f.square().mean().sqrt()
                )
                logs[f"v18_direction_{scale}_rms"].append(
                    float(direction_rms.cpu())
                )
                if scale == "f":
                    direction_q95 = v18.get("direction_f_abs_q95")
                    if torch.is_tensor(direction_q95):
                        direction_q95 = direction_q95.detach().float().mean()
                    else:
                        direction_q95 = torch.quantile(
                            direction_f.abs().flatten(), 0.95
                        )
                    logs["v18_direction_f_abs_q95"].append(
                        float(direction_q95.cpu())
                    )

            residual = features.get("effective_residual")
            if residual is not None and torch.is_tensor(residual):
                residual_f = residual.detach().float()
                residual_rms_values = v18.get("effective_residual_rms")
                residual_rms = (
                    residual_rms_values.detach().float().mean()
                    if torch.is_tensor(residual_rms_values)
                    else residual_f.square().mean().sqrt()
                )
                logs["v18_effective_residual_rms"].append(
                    float(residual_rms.cpu())
                )
                residual_q95 = v18.get("effective_residual_q95")
                if torch.is_tensor(residual_q95):
                    residual_q95 = residual_q95.detach().float().mean()
                else:
                    residual_q95 = torch.quantile(
                        residual_f.abs().flatten(), 0.95
                    )
                logs["v18_effective_residual_abs_q95"].append(
                    float(residual_q95.cpu())
                )
                residual_scale_ratio = v18.get(
                    "effective_residual_scale_ratio"
                )
                if torch.is_tensor(residual_scale_ratio):
                    logs["v18_effective_residual_over_observed_scale"].append(
                        float(residual_scale_ratio.detach().float().mean().cpu())
                    )
                else:
                    scale_f = v18.get("scale_f_mean")
                    if scale_f is not None and torch.is_tensor(scale_f):
                        scale_mean = scale_f.detach().float().mean().clamp_min(1e-6)
                        logs["v18_effective_residual_over_observed_scale"].append(
                            float((residual_rms / scale_mean).cpu())
                        )

    if isinstance(v14, dict) or isinstance(v18, dict):
        selected_masks = outputs.get("selected_masks")
        if not isinstance(selected_masks, dict):
            selected_masks = {}
        for scale in ("fine", "mid", "coarse"):
            gate = outputs.get("gates", {}).get(scale)
            if gate is None or not torch.is_tensor(gate):
                continue
            gate_f = gate.detach().float()
            entropy = -(gate_f * gate_f.clamp_min(1e-8).log()).sum(dim=1)
            logs[f"expert_entropy_{scale}"].append(float(entropy.mean().cpu()))
            selected = selected_masks.get(scale)
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
    metric_accumulators = {"": MaskedMetricAccumulator()}
    v18_epoch_diagnostics = _V18EpochDiagnostics()
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        logs["__batch_size__"].append(float(batch["x_f_gt"].shape[0]))
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

        metrics = metric_accumulators[""].update(
            outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"]
        )
        if outputs.get("x_hat_shared") is not None:
            shared_metrics = metric_accumulators.setdefault(
                "shared_aux", MaskedMetricAccumulator()
            ).update(
                outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"]
            )
            metrics.update({f"{key}_shared_aux": value for key, value in shared_metrics.items()})
        if outputs.get("x_hat_route") is not None:
            route_metrics = metric_accumulators.setdefault(
                "route_aux", MaskedMetricAccumulator()
            ).update(
                outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"]
            )
            metrics.update({f"{key}_route_aux": value for key, value in route_metrics.items()})
        _update_v18_metric_accumulators(
            metric_accumulators, outputs, batch
        )
        v18_epoch_diagnostics.update(outputs, batch, cfg)
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs, cfg)
        _append_lr_logs(logs, optimizer)
        progress.set_postfix(loss=logs["loss"][-1], mae=logs["mae"][-1], rmse=logs["rmse"][-1])
    return _finalize_logs(
        logs, metric_accumulators, v18_epoch_diagnostics
    )


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
    metric_accumulators = {"": MaskedMetricAccumulator()}
    v18_epoch_diagnostics = _V18EpochDiagnostics()
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch_to_device(batch, device)
        logs["__batch_size__"].append(float(batch["x_f_gt"].shape[0]))
        outputs = model(batch)
        _, loss_dict = compute_main_stage_loss(outputs, batch, cfg, epoch=epoch)
        metrics = metric_accumulators[""].update(
            outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"]
        )
        if outputs.get("x_hat_shared") is not None:
            shared_metrics = metric_accumulators.setdefault(
                "shared_aux", MaskedMetricAccumulator()
            ).update(
                outputs["x_hat_shared"], batch["x_f_gt"], batch["m_f"]
            )
            metrics.update({f"{key}_shared_aux": value for key, value in shared_metrics.items()})
        if outputs.get("x_hat_route") is not None:
            route_metrics = metric_accumulators.setdefault(
                "route_aux", MaskedMetricAccumulator()
            ).update(
                outputs["x_hat_route"], batch["x_f_gt"], batch["m_f"]
            )
            metrics.update({f"{key}_route_aux": value for key, value in route_metrics.items()})
        _update_v18_metric_accumulators(
            metric_accumulators, outputs, batch
        )
        v18_epoch_diagnostics.update(outputs, batch, cfg)
        for key, value in {**loss_dict, **metrics}.items():
            logs[key].append(float(value.detach().cpu()))
        _append_model_diagnostics(logs, outputs, cfg)
    return _finalize_logs(
        logs, metric_accumulators, v18_epoch_diagnostics
    )
