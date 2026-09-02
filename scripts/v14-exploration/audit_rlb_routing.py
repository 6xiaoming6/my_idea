#!/usr/bin/env python3
"""Audit V14 Top-K routing and prove the legacy hard-load gradient behavior."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.data import FlowNPZDataset, build_loader
from stmoe_imputer.losses import masked_loss, moe_balance_loss
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.routing_metrics import (
    RoutingMetricAccumulator,
    active_routing_scales,
)
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


RATES = ("0.2", "0.4", "0.6", "0.8")
DATASETS = {
    "TaxiBJ": {
        "output_name": "TaxiBJ",
        "train": "data/TaxiBJ/taxibj_train.npz",
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
    },
    "BikeNYC": {
        "output_name": "BikeNYC",
        "train": "data/BikeNYC/bikenyc_train.npz",
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
    },
    "CHAP": {
        "output_name": "CHAP_Beijing",
        "train": "data/CHAP/beijing/chap_beijing_train.npz",
        "val": "data/CHAP/beijing/chap_beijing_val.npz",
        "test": "data/CHAP/beijing/chap_beijing_test.npz",
    },
}
CORE6 = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.8"),
    ("CHAP", "fixed", "0.2"),
    ("CHAP", "random", "0.4"),
)
GRADIENT_CALIBRATION_POINT = ("BikeNYC", "random", "0.2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", choices=("core6", "all24"), default="all24")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--output-dir",
        default="outputs/v14-exploration/diagnostics/rlb_stage0",
    )
    return parser.parse_args()


def _points(scope: str) -> tuple[tuple[str, str, str], ...]:
    if scope == "core6":
        return CORE6
    return tuple(
        (dataset, pattern, rate)
        for dataset in DATASETS
        for pattern in ("fixed", "random")
        for rate in RATES
    )


def _latest_run(dataset: str, pattern: str, rate: str, seed: int) -> Path:
    root = (
        ROOT
        / "outputs/v14-single"
        / DATASETS[dataset]["output_name"]
        / "full/model"
        / pattern
        / f"rate{rate}"
    )
    candidates = []
    for checkpoint in root.glob("*/checkpoints/best.pt"):
        run_dir = checkpoint.parent.parent
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config.get("seed", -1)) == seed:
            candidates.append(run_dir)
    candidates.sort()
    if not candidates:
        raise FileNotFoundError(
            f"No complete V14 seed={seed} checkpoint under {root}"
        )
    return candidates[-1]


def _dataset(cfg: dict, dataset: str, split: str) -> FlowNPZDataset:
    scale_cfg = cfg["data"]["scales"]
    mask_cfg = cfg["data"]["mask"]
    mask_path = Path(mask_cfg[f"{split}_csv"])
    if not mask_path.is_absolute():
        mask_path = ROOT / mask_path
    seed_offset = {"train": 10000, "val": 20000, "test": 30000}[split]
    return FlowNPZDataset(
        ROOT / DATASETS[dataset][split],
        mask_cfg=mask_cfg,
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        seed=int(cfg.get("seed", 42)) + seed_offset,
        mask_csv=mask_path,
    )


def _gradient_check() -> dict[str, object]:
    logits = torch.tensor(
        [
            [4.0, 3.0, 0.0, -1.0],
            [3.5, 2.5, 0.1, -0.5],
            [4.2, 3.2, -0.2, -1.2],
            [3.8, 2.8, 0.2, -0.8],
        ],
        requires_grad=True,
    )
    gates = {"fine": logits.softmax(dim=1)}
    top_indices = gates["fine"].topk(k=2, dim=1).indices
    selected = torch.zeros_like(gates["fine"])
    selected.scatter_(1, top_indices, 1.0)
    masks = {"fine": selected}

    legacy_total, legacy_importance, legacy_load = moe_balance_loss(
        gates,
        masks,
        scale_names=("fine",),
        load_balance_mode="legacy_hard",
    )
    legacy_total_grad = torch.autograd.grad(
        legacy_total, logits, retain_graph=True
    )[0]
    importance_grad = torch.autograd.grad(
        legacy_importance, logits, retain_graph=True
    )[0]

    switch_total, _, switch_load = moe_balance_loss(
        gates,
        masks,
        scale_names=("fine",),
        load_balance_mode="switch_topk",
    )
    switch_load_grad = torch.autograd.grad(
        switch_load, logits, retain_graph=True
    )[0]

    uniform_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    _, _, uniform_switch_load = moe_balance_loss(
        gates,
        {"fine": uniform_mask},
        scale_names=("fine",),
        load_balance_mode="switch_topk",
    )
    uniform_grad = torch.autograd.grad(uniform_switch_load, logits)[0]
    legacy_difference = float(
        (legacy_total_grad - importance_grad).abs().max()
    )
    switch_grad_max = float(switch_load_grad.abs().max())
    uniform_grad_max = float(uniform_grad.abs().max())
    return {
        "selected_mask_requires_grad": selected.requires_grad,
        "legacy_load_requires_grad": legacy_load.requires_grad,
        "legacy_load_value": float(legacy_load),
        "legacy_total_vs_importance_max_abs_grad_difference": legacy_difference,
        "switch_load_requires_grad": switch_load.requires_grad,
        "switch_load_value": float(switch_load),
        "switch_load_max_abs_router_gradient": switch_grad_max,
        "uniform_switch_load_value": float(uniform_switch_load),
        "uniform_switch_max_abs_router_gradient": uniform_grad_max,
        "legacy_proof_passed": (
            not selected.requires_grad
            and not legacy_load.requires_grad
            and legacy_difference < 1e-12
        ),
        "switch_gradient_passed": (
            switch_load.requires_grad
            and switch_grad_max > 1e-8
            and uniform_grad_max < 1e-7
        ),
        "unused_switch_total_value": float(switch_total),
    }


def _gradient_norm(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared_norm = torch.zeros((), device=loss.device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared_norm += gradient.detach().double().square().sum()
    return float(squared_norm.sqrt().cpu())


def _real_batch_gradient_check(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    scale_names: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    """Calibrate auxiliary router gradients against one real validation batch."""
    try:
        batch = next(iter(loader))
    except StopIteration as error:
        raise RuntimeError(f"Empty loader during real-batch gradient check: {label}") from error
    batch = move_batch_to_device(batch, device)
    model.eval()
    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    router_parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(token in name for token in ("router_f", "router_m", "router_c"))
    )
    if not router_parameters:
        raise RuntimeError("No trainable router parameters found")
    main = masked_loss(
        outputs["x_hat_main"],
        batch["x_f_gt"],
        batch["m_f"],
        loss_type=cfg["loss"].get("type", "smooth_l1"),
    )
    _, importance, switch_load = moe_balance_loss(
        outputs["gates"],
        outputs.get("selected_masks"),
        use_load_balance=True,
        scale_names=scale_names,
        load_balance_mode="switch_topk",
    )
    main_norm = _gradient_norm(main, router_parameters, retain_graph=True)
    importance_norm = _gradient_norm(
        importance, router_parameters, retain_graph=True
    )
    switch_norm = _gradient_norm(
        switch_load, router_parameters, retain_graph=False
    )
    weight_ratios = {}
    for weight in (1e-4, 1e-3, 1e-2):
        key = f"{weight:.0e}"
        weight_ratios[key] = (
            weight * switch_norm / main_norm if main_norm > 0 else None
        )
    return {
        "label": label,
        "batch_size": int(batch["x_f_gt"].shape[0]),
        "active_scales": list(scale_names),
        "router_parameter_count": sum(
            parameter.numel() for parameter in router_parameters
        ),
        "main_loss": float(main.detach().cpu()),
        "importance_loss": float(importance.detach().cpu()),
        "switch_load_loss": float(switch_load.detach().cpu()),
        "main_router_grad_norm": main_norm,
        "importance_router_grad_norm": importance_norm,
        "switch_router_grad_norm": switch_norm,
        "weighted_switch_to_main_router_grad_ratio": weight_ratios,
        "all_finite": all(
            torch.isfinite(torch.tensor(value))
            for value in (main_norm, importance_norm, switch_norm)
        ),
    }


@torch.no_grad()
def _audit_run(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    scale_names: tuple[str, ...],
    desc: str,
) -> dict[str, float]:
    accumulator = RoutingMetricAccumulator(scale_names)
    model.eval()
    for batch in tqdm(loader, desc=desc, leave=False):
        outputs = model(move_batch_to_device(batch, device))
        accumulator.update(
            outputs.get("gates", {}), outputs.get("selected_masks")
        )
    metrics = accumulator.compute()
    if not metrics:
        raise RuntimeError(f"No Top-K routing metrics were produced for {desc}")
    return metrics


def _dataset_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    result: list[dict[str, object]] = []
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith("routing_") and isinstance(value, (int, float))
        }
    )
    for dataset, dataset_rows in grouped.items():
        summary: dict[str, object] = {
            "dataset": dataset,
            "runs": len(dataset_rows),
        }
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in dataset_rows
                if isinstance(row.get(metric), (int, float))
            ]
            if values:
                summary[metric] = sum(values) / len(values)
        result.append(summary)
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report(
    rows: list[dict[str, object]],
    gradient: dict[str, object],
    scope: str,
    split: str,
    seed: int,
    created_at: str,
) -> str:
    lines = [
        "# V14 RLB 阶段 0 路由审计",
        "",
        f"- 生成时间：{created_at}",
        f"- 范围：{scope}",
        f"- 数据划分：{split}",
        f"- 模型种子：{seed}",
        "- 本审计只读取已有 V14 checkpoint，不训练、不覆盖任何参数。",
        "",
        "## 梯度证明",
        "",
        f"- legacy hard load 无梯度：{gradient['legacy_proof_passed']}",
        (
            "- legacy 总梯度与 importance-only 最大绝对差："
            f"{gradient['legacy_total_vs_importance_max_abs_grad_difference']:.3e}"
        ),
        f"- switch_topk 产生有效路由梯度：{gradient['switch_gradient_passed']}",
        (
            "- switch_topk 最大绝对路由梯度："
            f"{gradient['switch_load_max_abs_router_gradient']:.3e}"
        ),
        (
            "- 均匀 hard load 下最大绝对额外梯度："
            f"{gradient['uniform_switch_max_abs_router_gradient']:.3e}"
        ),
        "",
        "### 真实数据 batch 梯度量级",
        "",
        (
            f"- 校准点：{gradient['real_batch']['label']}，"
            f"batch size={gradient['real_batch']['batch_size']}"
        ),
        (
            "- 主损失/router、importance/router、RLB/router 梯度范数："
            f"{gradient['real_batch']['main_router_grad_norm']:.3e} / "
            f"{gradient['real_batch']['importance_router_grad_norm']:.3e} / "
            f"{gradient['real_batch']['switch_router_grad_norm']:.3e}"
        ),
        (
            "- 加权 RLB/主损失 router 梯度比："
            + ", ".join(
                f"{weight}={ratio:.3e}"
                for weight, ratio in gradient["real_batch"][
                    "weighted_switch_to_main_router_grad_ratio"
                ].items()
                if ratio is not None
            )
        ),
        "",
        "## 每个实验的聚合路由状态",
        "",
        "| 数据集 | 模式 | 缺失率 | 聚合CV | 尺度平均CV | 最差尺度CV | 选择熵 | Dead率 | Always率 | Soft-hard差距 | Top-K margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {pattern} | {rate} | "
            "{routing_all_hard_load_cv:.4f} | "
            "{routing_scales_mean_hard_load_cv:.4f} | "
            "{routing_scales_max_hard_load_cv:.4f} | "
            "{routing_all_selection_entropy:.4f} | "
            "{routing_all_dead_expert_rate:.4f} | "
            "{routing_all_always_selected_rate:.4f} | "
            "{routing_all_soft_hard_l1_gap:.4f} | "
            "{routing_all_topk_boundary_margin:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 判读原则",
            "",
            "- 路由均衡指标只能验证机制，不能代替 MAE/RMSE。",
            "- 阶段 1 只有在验证 MAE 安全且 hard load CV 实质下降时才能晋级。",
            "- 若路由变均衡但精度不提升，应认为当前集中是有益专门化并终止 RLB。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = (
        torch.device(f"cuda:{args.gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient = _gradient_check()
    rows: list[dict[str, object]] = []
    points = _points(args.scope)

    for index, (dataset, pattern, rate) in enumerate(points, start=1):
        print(
            f"[{index}/{len(points)}] AUDIT {dataset} {pattern}@{rate} "
            f"split={args.split}",
            flush=True,
        )
        run_dir = _latest_run(dataset, pattern, rate, args.seed)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cfg["data"]["num_workers"] = 0
        model = DualBranchSTImputer.from_config(cfg).to(device)
        checkpoint = load_checkpoint(
            run_dir / "checkpoints/best.pt", model, map_location=device
        )
        dataset_object = _dataset(cfg, dataset, args.split)
        loader = build_loader(dataset_object, cfg, shuffle=False)
        scale_mode = cfg["model"]["main"].get(
            "scale_mode",
            cfg["model"].get("scale_mode", "fine_mid_coarse"),
        )
        scale_names = active_routing_scales(scale_mode)
        metrics = _audit_run(
            model,
            loader,
            device,
            scale_names,
            f"{dataset} {pattern}@{rate}",
        )
        rows.append(
            {
                "dataset": dataset,
                "pattern": pattern,
                "rate": rate,
                "split": args.split,
                "seed": int(cfg.get("seed", 42)),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "run_dir": str(run_dir.relative_to(ROOT)),
                **metrics,
            }
        )
        del model, checkpoint, dataset_object, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gradient_dataset, gradient_pattern, gradient_rate = GRADIENT_CALIBRATION_POINT
    gradient_run = _latest_run(
        gradient_dataset, gradient_pattern, gradient_rate, args.seed
    )
    gradient_cfg = json.loads(
        (gradient_run / "config.json").read_text(encoding="utf-8")
    )
    gradient_cfg["data"]["num_workers"] = 0
    gradient_model = DualBranchSTImputer.from_config(gradient_cfg).to(device)
    load_checkpoint(
        gradient_run / "checkpoints/best.pt",
        gradient_model,
        map_location=device,
    )
    gradient_dataset_object = _dataset(
        gradient_cfg, gradient_dataset, args.split
    )
    gradient_loader = build_loader(
        gradient_dataset_object, gradient_cfg, shuffle=False
    )
    gradient_scale_mode = gradient_cfg["model"]["main"].get(
        "scale_mode",
        gradient_cfg["model"].get("scale_mode", "fine_mid_coarse"),
    )
    gradient["real_batch"] = _real_batch_gradient_check(
        gradient_model,
        gradient_loader,
        device,
        gradient_cfg,
        active_routing_scales(gradient_scale_mode),
        (
            f"{gradient_dataset} {gradient_pattern}@{gradient_rate} "
            f"{args.split}"
        ),
    )
    del gradient_model, gradient_dataset_object, gradient_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_rows = _dataset_summary(rows)
    per_run_path = output_dir / f"{args.scope}_{args.split}_per_run.csv"
    per_dataset_path = output_dir / f"{args.scope}_{args.split}_per_dataset.csv"
    gradient_path = output_dir / "routing_gradient_check.json"
    report_path = output_dir / f"{args.scope}_{args.split}_report.md"
    _write_csv(per_run_path, rows)
    _write_csv(per_dataset_path, dataset_rows)
    gradient_path.write_text(
        json.dumps(gradient, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(
        _report(rows, gradient, args.scope, args.split, args.seed, created_at),
        encoding="utf-8",
    )
    for path in (per_run_path, per_dataset_path, gradient_path, report_path):
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
