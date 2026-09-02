#!/usr/bin/env python3
"""Audit V14 stage-supervision gradients on existing validation checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_gate_calibration as checkpoint_data
from stmoe_imputer.data import build_loader
from stmoe_imputer.losses import masked_loss, multi_resolution_supervision_loss
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", choices=("core6", "all24"), default="all24")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="outputs/v14-exploration/diagnostics/multiscale_gradients",
    )
    return parser.parse_args()


def _gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _gradient_stats(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
) -> tuple[float, float, float]:
    device = next(value.device for value in first if value is not None)
    dot = torch.zeros((), device=device, dtype=torch.float64)
    norm_first = torch.zeros_like(dot)
    norm_second = torch.zeros_like(dot)
    common = 0
    for left, right in zip(first, second):
        if left is None or right is None:
            continue
        common += 1
        left_f = left.detach().double()
        right_f = right.detach().double()
        dot += (left_f * right_f).sum()
        norm_first += left_f.square().sum()
        norm_second += right_f.square().sum()
    first_value = float(norm_first.sqrt().cpu())
    second_value = float(norm_second.sqrt().cpu())
    if common == 0 or first_value <= 0.0 or second_value <= 0.0:
        return float("nan"), first_value, second_value
    cosine = float((dot / (norm_first.sqrt() * norm_second.sqrt())).cpu())
    return cosine, first_value, second_value


def _mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _audit_point(
    dataset: str,
    pattern: str,
    rate: str,
    seed: int,
    batches: int,
    device: torch.device,
) -> dict:
    run_dir = checkpoint_data._latest_run(dataset, pattern, rate, seed)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg["data"]["num_workers"] = 0
    model = DualBranchSTImputer.from_config(cfg).to(device).eval()
    checkpoint = load_checkpoint(run_dir / "checkpoints/best.pt", model, map_location=device)
    loader = build_loader(checkpoint_data._dataset(cfg, dataset), cfg, shuffle=False)
    parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            "main_branch.refiner" in name
            or "main_branch.controller" in name
        )
    )
    if not parameters:
        raise RuntimeError("No V14 refiner/controller parameters found")
    loss_type = cfg["loss"].get("type", "smooth_l1")
    lambda_mid = float(cfg["loss"].get("lambda_v14_mid", 0.0))
    lambda_coarse = float(cfg["loss"].get("lambda_v14_coarse", 0.0))
    values: dict[str, list[float]] = {
        key: []
        for key in (
            "main_loss",
            "mid_loss",
            "coarse_loss",
            "aux_value_ratio",
            "mid_cosine",
            "coarse_cosine",
            "mid_weighted_grad_ratio",
            "coarse_weighted_grad_ratio",
        )
    }
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        batch = move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(batch)
        main = masked_loss(
            outputs["x_hat_main"],
            batch["x_f_gt"],
            batch["m_f"],
            loss_type=loss_type,
        )
        scale_cfg = cfg["data"]["scales"]
        mid, coarse = multi_resolution_supervision_loss(
            outputs,
            batch["x_f_gt"],
            batch["m_m"],
            batch["m_c"],
            fine_to_mid=scale_cfg["fine_to_mid"],
            fine_to_coarse=scale_cfg["fine_to_coarse"],
            pooling_mode=scale_cfg.get("pooling_mode", "avg"),
            loss_type=loss_type,
        )
        main_grad = _gradients(main, parameters, retain_graph=True)
        mid_grad = _gradients(mid, parameters, retain_graph=True)
        coarse_grad = _gradients(coarse, parameters, retain_graph=False)
        mid_cosine, main_norm_mid, mid_norm = _gradient_stats(main_grad, mid_grad)
        coarse_cosine, main_norm_coarse, coarse_norm = _gradient_stats(
            main_grad, coarse_grad
        )
        main_value = float(main.detach())
        mid_value = float(mid.detach())
        coarse_value = float(coarse.detach())
        values["main_loss"].append(main_value)
        values["mid_loss"].append(mid_value)
        values["coarse_loss"].append(coarse_value)
        values["aux_value_ratio"].append(
            (lambda_mid * mid_value + lambda_coarse * coarse_value)
            / max(main_value, 1e-12)
        )
        values["mid_cosine"].append(mid_cosine)
        values["coarse_cosine"].append(coarse_cosine)
        values["mid_weighted_grad_ratio"].append(
            lambda_mid * mid_norm / max(main_norm_mid, 1e-12)
        )
        values["coarse_weighted_grad_ratio"].append(
            lambda_coarse * coarse_norm / max(main_norm_coarse, 1e-12)
        )
    row = {
        "dataset": dataset,
        "pattern": pattern,
        "rate": rate,
        "run_dir": str(run_dir),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "batches": min(batches, len(loader)),
        "lambda_mid": lambda_mid,
        "lambda_coarse": lambda_coarse,
    }
    row.update({key: _mean_finite(value) for key, value in values.items()})
    del model, loader
    torch.cuda.empty_cache()
    return row


def _markdown(rows: list[dict], scope: str, created_at: str) -> str:
    lines = [
        "# V14 多尺度监督梯度诊断",
        "",
        f"- 生成时间：{created_at}",
        f"- 范围：{scope}",
        "- 参数范围：V14 refiner + safety controller。",
        "- 正梯度余弦表示辅助监督与最终缺失值目标方向一致；负值表示冲突。",
        "",
        "| 数据集 | 模式 | 缺失率 | aux/main值比 | mid余弦 | coarse余弦 | 加权mid梯度比 | 加权coarse梯度比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def number(key: str) -> str:
            value = row[key]
            return f"{value:+.3f}" if math.isfinite(value) else "N/A"
        lines.append(
            f"| {row['dataset']} | {row['pattern']} | {row['rate']} | "
            f"{row['aux_value_ratio']:.3f} | {number('mid_cosine')} | "
            f"{number('coarse_cosine')} | {row['mid_weighted_grad_ratio']:.2f} | "
            f"{row['coarse_weighted_grad_ratio']:.2f} |"
        )
    mid_valid = [row for row in rows if math.isfinite(row["mid_cosine"])]
    coarse_valid = [row for row in rows if math.isfinite(row["coarse_cosine"])]
    lines.extend([
        "",
        "## 汇总",
        "",
        f"- mid 梯度冲突点：{sum(row['mid_cosine'] < 0 for row in mid_valid)}/{len(mid_valid)}。",
        f"- coarse 梯度冲突点：{sum(row['coarse_cosine'] < 0 for row in coarse_valid)}/{len(coarse_valid)}。",
        f"- 加权 mid 梯度相对最终梯度的平均倍数：{_mean_finite([row['mid_weighted_grad_ratio'] for row in rows]):.2f}。",
        f"- 加权 coarse 梯度相对最终梯度的平均倍数：{_mean_finite([row['coarse_weighted_grad_ratio'] for row in rows]):.2f}。",
        "",
        "若辅助梯度在多个点冲突或显著大于最终梯度，应先降低阶段监督强度，不能据此删除多尺度结构本身。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gradient diagnosis")
    points = checkpoint_data._points(args.scope)
    device = torch.device(f"cuda:{args.gpu}")
    rows = []
    for index, (dataset, pattern, rate) in enumerate(points, start=1):
        print(f"[{index}/{len(points)}] {dataset} {pattern}@{rate}", flush=True)
        row = _audit_point(
            dataset, pattern, rate, args.seed, args.batches, device
        )
        rows.append(row)
        print(
            f"  mid cos={row['mid_cosine']:+.3f}, "
            f"weighted grad={row['mid_weighted_grad_ratio']:.2f}x; "
            f"coarse cos={row['coarse_cosine']:+.3f}",
            flush=True,
        )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path = output_dir / f"{args.scope}_summary.csv"
    json_path = output_dir / f"{args.scope}_details.json"
    report_path = output_dir / f"{args.scope}_report.md"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({"created_at": created_at, "scope": args.scope, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(_markdown(rows, args.scope, created_at), encoding="utf-8")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
