#!/usr/bin/env python3
"""Diagnose whether V14's target-free final gate predicts missing-region gain.

This script is read-only: it loads existing seed-42 V14 checkpoints and never
changes a checkpoint.  All calibration statistics are computed per sample on
the validation split.  Hidden targets are used only for diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.data import FlowNPZDataset, build_loader
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


RATES = ("0.2", "0.4", "0.6", "0.8")
DATASETS = {
    "TaxiBJ": {
        "output_name": "TaxiBJ",
        "val": "data/TaxiBJ/taxibj_val.npz",
    },
    "BikeNYC": {
        "output_name": "BikeNYC",
        "val": "data/BikeNYC/bikenyc_val.npz",
    },
    "CHAP": {
        "output_name": "CHAP_Beijing",
        "val": "data/CHAP/beijing/chap_beijing_val.npz",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", choices=("core6", "all24"), default="core6")
    parser.add_argument(
        "--output-dir",
        default="outputs/v14-exploration/diagnostics/gate_calibration",
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
    candidates: list[Path] = []
    for checkpoint in root.glob("*/checkpoints/best.pt"):
        run_dir = checkpoint.parent.parent
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if int(cfg.get("seed", -1)) == seed:
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No V14 seed={seed} checkpoint under {root}")
    return sorted(candidates)[-1]


def _dataset(cfg: dict, dataset: str) -> FlowNPZDataset:
    scale_cfg = cfg["data"]["scales"]
    mask_cfg = cfg["data"]["mask"]
    mask_path = Path(mask_cfg["val_csv"])
    if not mask_path.is_absolute():
        mask_path = ROOT / mask_path
    return FlowNPZDataset(
        ROOT / DATASETS[dataset]["val"],
        mask_cfg=mask_cfg,
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        seed=int(cfg.get("seed", 42)) + 20000,
        mask_csv=mask_path,
    )


def _sample_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    missing: torch.Tensor,
) -> torch.Tensor:
    weight = missing.expand_as(prediction).float()
    count = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction.float() - target.float()).abs() * weight).flatten(1).sum(1) / count


def _oracle_mse_alpha(
    x_base: torch.Tensor,
    delta: torch.Tensor,
    target: torch.Tensor,
    missing: torch.Tensor,
    alpha_max: float,
) -> torch.Tensor:
    weight = missing.expand_as(delta).float()
    residual = target.float() - x_base.float()
    delta_f = delta.float()
    numerator = (residual * delta_f * weight).flatten(1).sum(dim=1)
    denominator = (delta_f.square() * weight).flatten(1).sum(dim=1).clamp_min(1e-12)
    return (numerator / denominator).clamp(0.0, alpha_max)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # Average tied ranks.  This is intentionally dependency-free.
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2:
        return float("nan")
    return _pearson(_ranks(left), _ranks(right))


def _evaluate(model, loader, device: torch.device, alpha_max: float, desc: str) -> dict:
    values: dict[str, list[float]] = {
        key: []
        for key in (
            "observed_advantage",
            "relative_observed_advantage",
            "hidden_ctf_advantage",
            "hidden_final_advantage",
            "alpha_final",
            "oracle_alpha",
            "base_mae",
            "ctf_mae",
            "final_mae",
        )
    }
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            x_base = outputs.get("x_hat_base")
            x_ctf = outputs.get("x_hat_ctf")
            x_final = outputs["x_hat_final"]
            if x_base is None or x_ctf is None:
                raise RuntimeError("Expected V14 x_hat_base and x_hat_ctf outputs")
            missing = 1.0 - batch["m_f"].float()
            target = batch["x_f_gt"]
            base_mae = _sample_mae(x_base, target, missing)
            ctf_mae = _sample_mae(x_ctf, target, missing)
            final_mae = _sample_mae(x_final, target, missing)
            oracle = _oracle_mse_alpha(
                x_base,
                x_ctf - x_base,
                target,
                missing,
                alpha_max,
            )
            v14 = outputs.get("diagnostics", {}).get("v14", {})
            observed_advantage = v14.get("observed_advantage")
            observed_error_base = v14.get("observed_error_base")
            observed_error_ctf = v14.get("observed_error_ctf")
            alpha = v14.get("alpha_final")
            if not all(
                torch.is_tensor(value)
                for value in (
                    observed_advantage,
                    observed_error_base,
                    observed_error_ctf,
                    alpha,
                )
            ):
                raise RuntimeError("V14 gate diagnostics are absent")
            relative_observed_advantage = observed_advantage / (
                observed_error_base + observed_error_ctf
            ).clamp_min(1e-6)
            tensors = {
                "observed_advantage": observed_advantage,
                "relative_observed_advantage": relative_observed_advantage,
                "hidden_ctf_advantage": base_mae - ctf_mae,
                "hidden_final_advantage": base_mae - final_mae,
                "alpha_final": alpha,
                "oracle_alpha": oracle,
                "base_mae": base_mae,
                "ctf_mae": ctf_mae,
                "final_mae": final_mae,
            }
            for key, tensor in tensors.items():
                values[key].extend(tensor.detach().float().cpu().flatten().tolist())

    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    observed = arrays["observed_advantage"]
    relative_observed = arrays["relative_observed_advantage"]
    hidden_ctf = arrays["hidden_ctf_advantage"]
    alpha = arrays["alpha_final"]
    oracle = arrays["oracle_alpha"]
    return {
        "samples": int(alpha.size),
        "observed_to_hidden_ctf_pearson": _pearson(observed, hidden_ctf),
        "observed_to_hidden_ctf_spearman": _spearman(observed, hidden_ctf),
        "relative_observed_to_hidden_ctf_spearman": _spearman(relative_observed, hidden_ctf),
        "observed_ctf_sign_accuracy": float(np.mean((observed > 0.0) == (hidden_ctf > 0.0))),
        "hidden_ctf_positive_rate": float(np.mean(hidden_ctf > 0.0)),
        "observed_positive_rate": float(np.mean(observed > 0.0)),
        "alpha_to_oracle_pearson": _pearson(alpha, oracle),
        "alpha_to_oracle_spearman": _spearman(alpha, oracle),
        "observed_to_oracle_spearman": _spearman(observed, oracle),
        "relative_observed_to_oracle_spearman": _spearman(relative_observed, oracle),
        "relative_observed_advantage_mean": float(np.mean(relative_observed)),
        "relative_observed_advantage_std": float(np.std(relative_observed)),
        "relative_observed_advantage_q05": float(np.quantile(relative_observed, 0.05)),
        "relative_observed_advantage_q95": float(np.quantile(relative_observed, 0.95)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha)),
        "oracle_alpha_mean": float(np.mean(oracle)),
        "oracle_alpha_std": float(np.std(oracle)),
        "oracle_zero_rate": float(np.mean(oracle <= 1e-8)),
        "oracle_max_rate": float(np.mean(oracle >= alpha_max - 1e-8)),
        "alpha_oracle_mae": float(np.mean(np.abs(alpha - oracle))),
        "base_mae": float(np.mean(arrays["base_mae"])),
        "ctf_mae": float(np.mean(arrays["ctf_mae"])),
        "final_mae": float(np.mean(arrays["final_mae"])),
        "final_improves_base_rate": float(np.mean(arrays["hidden_final_advantage"] > 0.0)),
    }


def _markdown(rows: list[dict], scope: str, created_at: str) -> str:
    lines = [
        "# V14 最终门控校准诊断",
        "",
        f"- 生成时间：{created_at}",
        f"- 范围：{scope}",
        "- 数据：已有 seed=42 V14 最优检查点的验证集。",
        "- 用途：只读诊断；缺失区真值没有进入模型输入。",
        "",
        "| 数据集 | 模式 | 缺失率 | obs→hidden ρ | α→oracle ρ | α均值 | oracle均值 | oracle=0 | 最终优于base |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {pattern} | {rate} | {observed_to_hidden_ctf_spearman:+.3f} | "
            "{alpha_to_oracle_spearman:+.3f} | {alpha_mean:.5f} | "
            "{oracle_alpha_mean:.5f} | {oracle_zero_rate:.1%} | "
            "{final_improves_base_rate:.1%} |".format(**row)
        )
    finite_obs = [row["observed_to_hidden_ctf_spearman"] for row in rows if np.isfinite(row["observed_to_hidden_ctf_spearman"])]
    finite_alpha = [row["alpha_to_oracle_spearman"] for row in rows if np.isfinite(row["alpha_to_oracle_spearman"])]
    lines.extend([
        "",
        "## 汇总",
        "",
        f"- 观测收益与缺失区 CTF 收益的平均 Spearman 相关：{np.mean(finite_obs):+.3f}。",
        f"- 门控 α 与解析 MSE 最优 α 的平均 Spearman 相关：{np.mean(finite_alpha):+.3f}。",
        f"- α 与 oracle 的平均绝对误差：{np.mean([row['alpha_oracle_mae'] for row in rows]):.5f}。",
        f"- 最终结果逐样本优于 base 的平均比例：{np.mean([row['final_improves_base_rate'] for row in rows]):.1%}。",
        "",
        "相关性接近 0 或为负，说明现有门控信号无法稳定排序真实残差收益；此时下一步应优先改善门控校准，而不是继续放大残差。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the checkpoint diagnostic")
    device = torch.device(f"cuda:{args.gpu}")
    points = _points(args.scope)
    rows: list[dict] = []
    for index, (dataset, pattern, rate) in enumerate(points, start=1):
        print(f"[{index}/{len(points)}] {dataset} {pattern}@{rate}", flush=True)
        run_dir = _latest_run(dataset, pattern, rate, args.seed)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cfg["data"]["num_workers"] = 0
        model = DualBranchSTImputer.from_config(cfg).to(device)
        checkpoint = load_checkpoint(run_dir / "checkpoints/best.pt", model, map_location=device)
        dataset_obj = _dataset(cfg, dataset)
        loader = build_loader(dataset_obj, cfg, shuffle=False)
        alpha_max = float(cfg.get("model", {}).get("v14", {}).get("alpha_final_max", 0.5))
        result = _evaluate(
            model,
            loader,
            device,
            alpha_max,
            f"val {dataset} {pattern}@{rate}",
        )
        row = {
            "dataset": dataset,
            "pattern": pattern,
            "rate": rate,
            "run_dir": str(run_dir),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "alpha_max": alpha_max,
            **result,
        }
        rows.append(row)
        print(
            f"  obs→hidden ρ={row['observed_to_hidden_ctf_spearman']:+.3f}; "
            f"α→oracle ρ={row['alpha_to_oracle_spearman']:+.3f}; "
            f"α={row['alpha_mean']:.5f}, oracle={row['oracle_alpha_mean']:.5f}",
            flush=True,
        )
        del model, checkpoint, loader, dataset_obj
        torch.cuda.empty_cache()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_path = output_dir / f"{args.scope}_details.json"
    csv_path = output_dir / f"{args.scope}_summary.csv"
    report_path = output_dir / f"{args.scope}_report.md"
    json_path.write_text(
        json.dumps({"created_at": created_at, "scope": args.scope, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(_markdown(rows, args.scope, created_at), encoding="utf-8")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
