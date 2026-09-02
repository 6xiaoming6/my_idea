#!/usr/bin/env python3
"""Audit V14 exact metrics and sweep the effective residual multiplier.

For every selected existing V14 checkpoint, kappa is selected using validation
MAE only.  The test split is then evaluated at kappa=0 (base), kappa=1 (original
V14), and the validation-selected kappa.  No checkpoint is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.data import FlowNPZDataset, build_loader
from stmoe_imputer.metrics import MaskedMetricAccumulator
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


RATES = ("0.2", "0.4", "0.6", "0.8")
DATASETS = {
    "TaxiBJ": {
        "output_name": "TaxiBJ",
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
    },
    "BikeNYC": {
        "output_name": "BikeNYC",
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
    },
    "CHAP": {
        "output_name": "CHAP_Beijing",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--scope", choices=("core6", "all24"), default="all24")
    parser.add_argument(
        "--kappas",
        nargs="+",
        type=float,
        default=(0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v14-exploration/diagnostics/residual_scale",
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


def _latest_run(dataset: str, pattern: str, rate: str) -> Path:
    root = (
        ROOT
        / "outputs/v14-single"
        / DATASETS[dataset]["output_name"]
        / "full/model"
        / pattern
        / f"rate{rate}"
    )
    candidates = sorted(
        path.parent.parent
        for path in root.glob("*/checkpoints/best.pt")
        if (path.parent.parent / "config.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No complete V14 run under {root}")
    return candidates[-1]


def _dataset(cfg: dict, split: str, path: Path) -> FlowNPZDataset:
    scale_cfg = cfg["data"]["scales"]
    mask_cfg = cfg["data"]["mask"]
    mask_csv = mask_cfg[f"{split}_csv"]
    mask_path = Path(mask_csv)
    if not mask_path.is_absolute():
        mask_path = ROOT / mask_path
    return FlowNPZDataset(
        path,
        mask_cfg=mask_cfg,
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        seed=int(cfg.get("seed", 42)) + (20000 if split == "val" else 30000),
        mask_csv=mask_path,
    )


def _evaluate_kappas(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    kappas: tuple[float, ...],
    desc: str,
) -> tuple[dict[float, dict[str, float]], dict[str, float]]:
    accumulators = {kappa: MaskedMetricAccumulator() for kappa in kappas}
    diagnostics: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            x_base = outputs.get("x_hat_base")
            x_final = outputs["x_hat_final"]
            if x_base is None:
                raise RuntimeError("The checkpoint is not a V14 model: x_hat_base is absent")
            effective_residual = x_final - x_base
            for kappa, accumulator in accumulators.items():
                accumulator.update(
                    x_base + kappa * effective_residual,
                    batch["x_f_gt"],
                    batch["m_f"],
                )
            v14 = outputs.get("diagnostics", {}).get("v14", {})
            for key in ("alpha_final", "delta_ctf_norm"):
                value = v14.get(key)
                if torch.is_tensor(value):
                    diagnostics[key].extend(value.detach().float().cpu().flatten().tolist())
            diagnostics["effective_residual_rms"].extend(
                effective_residual.detach()
                .float()
                .square()
                .mean(dim=tuple(range(1, effective_residual.ndim)))
                .sqrt()
                .cpu()
                .tolist()
            )
    diagnostic_summary = {
        f"{key}_mean": sum(values) / max(1, len(values))
        for key, values in diagnostics.items()
    }
    return {
        kappa: accumulator.compute() for kappa, accumulator in accumulators.items()
    }, diagnostic_summary


def _legacy_test_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    result = {}
    for key in ("mae", "rmse", "mape"):
        match = re.search(rf"^  {key}: ([0-9eE+.-]+)$", text, flags=re.MULTILINE)
        if match:
            result[key] = float(match.group(1))
    return result


def _markdown(rows: list[dict], scope: str, created_at: str) -> str:
    lines = [
        "# V14 残差倍率与全局指标诊断",
        "",
        f"- 生成时间：{created_at}",
        f"- 范围：{scope}",
        "- κ 只在验证集选择；测试集没有参与选择。",
        "- exact 指标按整个数据划分的全部缺失元素一次性累计。",
        "",
        "| 数据集 | 模式 | 缺失率 | 最佳κ | Val MAE κ=1 | Val MAE best | Test MAE κ=1 | Test MAE best | Test RMSE κ=1 | Test RMSE best |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {pattern} | {rate} | {best_kappa:g} | "
            "{val_mae_k1:.6f} | {val_mae_best:.6f} | "
            "{test_mae_k1:.6f} | {test_mae_best:.6f} | "
            "{test_rmse_k1:.6f} | {test_rmse_best:.6f} |".format(**row)
        )
    non_unit = sum(abs(float(row["best_kappa"]) - 1.0) > 1e-12 for row in rows)
    mae_wins = sum(row["test_mae_best"] < row["test_mae_k1"] for row in rows)
    rmse_wins = sum(row["test_rmse_best"] < row["test_rmse_k1"] for row in rows)
    mean_mae_change = (
        sum(
            (row["test_mae_best"] / row["test_mae_k1"] - 1.0) * 100.0
            for row in rows
        )
        / max(1, len(rows))
    )
    legacy_rows = [
        row for row in rows
        if row.get("legacy_test_mae") is not None
        and row.get("legacy_test_rmse") is not None
    ]
    mean_mae_audit = (
        sum(
            row["exact_minus_legacy_mae"] / row["legacy_test_mae"] * 100.0
            for row in legacy_rows
        )
        / max(1, len(legacy_rows))
    )
    mean_rmse_audit = (
        sum(
            row["exact_minus_legacy_rmse"] / row["legacy_test_rmse"] * 100.0
            for row in legacy_rows
        )
        / max(1, len(legacy_rows))
    )
    max_mae_audit = max(
        (abs(row["exact_minus_legacy_mae"]) for row in legacy_rows), default=0.0
    )
    max_rmse_audit = max(
        (abs(row["exact_minus_legacy_rmse"]) for row in legacy_rows), default=0.0
    )
    lines.extend([
        "",
        "## 自动摘要",
        "",
        f"- 验证集选择非 1.0 倍率：{non_unit}/{len(rows)}。",
        f"- 选定倍率在测试集降低 MAE：{mae_wins}/{len(rows)}。",
        f"- 选定倍率在测试集降低 RMSE：{rmse_wins}/{len(rows)}。",
        f"- 测试 MAE 平均相对变化：{mean_mae_change:+.3f}%。",
        "",
        "## 全局指标核查",
        "",
        "- 旧日志采用 batch 指标等权平均；exact 指标按全部缺失元素累计。",
        f"- exact MAE 相对旧日志平均变化：{mean_mae_audit:+.3f}%，最大绝对差：{max_mae_audit:.6f}。",
        f"- exact RMSE 相对旧日志平均变化：{mean_rmse_audit:+.3f}%，最大绝对差：{max_rmse_audit:.6f}。",
        "- RMSE 必须使用 exact 口径，因为“各 batch RMSE 的平均值”不等于全局 RMSE。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.kappas:
        raise ValueError("--kappas must not be empty")
    kappas = tuple(dict.fromkeys(float(value) for value in args.kappas))
    if 0.0 not in kappas or 1.0 not in kappas:
        raise ValueError("--kappas must include 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full V14 checkpoint audit")
    device = torch.device(f"cuda:{args.gpu}")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict] = []
    details: list[dict] = []

    for index, (dataset, pattern, rate) in enumerate(_points(args.scope), start=1):
        print(
            f"[{index}/{len(_points(args.scope))}] {dataset} {pattern}@{rate}",
            flush=True,
        )
        run_dir = _latest_run(dataset, pattern, rate)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cfg["data"]["num_workers"] = 0
        model = DualBranchSTImputer.from_config(cfg).to(device)
        checkpoint = load_checkpoint(
            run_dir / "checkpoints/best.pt", model, map_location=device
        )

        val_dataset = _dataset(cfg, "val", ROOT / DATASETS[dataset]["val"])
        val_loader = build_loader(val_dataset, cfg, shuffle=False)
        val_metrics, diagnostics = _evaluate_kappas(
            model, val_loader, device, kappas, f"val {dataset} {pattern}@{rate}"
        )
        best_kappa = min(
            kappas,
            key=lambda value: (
                val_metrics[value]["mae"],
                val_metrics[value]["rmse"],
                abs(value - 1.0),
            ),
        )
        test_kappas = tuple(dict.fromkeys((0.0, 1.0, best_kappa)))
        test_dataset = _dataset(cfg, "test", ROOT / DATASETS[dataset]["test"])
        test_loader = build_loader(test_dataset, cfg, shuffle=False)
        test_metrics, _ = _evaluate_kappas(
            model, test_loader, device, test_kappas, f"test {dataset} {pattern}@{rate}"
        )
        legacy = _legacy_test_metrics(run_dir / "logs/test.log")
        row = {
            "dataset": dataset,
            "pattern": pattern,
            "rate": rate,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "best_kappa": best_kappa,
            "val_mae_k1": val_metrics[1.0]["mae"],
            "val_rmse_k1": val_metrics[1.0]["rmse"],
            "val_mae_best": val_metrics[best_kappa]["mae"],
            "val_rmse_best": val_metrics[best_kappa]["rmse"],
            "test_mae_base": test_metrics[0.0]["mae"],
            "test_rmse_base": test_metrics[0.0]["rmse"],
            "test_mae_k1": test_metrics[1.0]["mae"],
            "test_rmse_k1": test_metrics[1.0]["rmse"],
            "test_mae_best": test_metrics[best_kappa]["mae"],
            "test_rmse_best": test_metrics[best_kappa]["rmse"],
            "legacy_test_mae": legacy.get("mae"),
            "legacy_test_rmse": legacy.get("rmse"),
            "exact_minus_legacy_mae": (
                test_metrics[1.0]["mae"] - legacy["mae"] if "mae" in legacy else None
            ),
            "exact_minus_legacy_rmse": (
                test_metrics[1.0]["rmse"] - legacy["rmse"] if "rmse" in legacy else None
            ),
            **diagnostics,
        }
        rows.append(row)
        details.append({
            "run_dir": str(run_dir),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "dataset": dataset,
            "pattern": pattern,
            "rate": rate,
            "best_kappa": best_kappa,
            "validation": {str(k): value for k, value in val_metrics.items()},
            "test": {str(k): value for k, value in test_metrics.items()},
            "legacy_test": legacy,
            "diagnostics": diagnostics,
        })
        print(
            f"  best κ={best_kappa:g}; "
            f"val MAE {val_metrics[1.0]['mae']:.6f} -> "
            f"{val_metrics[best_kappa]['mae']:.6f}; "
            f"test MAE {test_metrics[1.0]['mae']:.6f} -> "
            f"{test_metrics[best_kappa]['mae']:.6f}",
            flush=True,
        )
        del model, checkpoint, val_loader, val_dataset, test_loader, test_dataset
        torch.cuda.empty_cache()

    json_path = output_dir / f"{args.scope}_details.json"
    csv_path = output_dir / f"{args.scope}_summary.csv"
    report_path = output_dir / f"{args.scope}_report.md"
    json_path.write_text(
        json.dumps(
            {"created_at": created_at, "scope": args.scope, "rows": details},
            ensure_ascii=False,
            indent=2,
        ),
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
