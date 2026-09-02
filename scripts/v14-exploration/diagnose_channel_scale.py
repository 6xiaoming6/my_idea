#!/usr/bin/env python3
"""Sweep two-channel V14 effective-residual multipliers without retraining."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
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
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
    },
    "BikeNYC": {
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--kappas",
        nargs="+",
        type=float,
        default=(0.75, 1.0, 1.25, 1.5),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v14-exploration/diagnostics/channel_scale",
    )
    return parser.parse_args()


def _latest_run(dataset: str, pattern: str, rate: str) -> Path:
    root = (
        ROOT / "outputs/v14-single" / dataset / "full/model"
        / pattern / f"rate{rate}"
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
    scales = cfg["data"]["scales"]
    mask_cfg = cfg["data"]["mask"]
    mask_path = Path(mask_cfg[f"{split}_csv"])
    if not mask_path.is_absolute():
        mask_path = ROOT / mask_path
    return FlowNPZDataset(
        path,
        mask_cfg=mask_cfg,
        fine_to_mid=scales["fine_to_mid"],
        fine_to_coarse=scales["fine_to_coarse"],
        pooling_mode=scales.get("pooling_mode", "avg"),
        seed=int(cfg.get("seed", 42)) + (20000 if split == "val" else 30000),
        mask_csv=mask_path,
    )


def _evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    multipliers: tuple[tuple[float, float], ...],
    desc: str,
) -> dict[tuple[float, float], dict[str, float]]:
    accumulators = {
        multiplier: MaskedMetricAccumulator() for multiplier in multipliers
    }
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            x_base = outputs["x_hat_base"]
            residual = outputs["x_hat_final"] - x_base
            if residual.shape[1] != 2:
                raise ValueError(
                    f"Channel sweep requires C=2, got {residual.shape[1]}"
                )
            for multiplier, accumulator in accumulators.items():
                scale = residual.new_tensor(multiplier).view(1, 2, 1, 1, 1)
                accumulator.update(
                    x_base + scale * residual,
                    batch["x_f_gt"],
                    batch["m_f"],
                )
    return {
        multiplier: accumulator.compute()
        for multiplier, accumulator in accumulators.items()
    }


def _markdown(rows: list[dict], created_at: str) -> str:
    different = sum(row["best_kappa_0"] != row["best_kappa_1"] for row in rows)
    test_wins = sum(row["test_mae_best"] < row["test_mae_original"] for row in rows)
    lines = [
        "# V14 双通道残差倍率诊断",
        "",
        f"- 生成时间：{created_at}",
        "- 倍率仅在验证集选择，测试集不参与选择。",
        "- 范围：TaxiBJ/BikeNYC × fixed/random × 四种缺失率，共16点。",
        "",
        "| 数据集 | 模式 | 缺失率 | 最佳通道倍率 | Val MAE原始 | Val MAE最佳 | Test MAE原始 | Test MAE最佳 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pattern']} | {row['rate']} | "
            f"({row['best_kappa_0']:g}, {row['best_kappa_1']:g}) | "
            f"{row['val_mae_original']:.6f} | {row['val_mae_best']:.6f} | "
            f"{row['test_mae_original']:.6f} | {row['test_mae_best']:.6f} |"
        )
    lines.extend([
        "",
        "## 自动摘要",
        "",
        f"- 两通道选择不同倍率：{different}/{len(rows)}。",
        f"- 验证选择的通道倍率在测试集降低 MAE：{test_wins}/{len(rows)}。",
        "",
        "S02 只有在不同倍率具有跨模式、缺失率的一致性时才应进入训练。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    kappas = tuple(dict.fromkeys(float(value) for value in args.kappas))
    if 1.0 not in kappas:
        raise ValueError("--kappas must include 1.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint diagnostics")
    multipliers = tuple(itertools.product(kappas, repeat=2))
    original = (1.0, 1.0)
    points = tuple(
        (dataset, pattern, rate)
        for dataset in DATASETS
        for pattern in ("fixed", "random")
        for rate in RATES
    )
    device = torch.device(f"cuda:{args.gpu}")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    details: list[dict] = []
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for index, (dataset, pattern, rate) in enumerate(points, 1):
        print(f"[{index}/{len(points)}] {dataset} {pattern}@{rate}", flush=True)
        run_dir = _latest_run(dataset, pattern, rate)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cfg["data"]["num_workers"] = 0
        model = DualBranchSTImputer.from_config(cfg).to(device)
        checkpoint = load_checkpoint(
            run_dir / "checkpoints/best.pt", model, map_location=device
        )
        val_dataset = _dataset(cfg, "val", ROOT / DATASETS[dataset]["val"])
        val_loader = build_loader(val_dataset, cfg, shuffle=False)
        validation = _evaluate(
            model, val_loader, device, multipliers,
            f"val channel {dataset} {pattern}@{rate}",
        )
        best = min(
            multipliers,
            key=lambda value: (
                validation[value]["mae"],
                validation[value]["rmse"],
                abs(value[0] - 1.0) + abs(value[1] - 1.0),
            ),
        )
        test_choices = tuple(dict.fromkeys((original, best)))
        test_dataset = _dataset(cfg, "test", ROOT / DATASETS[dataset]["test"])
        test_loader = build_loader(test_dataset, cfg, shuffle=False)
        test = _evaluate(
            model, test_loader, device, test_choices,
            f"test channel {dataset} {pattern}@{rate}",
        )
        row = {
            "dataset": dataset,
            "pattern": pattern,
            "rate": rate,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "best_kappa_0": best[0],
            "best_kappa_1": best[1],
            "val_mae_original": validation[original]["mae"],
            "val_rmse_original": validation[original]["rmse"],
            "val_mae_best": validation[best]["mae"],
            "val_rmse_best": validation[best]["rmse"],
            "test_mae_original": test[original]["mae"],
            "test_rmse_original": test[original]["rmse"],
            "test_mae_best": test[best]["mae"],
            "test_rmse_best": test[best]["rmse"],
        }
        rows.append(row)
        details.append({
            **row,
            "run_dir": str(run_dir),
            "validation_grid": {
                f"{key[0]:g},{key[1]:g}": value
                for key, value in validation.items()
            },
        })
        print(
            f"  best={best}; val MAE {validation[original]['mae']:.6f} -> "
            f"{validation[best]['mae']:.6f}; test MAE "
            f"{test[original]['mae']:.6f} -> {test[best]['mae']:.6f}",
            flush=True,
        )
        del model, checkpoint, val_loader, val_dataset, test_loader, test_dataset
        torch.cuda.empty_cache()

    json_path = output_dir / "dual16_details.json"
    csv_path = output_dir / "dual16_summary.csv"
    report_path = output_dir / "dual16_report.md"
    json_path.write_text(
        json.dumps(
            {"created_at": created_at, "rows": details},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(_markdown(rows, created_at), encoding="utf-8")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
