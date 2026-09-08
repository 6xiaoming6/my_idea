#!/usr/bin/env python3
"""Summarize matched completed V21.2 runs without selecting on test scores."""
import argparse
import json
from pathlib import Path
import sys

from train import DEFAULT_CONFIG, ROOT, load_suite
from run_core6 import completed

sys.path.insert(0, str(ROOT / "scripts/v21.1-single"))
from v21_1_protocol import Point, VARIANTS, find_completed_run, read_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    suite = load_suite(args.config)
    variants = args.variants or suite["default_variants"]
    rows, missing = [], []
    for dataset, pattern, rate in suite["core6"]:
        point = Point(dataset, pattern, rate)
        epochs = args.epochs or suite["datasets"][dataset]["train"]["epochs"]
        reference = find_completed_run(VARIANTS["P0"], point, args.seed, epochs)
        base = read_result(reference) if reference else None
        for variant in variants:
            path = completed(suite, variant, dataset, pattern, rate, args.seed, args.epochs)
            if path is None:
                missing.append(f"{variant} {point.label}")
                continue
            result = read_result(path)
            rows.append({"variant": variant, "point": point.label, **result,
                         "p0_run_dir": base["run_dir"] if base else None,
                         "val_change_pct": 100 * (result["validation"]["mae"] / base["validation"]["mae"] - 1) if base else None})
    lines = ["# V21.2 Core-6", "", "验证集选优；测试仅作冻结模型的最终描述。正变化为退步。P0为已有同点同种子同epoch对照。", "",
             "| Variant | Point | Best epoch | Val MAE | Test MAE | Test RMSE | Val Δ vs P0 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        va, te = row["validation"], row["test"]
        delta = "n/a" if row["val_change_pct"] is None else f"{row['val_change_pct']:+.3f}%"
        lines.append(f"| {row['variant']} | {row['point']} | {va['epoch']} | {va['mae']:.5f} | {te['mae']:.5f} | {te['rmse']:.5f} | {delta} |")
    lines += ["", f"完成 {len(rows)}/{len(variants)*len(suite['core6'])}；缺失 {len(missing)}。", "", *[f"- 未完成：{item}" for item in missing], "",
              "不能只根据主指标宣称机制成立：还需固定系数、去失真、打乱监督对照，以及留出风险对真实尺度误差的迁移验证。"]
    tag = f"seed{args.seed}_epochs{args.epochs or 'full'}"
    root = ROOT / suite["common"]["output_dir"] / "summary"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{tag}.md").write_text("\n".join(lines) + "\n")
    (root / f"{tag}.json").write_text(json.dumps({"rows": rows, "missing": missing}, ensure_ascii=False, indent=2))
    print(root / f"{tag}.md")


if __name__ == "__main__":
    main()
