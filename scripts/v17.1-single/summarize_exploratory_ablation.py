#!/usr/bin/env python3
"""Summarize completed V17.1 exploratory ablations and routing diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "v17.1-single"
VARIANTS = (
    "full",
    "no_adapter",
    "decoupled_expert_router",
    "progressive_fusion",
    "no_fine_floor",
    "hard_fine_floor",
    "global_route_gamma",
    "independent_shared_scale",
)
VARIANT_CODES = {variant: f"E{index}" for index, variant in enumerate(VARIANTS)}
POINTS = {
    ("TaxiBJ", "random", 0.4): "P1",
    ("BikeNYC", "fixed", 0.8): "P2",
    ("TaxiBJ", "fixed", 0.4): "P3",
    ("CHAP_Beijing", "fixed", 0.4): "P4",
}
POINT_LABELS = {
    "P1": "Taxi random@0.4",
    "P2": "Bike fixed@0.8",
    "P3": "Taxi fixed@0.4",
    "P4": "CHAP fixed@0.4",
}


@dataclass
class Result:
    variant: str
    point: str
    seed: int
    run_dir: Path
    mae: float
    rmse: float
    loss: float
    best_epoch: int
    best_val_mae: float
    total_time_sec: float
    peak_memory_gb: float
    diagnostics: dict[str, float]
    collapse_flags: dict
    relative_to_full: float | None = None


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_records(path: Path) -> list[dict]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _point_from_config(config: dict) -> str | None:
    dataset = config.get("data", {}).get("dataset_name")
    mask = config.get("data", {}).get("mask", {})
    rate = _finite(mask.get("missing_rate"))
    if rate is None:
        return None
    for (expected_dataset, expected_mask, expected_rate), point in POINTS.items():
        if (
            dataset == expected_dataset
            and mask.get("pattern") == expected_mask
            and math.isclose(rate, expected_rate, abs_tol=1e-9)
        ):
            return point
    return None


def _parse_result(config_path: Path) -> Result | None:
    run_dir = config_path.parent
    config = _load_json(config_path)
    experiment = config.get("experiment", {})
    variant = experiment.get("variant")
    if (
        config.get("model", {}).get("version") != "v17.1-single"
        or experiment.get("group") != "v17_exploratory_ablation"
        or variant not in VARIANTS
    ):
        return None
    point = _point_from_config(config)
    if point is None:
        return None

    records = _read_records(run_dir / "logs" / "metrics.jsonl")
    test_record = next(
        (record for record in reversed(records) if record.get("stage") == "test"),
        None,
    )
    if test_record is None:
        return None
    metrics = test_record.get("metrics") or {}
    mae = _finite(metrics.get("mae"))
    rmse = _finite(metrics.get("rmse"))
    loss = _finite(metrics.get("loss"))
    if None in (mae, rmse, loss):
        return None

    extra = test_record.get("extra") or {}
    epoch_records = [record for record in records if "epoch" in record]
    total_time = sum(
        _finite(record.get("perf", {}).get("epoch_time_sec")) or 0.0
        for record in epoch_records
    )
    peak_memory = max(
        (_finite(record.get("perf", {}).get("peak_memory_gb")) or 0.0 for record in epoch_records),
        default=0.0,
    )
    router = _load_json(run_dir / "router_diagnostics.json")
    diagnostics = {
        key: float(value)
        for key, value in (router.get("metrics") or {}).items()
        if _finite(value) is not None
    }
    return Result(
        variant=variant,
        point=point,
        seed=int(config.get("seed", -1)),
        run_dir=run_dir,
        mae=float(mae),
        rmse=float(rmse),
        loss=float(loss),
        best_epoch=int(extra.get("best_epoch", 0)),
        best_val_mae=float(extra.get("best_val_mae", math.nan)),
        total_time_sec=total_time,
        peak_memory_gb=peak_memory,
        diagnostics=diagnostics,
        collapse_flags=router.get("collapse_flags") or {},
    )


def collect(output_root: Path, seeds: set[int]) -> list[Result]:
    latest: dict[tuple[str, str, int], Result] = {}
    for config_path in output_root.glob("*/ablation/*/*/rate*/*/config.json"):
        result = _parse_result(config_path)
        if result is None or result.seed not in seeds:
            continue
        key = (result.variant, result.point, result.seed)
        previous = latest.get(key)
        if previous is None or result.run_dir.stat().st_mtime > previous.run_dir.stat().st_mtime:
            latest[key] = result

    full = {
        (result.point, result.seed): result
        for result in latest.values()
        if result.variant == "full"
    }
    for result in latest.values():
        baseline = full.get((result.point, result.seed))
        if baseline is not None:
            result.relative_to_full = (result.mae - baseline.mae) / baseline.mae * 100.0
    return sorted(
        latest.values(),
        key=lambda result: (
            VARIANTS.index(result.variant),
            result.point,
            result.seed,
        ),
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _format_cell(results: list[Result]) -> str:
    if not results:
        return "—"
    maes = [result.mae for result in results]
    relatives = [
        result.relative_to_full
        for result in results
        if result.relative_to_full is not None
    ]
    if len(maes) == 1:
        metric = f"{maes[0]:.6f}"
    else:
        metric = f"{_mean(maes):.6f}±{statistics.pstdev(maes):.6f}"
    return f"{metric} ({_mean(relatives):+.2f}%)" if relatives else metric


def _collapse_text(results: list[Result]) -> str:
    if not results:
        return "—"
    scale = sum(bool(result.collapse_flags.get("scale")) for result in results)
    route = sum(bool(result.collapse_flags.get("route_gate")) for result in results)
    expert = sum(
        any(bool(value) for value in (result.collapse_flags.get("expert") or {}).values())
        for result in results
    )
    return f"scale {scale}/{len(results)}, expert {expert}/{len(results)}, route {route}/{len(results)}"


def _build_markdown(results: list[Result], seeds: list[int]) -> str:
    lines = [
        "# V17.1 探索性消融实验汇总",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        f"> Seeds：{', '.join(map(str, seeds))}  ",
        "> 单元格：Test MAE（相对同 seed Full 的变化；负数为改善）",
        "",
        "## 核心四点结果",
        "",
        "| Variant | P1 Taxi random0.4 | P2 Bike fixed0.8 | P3 Taxi fixed0.4 | P4 CHAP fixed0.4 | 四点平均变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        variant_results = [result for result in results if result.variant == variant]
        cells = []
        for point in ("P1", "P2", "P3", "P4"):
            cells.append(_format_cell([result for result in variant_results if result.point == point]))
        relative = [
            result.relative_to_full
            for result in variant_results
            if result.relative_to_full is not None
        ]
        relative_text = f"{_mean(relative):+.2f}%" if relative else "—"
        lines.append(
            f"| {VARIANT_CODES[variant]} {variant} | {' | '.join(cells)} | {relative_text} |"
        )

    lines += [
        "",
        "## 成功点保护与失败点修复",
        "",
        "| Variant | 成功点 P1/P2 平均变化 | 失败点 P3/P4 平均变化 | 初步判定依据 |",
        "|---|---:|---:|---|",
    ]
    for variant in VARIANTS:
        variant_results = [result for result in results if result.variant == variant]
        success = [
            result.relative_to_full
            for result in variant_results
            if result.point in {"P1", "P2"} and result.relative_to_full is not None
        ]
        failure = [
            result.relative_to_full
            for result in variant_results
            if result.point in {"P3", "P4"} and result.relative_to_full is not None
        ]
        success_mean = _mean(success)
        failure_mean = _mean(failure)
        if variant == "full":
            decision = "基线"
        elif math.isfinite(success_mean) and math.isfinite(failure_mean):
            if success_mean <= 1.5 and failure_mean < -3.0:
                decision = "Full 当前实现可能有害"
            elif success_mean > 2.0 and failure_mean < -3.0:
                decision = "模式依赖，需条件化"
            elif success_mean > 2.0 and failure_mean >= -3.0:
                decision = "被删除组件可能有效"
            elif abs(success_mean) < 1.0 and abs(failure_mean) < 1.0:
                decision = "单 seed 下不明确"
            else:
                decision = "需结合逐点和路由诊断"
        else:
            decision = "结果不完整"
        success_text = f"{success_mean:+.2f}%" if math.isfinite(success_mean) else "—"
        failure_text = f"{failure_mean:+.2f}%" if math.isfinite(failure_mean) else "—"
        lines.append(f"| {VARIANT_CODES[variant]} {variant} | {success_text} | {failure_text} | {decision} |")

    lines += [
        "",
        "## 塌缩诊断",
        "",
        "| Variant | 检测结果（触发次数/已完成运行） |",
        "|---|---|",
    ]
    for variant in VARIANTS:
        variant_results = [result for result in results if result.variant == variant]
        lines.append(f"| {VARIANT_CODES[variant]} {variant} | {_collapse_text(variant_results)} |")

    lines += [
        "",
        "## 运行信息",
        "",
        "| Variant | 完成数 | 总训练时间(h) | 峰值显存(GB) |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        variant_results = [result for result in results if result.variant == variant]
        total_hours = sum(result.total_time_sec for result in variant_results) / 3600.0
        peak = max((result.peak_memory_gb for result in variant_results), default=0.0)
        lines.append(
            f"| {VARIANT_CODES[variant]} {variant} | {len(variant_results)} | {total_hours:.3f} | {peak:.3f} |"
        )

    expected = len(VARIANTS) * 4 * len(seeds)
    lines += [
        "",
        "## 完整性",
        "",
        f"- 计划运行：{expected}",
        f"- 已完成且指标有限：{len(results)}",
        f"- 缺失：{expected - len(results)}",
        "- TaxiBJ/BikeNYC 的 MAPE 不参与消融判断。",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, results: list[Result]) -> None:
    fields = [
        "variant", "point", "seed", "mae", "rmse", "loss",
        "relative_to_full_pct", "best_epoch", "best_val_mae",
        "total_time_sec", "peak_memory_gb", "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "variant": result.variant,
                    "point": result.point,
                    "seed": result.seed,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "loss": result.loss,
                    "relative_to_full_pct": result.relative_to_full,
                    "best_epoch": result.best_epoch,
                    "best_val_mae": result.best_val_mae,
                    "total_time_sec": result.total_time_sec,
                    "peak_memory_gb": result.peak_memory_gb,
                    "run_dir": str(result.run_dir.relative_to(ROOT)),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    results = collect(output_root, set(args.seeds))
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    seed_suffix = "_".join(map(str, args.seeds))
    markdown_path = summary_dir / f"exploratory_ablation_summary_seed{seed_suffix}.md"
    csv_path = summary_dir / f"exploratory_ablation_results_seed{seed_suffix}.csv"
    json_path = summary_dir / f"exploratory_ablation_results_seed{seed_suffix}.json"

    markdown = _build_markdown(results, args.seeds)
    markdown_path.write_text(markdown, encoding="utf-8")
    _write_csv(csv_path, results)
    json_path.write_text(
        json.dumps(
            [
                {
                    **{
                        key: value
                        for key, value in result.__dict__.items()
                        if key not in {"run_dir"}
                    },
                    "run_dir": str(result.run_dir.relative_to(ROOT)),
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(markdown)
    print(f"[saved] {markdown_path.relative_to(ROOT)}")
    print(f"[saved] {csv_path.relative_to(ROOT)}")
    print(f"[saved] {json_path.relative_to(ROOT)}")

    expected = len(VARIANTS) * 4 * len(args.seeds)
    if args.require_complete and len(results) != expected:
        raise SystemExit(f"Incomplete results: expected {expected}, found {len(results)}")


if __name__ == "__main__":
    main()
