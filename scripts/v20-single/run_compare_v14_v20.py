#!/usr/bin/env python3
"""Run a paired, resumable V14-vs-V20 validation protocol and summarize it."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
PATTERNS = ("fixed", "random")
RATES = ("0.2", "0.4", "0.6", "0.8")
OUTPUT_DATASET = {
    "TaxiBJ": "TaxiBJ",
    "BikeNYC": "BikeNYC",
    "CHAP": "CHAP_Beijing",
}
TEST_NPZ = {
    "TaxiBJ": "data/TaxiBJ/taxibj_test.npz",
    "BikeNYC": "data/BikeNYC/bikenyc_test.npz",
    "CHAP": "data/CHAP/beijing/chap_beijing_test.npz",
}
MODEL_SPEC = {
    "v14": {
        "script": "scripts/v14-single/train.py",
        "output": "outputs/v14-single",
        "architecture": "v14_safe_c2f_moe",
    },
    "v20": {
        "script": "scripts/v20-single/train.py",
        "output": "outputs/v20-single",
        "architecture": "v20_probe_validated_c2f_moe",
    },
}
SCREENING_POINTS = (
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.4"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.4"),
)
CORE_POINTS = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "random", "0.4"),
    ("CHAP", "fixed", "0.4"),
)
FULL_POINTS = tuple(
    (dataset, pattern, rate)
    for dataset in DATASETS
    for pattern in PATTERNS
    for rate in RATES
)
ABLATIONS = (
    "random_exam_only",
    "geometry_exam_only",
    "random_hybrid",
    "geometry_prior_only",
)


@dataclass(frozen=True)
class Case:
    dataset: str
    pattern: str
    rate: str
    seed: int


@dataclass(frozen=True)
class Job:
    model: str
    case: Case
    run_name: str
    ablation: str = "none"


@dataclass
class Result:
    run_dir: Path
    config: dict
    best_epoch: int
    val: dict
    test: dict
    avg_epoch_time: float
    peak_memory: float


def _safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not tag:
        raise ValueError("--tag cannot be empty")
    return tag


def _rate_dir(rate: str) -> str:
    return f"rate{float(rate):g}"


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _load_result(metrics_path: Path, expected_architecture: str) -> Result | None:
    run_dir = metrics_path.parent.parent
    config_path = run_dir / "config.json"
    checkpoint = run_dir / "checkpoints/best.pt"
    if not config_path.is_file() or not checkpoint.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        records = _read_jsonl(metrics_path)
    except (OSError, json.JSONDecodeError):
        return None
    if config.get("model", {}).get("architecture") != expected_architecture:
        return None
    epochs = [record for record in records if "epoch" in record and "train" in record]
    tests = [record for record in records if record.get("stage") == "test"]
    best = [record for record in epochs if record.get("is_best") and record.get("val")]
    configured_epochs = int(config.get("train", {}).get("epochs", 0))
    if not epochs or not tests or not best or max(int(row["epoch"]) for row in epochs) != configured_epochs:
        return None
    perfs = [row.get("perf", {}) for row in epochs]
    epoch_times = [float(perf["epoch_time_sec"]) for perf in perfs if "epoch_time_sec" in perf]
    peak_memories = [float(perf["peak_memory_gb"]) for perf in perfs if "peak_memory_gb" in perf]
    return Result(
        run_dir=run_dir,
        config=config,
        best_epoch=int(best[-1]["epoch"]),
        val=best[-1]["val"],
        test=tests[-1]["metrics"],
        avg_epoch_time=statistics.fmean(epoch_times) if epoch_times else float("nan"),
        peak_memory=max(peak_memories, default=float("nan")),
    )


def _result_matches(result: Result, job: Job, epochs: int | None) -> bool:
    data = result.config.get("data", {})
    mask = data.get("mask", {})
    if int(result.config.get("seed", -1)) != job.case.seed:
        return False
    if mask.get("pattern") != job.case.pattern:
        return False
    if not math.isclose(float(mask.get("missing_rate", -1.0)), float(job.case.rate)):
        return False
    if epochs is not None and int(result.config.get("train", {}).get("epochs", -1)) != epochs:
        return False
    return True


def _find_result(job: Job, epochs: int | None) -> Result | None:
    spec = MODEL_SPEC[job.model]
    base = (
        ROOT
        / spec["output"]
        / OUTPUT_DATASET[job.case.dataset]
        / "custom"
        / job.run_name
        / job.case.pattern
        / _rate_dir(job.case.rate)
    )
    candidates = []
    for metrics_path in base.glob("*/logs/metrics.jsonl"):
        result = _load_result(metrics_path, spec["architecture"])
        if result is not None and _result_matches(result, job, epochs):
            candidates.append(result)
    return max(candidates, key=lambda item: item.run_dir.stat().st_mtime, default=None)


def _profile_cases(profile: str, seeds: list[int]) -> list[Case]:
    primary_seed = seeds[0]
    if profile == "screening":
        points_and_seeds = ((point, primary_seed) for point in SCREENING_POINTS)
    elif profile == "robust":
        points_and_seeds = ((point, seed) for seed in seeds for point in CORE_POINTS)
    elif profile == "full":
        points_and_seeds = ((point, primary_seed) for point in FULL_POINTS)
    else:
        # Full 24-point coverage on the primary seed, then two additional
        # paired seeds on the four most informative points.
        points_and_seeds = (
            *((point, primary_seed) for point in FULL_POINTS),
            *((point, seed) for seed in seeds[1:] for point in CORE_POINTS),
        )
    unique = []
    seen = set()
    for point, seed in points_and_seeds:
        case = Case(*point, int(seed))
        if case not in seen:
            seen.add(case)
            unique.append(case)
    return unique


def _command(job: Job, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        MODEL_SPEC[job.model]["script"],
        "--dataset", job.case.dataset,
        "--mask", job.case.pattern,
        "--rate", job.case.rate,
        "--gpu", args.gpu,
        "--conda-env", args.conda_env,
        "--cpu-threads", str(args.cpu_threads),
        "--seed", str(job.case.seed),
        "--run-name", job.run_name,
    ]
    if job.model == "v20":
        command.extend(("--ablation", job.ablation))
    if args.epochs is not None:
        command.extend(("--epochs", str(args.epochs)))
    if args.dry_run:
        command.append("--dry-run")
    return command


def _write_progress(path: Path, *, profile: str, total: int, states: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "total_jobs": total,
        "states": states,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_jobs(
    jobs: list[Job],
    args: argparse.Namespace,
    progress_path: Path,
) -> list[str]:
    states = []
    failures = []
    for index, job in enumerate(jobs, start=1):
        label = (
            f"{job.model.upper()} {job.case.dataset} {job.case.pattern}@{job.case.rate} "
            f"seed={job.case.seed}"
        )
        if job.ablation != "none":
            label += f" ablation={job.ablation}"
        existing = None if args.rerun_completed else _find_result(job, args.epochs)
        if existing is not None:
            print(f"[{index}/{len(jobs)}] SKIP complete: {label}", flush=True)
            states.append({"job": label, "status": "skipped", "run_dir": str(existing.run_dir)})
            _write_progress(progress_path, profile=args.profile, total=len(jobs), states=states)
            continue
        command = _command(job, args)
        print(f"[{index}/{len(jobs)}] START {label}", flush=True)
        print(" ".join(command), flush=True)
        started = time.monotonic()
        status = "dry_run" if args.dry_run else "finished"
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            if not args.dry_run and _find_result(job, args.epochs) is None:
                raise RuntimeError("training command returned successfully but no complete result was found")
        except (subprocess.CalledProcessError, RuntimeError) as error:
            status = "failed"
            failures.append(f"{label}: {error}")
            print(f"[{index}/{len(jobs)}] FAILED {label}: {error}", flush=True)
            if args.stop_on_error:
                raise
        elapsed = time.monotonic() - started
        states.append({"job": label, "status": status, "elapsed_sec": round(elapsed, 2)})
        _write_progress(progress_path, profile=args.profile, total=len(jobs), states=states)
        if not args.dry_run and args.cooldown > 0:
            time.sleep(args.cooldown)
    if failures:
        print("\nCompleted with failures:\n  - " + "\n  - ".join(failures), file=sys.stderr)
    return failures


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _relative_improvement(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / max(abs(baseline), 1e-12)


def _bootstrap_ci(values: list[float], samples: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    generator = random.Random(20260902)
    estimates = sorted(
        _mean([values[generator.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def _paired_rows(
    cases: list[Case],
    model_run_names: dict[str, str],
    epochs: int | None,
) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for case in cases:
        results = {
            model: _find_result(Job(model, case, model_run_names[model]), epochs)
            for model in ("v14", "v20")
        }
        if any(result is None for result in results.values()):
            absent = ",".join(model for model, result in results.items() if result is None)
            missing.append(f"{case.dataset} {case.pattern}@{case.rate} seed={case.seed}: {absent}")
            continue
        v14 = results["v14"]
        v20 = results["v20"]
        assert v14 is not None and v20 is not None
        rows.append({
            "dataset": case.dataset,
            "pattern": case.pattern,
            "rate": float(case.rate),
            "seed": case.seed,
            "v14_best_epoch": v14.best_epoch,
            "v20_best_epoch": v20.best_epoch,
            "v14_val_mae": float(v14.val["mae"]),
            "v20_val_mae": float(v20.val["mae"]),
            "v14_test_mae": float(v14.test["mae"]),
            "v20_test_mae": float(v20.test["mae"]),
            "mae_improvement_pct": _relative_improvement(float(v14.test["mae"]), float(v20.test["mae"])),
            "v14_test_rmse": float(v14.test["rmse"]),
            "v20_test_rmse": float(v20.test["rmse"]),
            "rmse_improvement_pct": _relative_improvement(float(v14.test["rmse"]), float(v20.test["rmse"])),
            "v14_wape": float(v14.test.get("wape", float("nan"))),
            "v20_wape": float(v20.test.get("wape", float("nan"))),
            "v14_epoch_time_sec": v14.avg_epoch_time,
            "v20_epoch_time_sec": v20.avg_epoch_time,
            "v14_peak_memory_gb": v14.peak_memory,
            "v20_peak_memory_gb": v20.peak_memory,
            "v14_run_dir": str(v14.run_dir.relative_to(ROOT)),
            "v20_run_dir": str(v20.run_dir.relative_to(ROOT)),
        })
    return rows, missing


def _group_summary(rows: list[dict]) -> dict:
    v14_mae = _mean([row["v14_test_mae"] for row in rows])
    v20_mae = _mean([row["v20_test_mae"] for row in rows])
    v14_rmse = _mean([row["v14_test_rmse"] for row in rows])
    v20_rmse = _mean([row["v20_test_rmse"] for row in rows])
    return {
        "count": len(rows),
        "v14_mae": v14_mae,
        "v20_mae": v20_mae,
        "mae_improvement": _relative_improvement(v14_mae, v20_mae),
        "mae_wins": sum(row["v20_test_mae"] < row["v14_test_mae"] for row in rows),
        "v14_rmse": v14_rmse,
        "v20_rmse": v20_rmse,
        "rmse_improvement": _relative_improvement(v14_rmse, v20_rmse),
        "rmse_wins": sum(row["v20_test_rmse"] < row["v14_test_rmse"] for row in rows),
    }


def _ablation_summary(
    v14_run_name: str,
    v20_run_name: str,
    ablation_run_names: dict[str, str],
    seed: int,
    epochs: int | None,
) -> list[dict]:
    variants = {
        "A0 V14": ("v14", "none", v14_run_name),
        "A4 V20 Full": ("v20", "none", v20_run_name),
        **{
            name: ("v20", name, run_name)
            for name, run_name in ablation_run_names.items()
        },
    }
    output = []
    for label, (model, ablation, run_name) in variants.items():
        results = []
        for point in CORE_POINTS:
            result = _find_result(Job(model, Case(*point, seed), run_name, ablation), epochs)
            if result is not None:
                results.append(result)
        if results:
            output.append({
                "variant": label,
                "complete_points": len(results),
                "mae": _mean([float(result.test["mae"]) for result in results]),
                "rmse": _mean([float(result.test["rmse"]) for result in results]),
            })
    baseline = next((row for row in output if row["variant"] == "A0 V14"), None)
    if baseline is not None:
        for row in output:
            row["mae_vs_v14_pct"] = _relative_improvement(baseline["mae"], row["mae"])
    return output


def _analysis_env(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(cpu_threads)
    return env


def _run_probe_analyses(
    args: argparse.Namespace,
    main_run_name: str,
    ablation_run_names: dict[str, str],
) -> None:
    variants = [("geometry_hybrid", "none", main_run_name)]
    if args.include_ablations:
        variants.append(("random_hybrid", "random_hybrid", ablation_run_names["random_hybrid"]))
    for label, ablation, run_name in variants:
        for point in CORE_POINTS:
            if (
                point[0] not in args.datasets
                or point[1] not in args.patterns
                or point[2] not in args.rates
            ):
                continue
            case = Case(*point, args.seeds[0])
            result = _find_result(Job("v20", case, run_name, ablation), args.epochs)
            if result is None:
                print(f"[analysis] missing complete run: {label} {case}", flush=True)
                continue
            summary = result.run_dir / "analysis/probe_oracle_summary.json"
            if summary.is_file() and not args.rerun_completed:
                print(f"[analysis] SKIP complete: {summary}", flush=True)
                continue
            command = [
                "conda", "run", "--no-capture-output", "-n", args.conda_env, "python",
                "scripts/v20-single/analyze_probe_ranking.py",
                "--run-dir", str(result.run_dir),
                "--test-npz", TEST_NPZ[case.dataset],
                "--device", "cuda:0",
            ]
            if args.analysis_max_batches is not None:
                command.extend(("--max-batches", str(args.analysis_max_batches)))
            print("[analysis]", " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=_analysis_env(args.gpu, args.cpu_threads),
                    check=args.stop_on_error,
                )


def _probe_rows(
    main_run_name: str,
    ablation_run_names: dict[str, str],
    seed: int,
    epochs: int | None,
) -> list[dict]:
    variants = [("Geometry Hybrid", "none", main_run_name)]
    if "random_hybrid" in ablation_run_names:
        variants.append(("Random Hybrid", "random_hybrid", ablation_run_names["random_hybrid"]))
    rows = []
    for label, ablation, run_name in variants:
        for point in CORE_POINTS:
            result = _find_result(Job("v20", Case(*point, seed), run_name, ablation), epochs)
            if result is None:
                continue
            path = result.run_dir / "analysis/probe_oracle_summary.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text(encoding="utf-8")).get("overall", {})
            rows.append({"variant": label, "dataset": point[0], "pattern": point[1], "rate": point[2], **summary})
    return rows


def _write_report(
    report_dir: Path,
    args: argparse.Namespace,
    cases: list[Case],
    model_run_names: dict[str, str],
    ablation_run_names: dict[str, str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = _paired_rows(cases, model_run_names, args.epochs)
    csv_path = report_dir / "paired_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    primary_seed = args.seeds[0]
    full_rows = [row for row in rows if row["seed"] == primary_seed]
    core_rows = [
        row for row in rows
        if (row["dataset"], row["pattern"], f"{row['rate']:.1f}") in CORE_POINTS
    ]
    lines = [
        "# V14 与 V20 配对验证结果",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- Profile：`{args.profile}`",
        f"- 主随机种子：`{primary_seed}`；全部随机种子：`{' '.join(map(str, args.seeds))}`",
        "- 正值 improvement 表示 V20 优于 V14；模型选择只使用验证集，表中测试指标来自各自验证最佳 checkpoint。",
        "",
        "## 逐点结果",
        "",
        "| 数据集 | 模式 | 缺失率 | Seed | V14 MAE | V20 MAE | MAE提升 | V14 RMSE | V20 RMSE | RMSE提升 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pattern']} | {row['rate']:.1f} | {row['seed']} | "
            f"{row['v14_test_mae']:.6f} | {row['v20_test_mae']:.6f} | {row['mae_improvement_pct']:+.2f}% | "
            f"{row['v14_test_rmse']:.6f} | {row['v20_test_rmse']:.6f} | {row['rmse_improvement_pct']:+.2f}% |"
        )

    lines.extend(("", "## 主种子全点位汇总", ""))
    lines.extend((
        "| 分组 | 点数 | V14 MAE | V20 MAE | MAE提升 | MAE胜点 | V14 RMSE | V20 RMSE | RMSE提升 | RMSE胜点 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    groups = [("Overall", full_rows)]
    groups.extend((dataset, [row for row in full_rows if row["dataset"] == dataset]) for dataset in DATASETS)
    groups.extend((pattern, [row for row in full_rows if row["pattern"] == pattern]) for pattern in PATTERNS)
    for label, group_rows in groups:
        if not group_rows:
            continue
        summary = _group_summary(group_rows)
        lines.append(
            f"| {label} | {summary['count']} | {summary['v14_mae']:.6f} | {summary['v20_mae']:.6f} | "
            f"{summary['mae_improvement']:+.2f}% | {summary['mae_wins']}/{summary['count']} | "
            f"{summary['v14_rmse']:.6f} | {summary['v20_rmse']:.6f} | {summary['rmse_improvement']:+.2f}% | "
            f"{summary['rmse_wins']}/{summary['count']} |"
        )

    if core_rows:
        improvements = [row["mae_improvement_pct"] for row in core_rows]
        ci_low, ci_high = _bootstrap_ci(improvements)
        lines.extend((
            "",
            "## 关键点多随机种子稳定性",
            "",
            f"- 配对样本数：{len(core_rows)}；V20 MAE 获胜：{sum(value > 0 for value in improvements)}/{len(improvements)}。",
            f"- 平均逐点相对 MAE 改善：{_mean(improvements):+.2f}%。",
            f"- 配对 bootstrap 95% CI：[{ci_low:+.2f}%, {ci_high:+.2f}%]。",
            f"- 最差单点/种子变化：{min(improvements):+.2f}%。",
        ))

    ablations = _ablation_summary(
        model_run_names["v14"],
        model_run_names["v20"],
        ablation_run_names,
        primary_seed,
        args.epochs,
    )
    if ablations:
        lines.extend((
            "",
            "## 核心机制消融（四个关键点，主种子）",
            "",
            "| Variant | 完成点数 | 平均 MAE | 平均 RMSE | 相对 V14 MAE |",
            "|---|---:|---:|---:|---:|",
        ))
        for row in ablations:
            lines.append(
                f"| {row['variant']} | {row['complete_points']}/4 | {row['mae']:.6f} | "
                f"{row['rmse']:.6f} | {row.get('mae_vs_v14_pct', float('nan')):+.2f}% |"
            )

    probes = _probe_rows(
        model_run_names["v20"], ablation_run_names, primary_seed, args.epochs
    )
    if probes:
        lines.extend((
            "",
            "## Probe 对真实缺失专家能力的预测质量",
            "",
            "| Variant | 数据集 | 点位 | Spearman | Top-1 agreement | Top-2 overlap |",
            "|---|---|---|---:|---:|---:|",
        ))
        for row in probes:
            lines.append(
                f"| {row['variant']} | {row['dataset']} | {row['pattern']}@{row['rate']} | "
                f"{row.get('spearman', float('nan')):.4f} | {row.get('top1_agreement', float('nan')):.4f} | "
                f"{row.get('top2_overlap', float('nan')):.4f} |"
            )

    if len(full_rows) == 24:
        summary = _group_summary(full_rows)
        dataset_summaries = {
            dataset: _group_summary([row for row in full_rows if row["dataset"] == dataset])
            for dataset in DATASETS
        }
        max_degradation = min(row["mae_improvement_pct"] for row in full_rows)
        criteria = {
            "MAE 至少 15/24 点获胜": summary["mae_wins"] >= 15,
            "RMSE 至少 15/24 点获胜": summary["rmse_wins"] >= 15,
            "整体平均 MAE 至少改善 1.5%": summary["mae_improvement"] >= 1.5,
            "三个数据集平均 MAE 均不退化": all(value["mae_improvement"] >= 0 for value in dataset_summaries.values()),
            "最差单点 MAE 退化不超过 3%": max_degradation >= -3.0,
        }
        lines.extend(("", "## V20 全量准入判断", ""))
        lines.extend(f"- {'PASS' if passed else 'FAIL'}：{label}" for label, passed in criteria.items())
        all_passed = all(criteria.values())
        lines.extend((
            "",
            f"**当前自动结论：{'V20 达到全量数值准入条件' if all_passed else 'V20 尚未同时达到全部准入条件'}。**",
        ))

    if missing:
        lines.extend(("", "## 尚未完成的配对", ""))
        lines.extend(f"- {item}" for item in missing)
    lines.extend((
        "",
        "## 解释原则",
        "",
        "- 先判断全量主种子覆盖，再判断关键点跨种子稳定性；不能用少数最好点代替总体结论。",
        "- 若 V20 数值提升但 Geometry Probe 排序质量不优于 Random Probe，不能把提升归因于几何匹配机制。",
        "- 若平均提升为正但置信区间跨 0 或最差退化过大，应继续视为探索性结果。",
        "",
    ))
    (report_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] wrote {report_dir / 'comparison.md'}", flush=True)
    if rows:
        print(f"[summary] wrote {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("screening", "robust", "full", "comprehensive"),
        default="comprehensive",
        help=(
            "screening=8点单种子；robust=4点多种子；full=24点单种子；"
            "comprehensive=24点主种子+4点额外种子"
        ),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPEC),
        default=tuple(MODEL_SPEC),
        help=(
            "选择本次实际训练的模型。默认同时训练 v14/v20；使用 "
            "--models v20 可复用同 tag 下已有的 V14 结果，只重跑 V20。"
        ),
    )
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 2026, 3407))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=PATTERNS)
    parser.add_argument("--rates", nargs="+", choices=RATES, default=RATES)
    parser.add_argument("--epochs", type=int, default=None, help="仅用于调试；正式比较请省略")
    parser.add_argument("--tag", default="v20_vs_v14_formal")
    parser.add_argument(
        "--v14-reference-tag",
        default=None,
        help=(
            "可选的已有 V14 对照 tag。设置后，新 V20 写入 --tag，汇总时从该 "
            "reference tag 读取 V14，适合复用已完成的 V14。"
        ),
    )
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--probe-analysis", action="store_true")
    parser.add_argument("--analysis-max-batches", type=int, default=None)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--only-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("--seeds requires at least one seed")
    args.seeds = list(dict.fromkeys(args.seeds))
    if args.profile == "comprehensive" and len(args.seeds) < 3:
        parser.error("comprehensive profile requires at least three distinct seeds")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.cooldown < 0:
        parser.error("--cooldown cannot be negative")
    if args.analysis_max_batches is not None and args.analysis_max_batches < 1:
        parser.error("--analysis-max-batches must be positive")
    return args


def main() -> None:
    args = parse_args()
    tag = _safe_tag(args.tag)
    v14_reference_tag = (
        _safe_tag(args.v14_reference_tag) if args.v14_reference_tag else tag
    )
    model_run_names = {
        "v14": f"compare_{v14_reference_tag}",
        "v20": f"compare_{tag}",
    }
    ablation_run_names = {
        ablation: f"compare_{tag}_ablation_{ablation}"
        for ablation in ABLATIONS
    }
    report_dir = ROOT / "outputs/v20-single/comparison" / tag
    cases = [
        case for case in _profile_cases(args.profile, args.seeds)
        if case.dataset in args.datasets
        and case.pattern in args.patterns
        and case.rate in args.rates
    ]
    if not cases:
        raise ValueError("The dataset/pattern/rate filters removed every protocol case")
    jobs = [
        Job(model, case, model_run_names[model])
        for case in cases
        for model in args.models
    ]
    if args.include_ablations:
        for point in CORE_POINTS:
            if (
                point[0] not in args.datasets
                or point[1] not in args.patterns
                or point[2] not in args.rates
            ):
                continue
            case = Case(*point, args.seeds[0])
            for ablation in ABLATIONS:
                jobs.append(Job("v20", case, ablation_run_names[ablation], ablation))
    print(
        f"Protocol={args.profile}, paired cases={len(cases)}, training jobs={len(jobs)}, "
        f"GPU={args.gpu}",
        flush=True,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "profile": args.profile,
        "tag": tag,
        "v14_reference_tag": v14_reference_tag,
        "seeds": args.seeds,
        "datasets": args.datasets,
        "patterns": args.patterns,
        "rates": args.rates,
        "models_to_train": args.models,
        "cases": [case.__dict__ for case in cases],
        "include_ablations": args.include_ablations,
        "probe_analysis": args.probe_analysis,
        "epochs_override": args.epochs,
        "model_run_names": model_run_names,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (report_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    failures = []
    if not args.only_summary:
        failures = _run_jobs(jobs, args, report_dir / "progress.json")
        if args.probe_analysis:
            _run_probe_analyses(
                args, model_run_names["v20"], ablation_run_names
            )
    _write_report(report_dir, args, cases, model_run_names, ablation_run_names)
    if failures:
        raise SystemExit(
            f"{len(failures)} training job(s) failed; rerun the same command to resume"
        )


if __name__ == "__main__":
    main()
