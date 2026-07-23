#!/usr/bin/env python3
"""Run the V17.1 E0-E7 exploratory ablations on the four core points."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "v17.1-single" / "exploratory"
OUTPUT_ROOT = ROOT / "outputs" / "v17.1-single"
V17_OUTPUT_ROOT = ROOT / "outputs" / "v17-single"

VARIANTS = (
    "full",
    "no_fine_floor",
    "hard_fine_floor",
    "global_route_gamma",
    "independent_shared_scale",
    "decoupled_expert_router",
    "progressive_fusion",
    "no_adapter",
)
VARIANT_CODES = {
    "full": "E0",
    "no_adapter": "E1",
    "decoupled_expert_router": "E2",
    "progressive_fusion": "E3",
    "no_fine_floor": "E4",
    "hard_fine_floor": "E5",
    "global_route_gamma": "E6",
    "independent_shared_scale": "E7",
}


@dataclass(frozen=True)
class Point:
    point_id: str
    dataset: str
    mask: str
    rate: str

    @property
    def output_dataset(self) -> str:
        return "CHAP_Beijing" if self.dataset == "CHAP" else self.dataset

    @property
    def label(self) -> str:
        return f"{self.point_id} {self.dataset} {self.mask}@{self.rate}"


POINTS = {
    "P1": Point("P1", "TaxiBJ", "random", "0.4"),
    "P2": Point("P2", "BikeNYC", "fixed", "0.8"),
    "P3": Point("P3", "TaxiBJ", "fixed", "0.4"),
    "P4": Point("P4", "CHAP", "fixed", "0.4"),
}


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _has_finite_test(metrics_path: Path) -> bool:
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("stage") != "test":
                continue
            metrics = record.get("metrics") or {}
            return all(math.isfinite(float(metrics[key])) for key in ("mae", "rmse", "loss"))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return False


def _test_mae(run_dir: Path) -> float | None:
    try:
        records = (run_dir / "logs" / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        for line in reversed(records):
            record = json.loads(line)
            if record.get("stage") != "test":
                continue
            mae = float((record.get("metrics") or {})["mae"])
            return mae if math.isfinite(mae) else None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return None


def _completed_run(point: Point, variant: str, seed: int) -> Path | None:
    base = (
        OUTPUT_ROOT
        / point.output_dataset
        / "ablation"
        / variant
        / point.mask
        / f"rate{float(point.rate):g}"
    )
    if not base.is_dir():
        return None
    completed: list[Path] = []
    for config_path in base.glob("*/config.json"):
        run_dir = config_path.parent
        config = _load_json(config_path)
        experiment = config.get("experiment", {})
        mask = config.get("data", {}).get("mask", {})
        matches = (
            config.get("model", {}).get("version") == "v17.1-single"
            and experiment.get("group") == "v17_exploratory_ablation"
            and experiment.get("variant") == variant
            and int(config.get("seed", -1)) == seed
            and mask.get("pattern") == point.mask
            and math.isclose(float(mask.get("missing_rate", -1)), float(point.rate), abs_tol=1e-9)
        )
        required = (
            run_dir / "checkpoints" / "best.pt",
            run_dir / "logs" / "train.log",
            run_dir / "logs" / "val.log",
            run_dir / "logs" / "test.log",
            run_dir / "logs" / "metrics.jsonl",
            run_dir / "router_diagnostics.json",
            run_dir / "git_metadata.json",
        )
        if matches and all(path.is_file() and path.stat().st_size > 0 for path in required):
            if _has_finite_test(run_dir / "logs" / "metrics.jsonl"):
                completed.append(run_dir)
    return max(completed, key=lambda path: path.stat().st_mtime) if completed else None


def _original_v17_run(point: Point, seed: int) -> Path | None:
    base = (
        V17_OUTPUT_ROOT
        / point.output_dataset
        / "full"
        / "model"
        / point.mask
        / f"rate{float(point.rate):g}"
    )
    if not base.is_dir():
        return None
    completed = []
    for config_path in base.glob("*/config.json"):
        run_dir = config_path.parent
        config = _load_json(config_path)
        mask = config.get("data", {}).get("mask", {})
        matches = (
            config.get("model", {}).get("version") == "v17-single"
            and int(config.get("seed", -1)) == seed
            and mask.get("pattern") == point.mask
            and math.isclose(
                float(mask.get("missing_rate", -1)), float(point.rate), abs_tol=1e-9
            )
        )
        if matches and _test_mae(run_dir) is not None:
            completed.append(run_dir)
    return max(completed, key=lambda path: path.stat().st_mtime) if completed else None


def _verify_full_reproduction(
    points: list[Point],
    seeds: list[int],
    tolerance_pct: float,
) -> None:
    comparisons = []
    for seed in seeds:
        if seed != 42:
            continue
        for point in points:
            original = _original_v17_run(point, seed)
            reproduced = _completed_run(point, "full", seed)
            if original is None:
                raise RuntimeError(
                    f"Missing original V17 baseline for {point.label} seed={seed}; "
                    "cannot validate E0 reproduction."
                )
            if reproduced is None:
                raise RuntimeError(
                    f"Missing completed E0 reproduction for {point.label} seed={seed}."
                )
            original_mae = _test_mae(original)
            reproduced_mae = _test_mae(reproduced)
            if original_mae is None or reproduced_mae is None:
                raise RuntimeError(f"Invalid E0/V17 Test MAE for {point.label} seed={seed}")
            relative = abs(reproduced_mae - original_mae) / original_mae * 100.0
            comparisons.append((point, original_mae, reproduced_mae, relative))
            print(
                f"[reproduce] {point.label}: V17={original_mae:.6f} "
                f"E0={reproduced_mae:.6f} diff={relative:.3f}%",
                flush=True,
            )
            if relative > tolerance_pct:
                raise RuntimeError(
                    f"E0 reproduction failed for {point.label}: {relative:.3f}% > "
                    f"{tolerance_pct:.3f}%. Stop before interpreting E1-E7."
                )
    if comparisons:
        print(
            f"[reproduce] PASS {len(comparisons)} paired point(s), "
            f"tolerance={tolerance_pct:.3f}%",
            flush=True,
        )


def _expand_selection(values: list[str], all_values: tuple[str, ...]) -> list[str]:
    if "all" in values or "core4" in values:
        return list(all_values)
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--points",
        nargs="+",
        choices=("core4", *POINTS),
        default=["core4"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("all", *VARIANTS),
        default=["all"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one train epoch, one validation and one test in the debug tree.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=0.5,
        help="Maximum E0 relative Test-MAE difference from the original V17 (percent).",
    )
    parser.add_argument(
        "--skip-reproduction-check",
        action="store_true",
        help="Explicitly bypass the E0-vs-V17 guard before E1-E7.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.reproduction_tolerance < 0:
        raise ValueError("--reproduction-tolerance must be non-negative")
    variants = _expand_selection(args.variants, VARIANTS)
    point_ids = _expand_selection(args.points, tuple(POINTS))
    points = [POINTS[point_id] for point_id in point_ids]
    jobs = [
        (variant, POINTS[point_id], seed)
        for variant in variants
        for point_id in point_ids
        for seed in args.seeds
    ]

    skipped = 0
    reproduction_checked = False
    for index, (variant, point, seed) in enumerate(jobs, start=1):
        if (
            variant != "full"
            and not reproduction_checked
            and not args.smoke
            and not args.dry_run
            and not args.skip_reproduction_check
        ):
            _verify_full_reproduction(points, args.seeds, args.reproduction_tolerance)
            reproduction_checked = True
        config_path = CONFIG_ROOT / f"{variant}.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        completed = None
        if not args.smoke and not args.force_rerun:
            completed = _completed_run(point, variant, seed)
        if completed is not None:
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP {VARIANT_CODES[variant]} {point.label} "
                f"seed={seed}: {completed.relative_to(ROOT)}",
                flush=True,
            )
            continue

        run_name = f"debug_v17_1_{variant}" if args.smoke else f"ablation_{variant}"
        command = [
            sys.executable,
            "scripts/v17-single/train.py",
            "--dataset", point.dataset,
            "--mask", point.mask,
            "--rate", point.rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(seed),
            "--variant-config", str(config_path.relative_to(ROOT)),
            "--output-dir", "outputs/v17.1-single",
            "--model-version", "v17.1-single",
            "--run-name", run_name,
        ]
        if args.smoke:
            command.extend(["--epochs", "1"])
        if args.dry_run:
            command.append("--dry-run")
        print(
            f"[{index}/{len(jobs)}] RUN {VARIANT_CODES[variant]} {point.label} seed={seed}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)

    print(
        f"[done] total={len(jobs)} skipped={skipped} executed={len(jobs) - skipped} "
        f"mode={'smoke' if args.smoke else 'formal'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
