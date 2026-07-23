"""Shared result discovery helpers for the V17.1 staged experiment pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "v17.1-single"
CONFIG_ROOT = ROOT / "configs" / "v17.1-single"

STAGE1_VARIANTS = (
    "full",
    "no_adapter",
    "decoupled_expert_router",
    "progressive_fusion",
    "no_fine_floor",
    "hard_fine_floor",
    "global_route_gamma",
    "independent_shared_scale",
)
COMBINATION_VARIANTS = (
    "c1_independent_shared_scale",
    "c2_independent_shared_hard_floor",
    "c3_independent_shared_hard_floor_global_gamma",
)
COMBINATION_CODES = {
    "c1_independent_shared_scale": "C1",
    "c2_independent_shared_hard_floor": "C2",
    "c3_independent_shared_hard_floor_global_gamma": "C3",
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


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_records(path: Path) -> list[dict]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_result(run_dir: Path) -> dict | None:
    records = read_records(run_dir / "logs" / "metrics.jsonl")
    test = next(
        (record for record in reversed(records) if record.get("stage") == "test"),
        None,
    )
    if test is None:
        return None
    metrics = test.get("metrics") or {}
    parsed = {name: finite(metrics.get(name)) for name in ("mae", "rmse", "loss")}
    if any(value is None for value in parsed.values()):
        return None
    epoch_records = [record for record in records if "epoch" in record]
    router = load_json(run_dir / "router_diagnostics.json")
    return {
        **parsed,
        "extra": test.get("extra") or {},
        "total_time_sec": sum(
            finite(record.get("perf", {}).get("epoch_time_sec")) or 0.0
            for record in epoch_records
        ),
        "peak_memory_gb": max(
            (
                finite(record.get("perf", {}).get("peak_memory_gb")) or 0.0
                for record in epoch_records
            ),
            default=0.0,
        ),
        "diagnostics": router.get("metrics") or {},
        "collapse_flags": router.get("collapse_flags") or {},
    }


def latest_run(
    point: Point,
    seed: int,
    experiment_type: str,
    variant: str,
    group: str,
) -> Path | None:
    base = (
        OUTPUT_ROOT
        / point.output_dataset
        / experiment_type
        / variant
        / point.mask
        / f"rate{float(point.rate):g}"
    )
    if not base.is_dir():
        return None
    completed = []
    for config_path in base.glob("*/config.json"):
        run_dir = config_path.parent
        config = load_json(config_path)
        experiment = config.get("experiment", {})
        mask = config.get("data", {}).get("mask", {})
        try:
            matches = (
                config.get("model", {}).get("version") == "v17.1-single"
                and experiment.get("group") == group
                and experiment.get("variant") == variant
                and int(config.get("seed", -1)) == seed
                and mask.get("pattern") == point.mask
                and math.isclose(
                    float(mask.get("missing_rate", -1)),
                    float(point.rate),
                    abs_tol=1e-9,
                )
            )
        except (TypeError, ValueError):
            matches = False
        required = (
            run_dir / "checkpoints" / "best.pt",
            run_dir / "logs" / "train.log",
            run_dir / "logs" / "val.log",
            run_dir / "logs" / "test.log",
            run_dir / "logs" / "metrics.jsonl",
            run_dir / "router_diagnostics.json",
            run_dir / "git_metadata.json",
        )
        if (
            matches
            and all(path.is_file() and path.stat().st_size > 0 for path in required)
            and read_result(run_dir) is not None
        ):
            completed.append(run_dir)
    return max(completed, key=lambda path: path.stat().st_mtime) if completed else None


def stage1_run(point: Point, seed: int, variant: str) -> Path | None:
    return latest_run(
        point,
        seed,
        experiment_type="ablation",
        variant=variant,
        group="v17_exploratory_ablation",
    )


def combination_run(point: Point, seed: int, variant: str) -> Path | None:
    if variant == "c1_independent_shared_scale":
        return stage1_run(point, seed, "independent_shared_scale")
    return latest_run(
        point,
        seed,
        experiment_type="combination",
        variant=variant,
        group="v17_exploratory_combination",
    )


def missing_stage1(seed: int = 42) -> list[str]:
    return [
        f"{variant}/{point.point_id}/seed{seed}"
        for variant in STAGE1_VARIANTS
        for point in POINTS.values()
        if stage1_run(point, seed, variant) is None
    ]


def missing_combination(variant: str, seeds: list[int]) -> list[str]:
    return [
        f"{variant}/{point.point_id}/seed{seed}"
        for seed in seeds
        for point in POINTS.values()
        if combination_run(point, seed, variant) is None
    ]
