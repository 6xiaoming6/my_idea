"""Shared definitions and result discovery for the V21 validation protocol."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
PATTERNS = ("fixed", "random")
RATES = ("0.2", "0.4", "0.6", "0.8")
SEEDS = (42, 2026, 3407)
FULL_EPOCHS = {"TaxiBJ": 160, "BikeNYC": 140, "CHAP": 150}
OUTPUT_NAMES = {"TaxiBJ": "TaxiBJ", "BikeNYC": "BikeNYC", "CHAP": "CHAP_Beijing"}


@dataclass(frozen=True)
class Point:
    dataset: str
    pattern: str
    rate: str

    @property
    def label(self) -> str:
        return f"{self.dataset} {self.pattern}@{self.rate}"


@dataclass(frozen=True)
class Variant:
    key: str
    train_variant: str
    run_name: str
    policy_name: str
    description: str


VARIANTS = {
    "P0": Variant(
        "P0",
        "legacy_control",
        "ablation_v21_p0_legacy_pyramid",
        "v14_legacy_pyramid_control",
        "V14 legacy hierarchical pyramid",
    ),
    "P1": Variant(
        "P1",
        "content_only",
        "ablation_v21_p1_measure_pyramid",
        "path_consistent_content_only",
        "path-consistent measure pyramid without 7-D evidence",
    ),
    "P2": Variant(
        "P2",
        "dual_state",
        "ablation_v21_p2_dual_state",
        "observation_moment_dual_state",
        "path-consistent content plus 7-D evidence",
    ),
}

# Deliberately includes both favorable and adverse SDI cases.  This is a
# falsification-oriented screen, not a selection of only easy points.
CORE6 = (
    Point("TaxiBJ", "fixed", "0.6"),
    Point("TaxiBJ", "random", "0.6"),
    Point("BikeNYC", "fixed", "0.4"),
    Point("BikeNYC", "random", "0.4"),
    Point("CHAP", "fixed", "0.4"),
    Point("CHAP", "random", "0.4"),
)
ALL24 = tuple(
    Point(dataset, pattern, rate)
    for dataset in DATASETS
    for pattern in PATTERNS
    for rate in RATES
)


def points_for_phase(phase: str) -> tuple[Point, ...]:
    if phase in {"core6", "multiseed"}:
        return CORE6
    if phase == "all24":
        return ALL24
    raise ValueError(f"Unknown phase: {phase}")


def filtered_points(
    phase: str,
    datasets: tuple[str, ...],
    patterns: tuple[str, ...],
    rates: tuple[str, ...],
) -> tuple[Point, ...]:
    return tuple(
        point
        for point in points_for_phase(phase)
        if point.dataset in datasets
        and point.pattern in patterns
        and point.rate in rates
    )


def expected_epochs(point: Point, override: int | None) -> int:
    return override if override is not None else FULL_EPOCHS[point.dataset]


def run_root(variant: Variant, point: Point) -> Path:
    output_variant = variant.run_name.removeprefix("ablation_")
    return (
        ROOT
        / "outputs/v21-single"
        / OUTPUT_NAMES[point.dataset]
        / "ablation"
        / output_variant
        / point.pattern
        / f"rate{point.rate}"
    )


def _finite_test_log(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if "Testing finished:" not in text:
        return False
    for metric in ("mae", "rmse"):
        match = re.search(rf"^  {metric}:\s+(\S+)\s*$", text, re.MULTILINE)
        if match is None:
            return False
        try:
            if not math.isfinite(float(match.group(1))):
                return False
        except ValueError:
            return False
    return True


def find_completed_run(
    variant: Variant,
    point: Point,
    seed: int,
    epochs: int,
) -> Path | None:
    root = run_root(variant, point)
    for config_path in sorted(root.glob("*/config.json"), reverse=True):
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if int(cfg.get("seed", -1)) != seed:
            continue
        if int(cfg.get("train", {}).get("epochs", -1)) != epochs:
            continue
        if cfg.get("experiment_policy", {}).get("name") != variant.policy_name:
            continue
        run_dir = config_path.parent
        required = (
            run_dir / "checkpoints/best.pt",
            run_dir / "logs/metrics.jsonl",
            run_dir / "logs/test.log",
        )
        if all(path.is_file() for path in required) and _finite_test_log(required[-1]):
            return run_dir
    return None


def read_result(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "logs/metrics.jsonl"
    best: dict | None = None
    test: dict | None = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("val") is not None and record.get("is_best"):
            best = {
                "epoch": int(record["epoch"]),
                "mae": float(record["val"]["mae"]),
                "rmse": float(record["val"]["rmse"]),
                "wape": float(record["val"].get("wape", float("nan"))),
            }
        if record.get("stage") == "test" and record.get("metrics") is not None:
            value = record["metrics"]
            test = {
                "mae": float(value["mae"]),
                "rmse": float(value["rmse"]),
                "wape": float(value.get("wape", float("nan"))),
            }
    if best is None or test is None:
        raise RuntimeError(f"Incomplete metrics in {metrics_path}")
    return {"run_dir": str(run_dir.relative_to(ROOT)), "validation": best, "test": test}
