"""Shared V21.1 Core-6 protocol and completed-result discovery."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    train_variant: str | None
    run_name: str
    policy_name: str
    output_version: str
    description: str


VARIANTS = {
    "P0": Variant(
        "P0", None, "ablation_v21_p0_legacy_pyramid",
        "v14_legacy_pyramid_control", "v21-single", "V14 strict anchor",
    ),
    "P1": Variant(
        "P1", None, "ablation_v21_p1_measure_pyramid",
        "path_consistent_content_only", "v21-single", "unconditional measure pyramid",
    ),
    "P2": Variant(
        "P2", None, "ablation_v21_p2_dual_state",
        "observation_moment_dual_state", "v21-single", "direct evidence-token routing",
    ),
    "P3": Variant(
        "P3", "distortion_calibrated", "ablation_v21_1_p3_distortion_calibrated",
        "distortion_adaptive_dual_pyramid", "v21.1-single",
        "bounded explicit-distortion acceptance",
    ),
    "P4": Variant(
        "P4", "evidence_gate_control", "ablation_v21_1_p4_evidence_gate_control",
        "evidence_acceptance_without_explicit_distortion", "v21.1-single",
        "matched evidence-only acceptance control",
    ),
}

CORE6 = (
    Point("TaxiBJ", "fixed", "0.6"),
    Point("TaxiBJ", "random", "0.6"),
    Point("BikeNYC", "fixed", "0.4"),
    Point("BikeNYC", "random", "0.4"),
    Point("CHAP", "fixed", "0.4"),
    Point("CHAP", "random", "0.4"),
)


def expected_epochs(point: Point, override: int | None) -> int:
    return override if override is not None else FULL_EPOCHS[point.dataset]


def run_root(variant: Variant, point: Point) -> Path:
    output_variant = variant.run_name.removeprefix("ablation_")
    return (
        ROOT / "outputs" / variant.output_version / OUTPUT_NAMES[point.dataset]
        / "ablation" / output_variant / point.pattern / f"rate{point.rate}"
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
    for config_path in sorted(run_root(variant, point).glob("*/config.json"), reverse=True):
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


def read_result(run_dir: Path) -> dict:
    best = None
    test = None
    for line in (run_dir / "logs/metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("val") is not None and record.get("is_best"):
            best = {"epoch": int(record["epoch"]), **{
                key: float(record["val"].get(key, float("nan")))
                for key in ("mae", "rmse", "wape")
            }}
        if record.get("stage") == "test" and record.get("metrics") is not None:
            test = {
                key: float(record["metrics"].get(key, float("nan")))
                for key in ("mae", "rmse", "wape")
            }
    if best is None or test is None:
        raise RuntimeError(f"Incomplete metrics: {run_dir}")
    return {"validation": best, "test": test, "run_dir": str(run_dir.relative_to(ROOT))}
