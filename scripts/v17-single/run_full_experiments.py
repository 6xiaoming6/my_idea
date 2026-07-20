#!/usr/bin/env python3
"""Run V17 sequentially (fixed first, then random) and skip completed tests."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "v17-single"
RATES = ("0.2", "0.4", "0.6", "0.8")
OUTPUT_NAMES = {"TaxiBJ": "TaxiBJ", "BikeNYC": "BikeNYC", "CHAP": "CHAP_Beijing"}


@dataclass(frozen=True)
class Job:
    dataset: str
    mask: str
    rate: str

    @property
    def label(self) -> str:
        return f"{self.dataset} {self.mask}@{self.rate}"

    @property
    def root(self) -> Path:
        return (
            OUTPUT_ROOT
            / OUTPUT_NAMES[self.dataset]
            / "full"
            / "model"
            / self.mask
            / f"rate{float(self.rate):g}"
        )


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _has_finite_test(metrics_path: Path) -> bool:
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("stage") != "test":
                continue
            metrics = record.get("metrics", {})
            return all(
                math.isfinite(float(metrics[name])) for name in ("mae", "rmse", "loss")
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return False


def completed_run(job: Job, seed: int) -> Path | None:
    if not job.root.is_dir():
        return None
    complete = []
    for config_path in job.root.glob("*/config.json"):
        run_dir = config_path.parent
        checkpoint = run_dir / "checkpoints" / "best.pt"
        test_log = run_dir / "logs" / "test.log"
        metrics = run_dir / "logs" / "metrics.jsonl"
        if not all(path.is_file() and path.stat().st_size > 0 for path in (checkpoint, test_log, metrics)):
            continue
        try:
            config = _load(config_path)
            mask_cfg = config.get("data", {}).get("mask", {})
            matches = (
                config.get("model", {}).get("version") == "v17-single"
                and config.get("model", {}).get("architecture") == "v17_hierarchical_scale_moe"
                and config.get("data", {}).get("dataset_name") == OUTPUT_NAMES[job.dataset]
                and mask_cfg.get("pattern") == job.mask
                and math.isclose(float(mask_cfg.get("missing_rate")), float(job.rate), abs_tol=1e-9)
                and int(config.get("seed", -1)) == seed
            )
        except (OSError, ValueError, TypeError):
            matches = False
        if matches and _has_finite_test(metrics):
            complete.append(run_dir)
    return max(complete, key=lambda path: path.stat().st_mtime) if complete else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", choices=tuple(OUTPUT_NAMES), default=list(OUTPUT_NAMES))
    parser.add_argument("--masks", nargs="+", choices=("fixed", "random"), default=["fixed", "random"])
    parser.add_argument("--rates", nargs="+", choices=RATES, default=list(RATES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-policy", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ordered_masks = [mask for mask in ("fixed", "random") if mask in args.masks]
    jobs = [
        Job(dataset, mask, rate)
        for mask in ordered_masks
        for dataset in args.datasets
        for rate in args.rates
    ]
    skipped = 0
    for index, job in enumerate(jobs, start=1):
        done = None if args.force_rerun else completed_run(job, args.seed)
        if done is not None:
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP {job.label}: {done.relative_to(ROOT)}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            "scripts/v17-single/train.py",
            "--dataset", job.dataset,
            "--mask", job.mask,
            "--rate", job.rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(args.seed),
        ]
        if args.training_policy:
            command.extend(["--training-policy", args.training_policy])
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {job.label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"[done] total={len(jobs)} skipped={skipped} executed={len(jobs)-skipped}")


if __name__ == "__main__":
    main()
