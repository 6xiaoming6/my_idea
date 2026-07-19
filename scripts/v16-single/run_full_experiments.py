#!/usr/bin/env python3
"""Run the 24-point V16 matrix sequentially and skip complete train/val/test runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "v16-single"
RATES = ("0.2", "0.4", "0.6", "0.8")
DATASETS = {
    "TaxiBJ": ("TaxiBJ", "configs/v16-single/taxibj.json"),
    "BikeNYC": ("BikeNYC", "configs/v16-single/bikenyc.json"),
    "CHAP": ("CHAP_Beijing", "configs/v16-single/chap.json"),
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _is_complete(dataset: str, mask: str, rate: str, seed: int) -> tuple[Path, float] | None:
    output_name, config_name = DATASETS[dataset]
    expected_epochs = int(_load(ROOT / config_name)["train"]["epochs"])
    root = OUTPUT_ROOT / output_name / "full" / "model" / mask / f"rate{float(rate):g}"
    complete: list[tuple[Path, float]] = []
    for config_path in root.glob("*/config.json"):
        run_dir = config_path.parent
        required = (
            run_dir / "checkpoints" / "best.pt",
            run_dir / "logs" / "test.log",
            run_dir / "logs" / "metrics.jsonl",
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            continue
        try:
            cfg = _load(config_path)
            data = cfg["data"]
            mask_cfg = data["mask"]
            matches = (
                cfg.get("seed") == seed
                and cfg.get("model", {}).get("version") == "v16-single"
                and data.get("dataset_name") == output_name
                and mask_cfg.get("pattern") == mask
                and math.isclose(float(mask_cfg.get("missing_rate")), float(rate), abs_tol=1e-9)
                and int(cfg.get("train", {}).get("epochs", -1)) == expected_epochs
            )
            if not matches:
                continue
            max_epoch = 0
            test_mae = None
            for line in required[2].read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if isinstance(record.get("epoch"), int):
                    max_epoch = max(max_epoch, record["epoch"])
                if record.get("stage") == "test":
                    test_mae = float(record["metrics"]["mae"])
            if max_epoch >= expected_epochs and test_mae is not None and math.isfinite(test_mae):
                complete.append((run_dir, test_mae))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return max(complete, key=lambda item: item[0].stat().st_mtime) if complete else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--mask", choices=("all", "fixed", "random"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    masks = ("fixed", "random") if args.mask == "all" else (args.mask,)
    jobs = [(dataset, mask, rate) for mask in masks for dataset in datasets for rate in RATES]
    skipped = 0
    for index, (dataset, mask, rate) in enumerate(jobs, 1):
        done = None if args.force_rerun else _is_complete(dataset, mask, rate, args.seed)
        if done is not None:
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP {dataset} {mask}@{rate}: "
                f"test_mae={done[1]:.6f}, run={done[0].relative_to(ROOT)}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts" / "v16-single" / "train.py"),
            "--dataset", dataset,
            "--mask", mask,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(args.seed),
            "--teacher-seed", "42",
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN  {dataset} {mask}@{rate}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"[done] jobs={len(jobs)} skipped={skipped} executed={len(jobs) - skipped}")


if __name__ == "__main__":
    main()
