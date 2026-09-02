#!/usr/bin/env python3
"""Run V14 and promoted exploration candidates on Core-6 model seeds."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE6 = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.8"),
    ("CHAP", "fixed", "0.2"),
    ("CHAP", "random", "0.4"),
)
CONFIGS: dict[str, Path | None] = {
    "V14": None,
    "E02": ROOT / "configs/v14-exploration/root_cause/E02_delta_scale_1e-4.json",
}
OUTPUT_NAMES = {"TaxiBJ": "TaxiBJ", "BikeNYC": "BikeNYC", "CHAP": "CHAP_Beijing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(CONFIGS),
        default=("E02",),
        help=(
            "Core-6 variants to run. Use V14 together with a candidate for "
            "matched-seed comparisons."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(42, 2026, 3407),
        help="Model seeds; offline mask CSV files remain unchanged.",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _test_completed(test_log: Path) -> bool:
    try:
        text = test_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if "Testing finished:" not in text:
        return False
    values = []
    for metric in ("mae", "rmse"):
        match = re.search(rf"^  {metric}:\s+(\S+)\s*$", text, re.MULTILINE)
        if match is None:
            return False
        try:
            values.append(float(match.group(1)))
        except ValueError:
            return False
    return all(math.isfinite(value) for value in values)


def _completed(
    dataset: str,
    pattern: str,
    rate: str,
    config_path: Path | None,
    seed: int,
) -> bool:
    if config_path is None:
        expected = None
        root = (
            ROOT
            / "outputs/v14-single"
            / OUTPUT_NAMES[dataset]
            / "full"
            / "model"
            / pattern
            / f"rate{rate}"
        )
    else:
        expected = json.loads(
            config_path.read_text(encoding="utf-8")
        )["experiment"]
        variant = f"v14_exploration_{config_path.stem}"
        root = (
            ROOT
            / "outputs/v14-exploration"
            / OUTPUT_NAMES[dataset]
            / "ablation"
            / variant
            / pattern
            / f"rate{rate}"
        )
    for test_log in sorted(root.glob("*/logs/test.log"), reverse=True):
        if not _test_completed(test_log):
            continue
        run_dir = test_log.parent.parent
        config_file = run_dir / "config.json"
        checkpoint = run_dir / "checkpoints/best.pt"
        if not config_file.is_file() or not checkpoint.is_file():
            continue
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if (
            int(config.get("seed", -1)) == seed
            and config.get("experiment") == expected
        ):
            return True
    return False


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    jobs = [
        (candidate, CONFIGS[candidate], seed, dataset, pattern, rate)
        for candidate in args.candidates
        for seed in args.seeds
        for dataset, pattern, rate in CORE6
    ]
    for index, (candidate, config_path, seed, dataset, pattern, rate) in enumerate(jobs, 1):
        label = f"{candidate} {dataset} {pattern}@{rate} seed={seed}"
        if (
            not args.rerun_completed
            and _completed(dataset, pattern, rate, config_path, seed)
        ):
            print(f"[{index}/{len(jobs)}] SKIP complete {label}", flush=True)
            continue
        command = [
            sys.executable,
            "scripts/v14-single/train.py",
            "--dataset", dataset,
            "--mask", pattern,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(seed),
        ]
        if config_path is not None:
            command.extend(
                ["--experiment-config", str(config_path.relative_to(ROOT))]
            )
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
