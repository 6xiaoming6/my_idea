#!/usr/bin/env python3
"""Run V14 cosine stage-auxiliary schedule exploration on one GPU."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/v14-exploration/supervision"
CONFIGS = {
    "C01": CONFIG_ROOT / "C01_cosine_decay_start_0.0.json",
    "C02": CONFIG_ROOT / "C02_cosine_decay_start_0.25.json",
    "C03": CONFIG_ROOT / "C03_cosine_decay_start_0.50.json",
}
SCREEN_POINTS = (
    ("TaxiBJ", "fixed", "0.4"),
    ("BikeNYC", "random", "0.8"),
    ("CHAP", "fixed", "0.8"),
)
CORE6 = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.8"),
    ("CHAP", "fixed", "0.2"),
    ("CHAP", "random", "0.4"),
)
ALL24 = tuple(
    (dataset, pattern, rate)
    for dataset in ("TaxiBJ", "BikeNYC", "CHAP")
    for pattern in ("fixed", "random")
    for rate in ("0.2", "0.4", "0.6", "0.8")
)
OUTPUT_NAMES = {
    "TaxiBJ": "TaxiBJ",
    "BikeNYC": "BikeNYC",
    "CHAP": "CHAP_Beijing",
}
FULL_EPOCHS = {"TaxiBJ": 160, "BikeNYC": 140, "CHAP": 150}


def _test_completed(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
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
    config_path: Path,
    seed: int,
    expected_epochs: int,
) -> bool:
    expected = json.loads(config_path.read_text(encoding="utf-8"))["experiment"]
    root = (
        ROOT
        / "outputs/v14-exploration"
        / OUTPUT_NAMES[dataset]
        / "ablation"
        / f"v14_exploration_{config_path.stem}"
        / pattern
        / f"rate{rate}"
    )
    for test_log in sorted(root.glob("*/logs/test.log"), reverse=True):
        if not _test_completed(test_log):
            continue
        run_dir = test_log.parent.parent
        paths = (
            run_dir / "config.json",
            run_dir / "logs/metrics.jsonl",
            run_dir / "checkpoints/best.pt",
        )
        if not all(path.is_file() for path in paths):
            continue
        config = json.loads(paths[0].read_text(encoding="utf-8"))
        if (
            int(config.get("seed", -1)) == seed
            and config.get("experiment") == expected
            and int(config.get("train", {}).get("epochs", -1)) == expected_epochs
        ):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("screen", "core6", "multiseed", "all24"),
        default="screen",
    )
    parser.add_argument(
        "--candidate",
        choices=tuple(CONFIGS),
        default=None,
        help="Required after screen; select with validation metrics only.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 2026, 3407))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[tuple[str, Path, str, str, str, int]]:
    if args.phase == "screen":
        return [
            (candidate, CONFIGS[candidate], dataset, pattern, rate, args.seed)
            for candidate in CONFIGS
            for dataset, pattern, rate in SCREEN_POINTS
        ]
    if args.candidate is None:
        raise ValueError(
            f"--candidate is required for --phase {args.phase}; "
            "select it using validation metrics only"
        )
    if args.phase == "core6":
        points, seeds = CORE6, (args.seed,)
    elif args.phase == "multiseed":
        points, seeds = CORE6, tuple(dict.fromkeys(args.seeds))
    else:
        points, seeds = ALL24, (args.seed,)
    return [
        (args.candidate, CONFIGS[args.candidate], dataset, pattern, rate, seed)
        for seed in seeds
        for dataset, pattern, rate in points
    ]


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    jobs = _jobs(args)
    for index, (candidate, config_path, dataset, pattern, rate, seed) in enumerate(
        jobs, start=1
    ):
        expected_epochs = args.epochs or FULL_EPOCHS[dataset]
        label = f"{candidate} {dataset} {pattern}@{rate} seed={seed}"
        if (
            not args.rerun_completed
            and _completed(dataset, pattern, rate, config_path, seed, expected_epochs)
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
            "--experiment-config", str(config_path.relative_to(ROOT)),
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {label}", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
