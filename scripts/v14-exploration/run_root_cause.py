#!/usr/bin/env python3
"""Run V14 E01/E02/E03 root-cause experiments sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/v14-exploration/root_cause"
CORE6 = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.8"),
    ("CHAP", "fixed", "0.2"),
    ("CHAP", "random", "0.4"),
)
PRESCREEN2 = (
    ("TaxiBJ", "fixed", "0.4"),
    ("CHAP", "random", "0.4"),
)
ALL24 = tuple(
    (dataset, pattern, rate)
    for dataset in ("TaxiBJ", "BikeNYC", "CHAP")
    for pattern in ("fixed", "random")
    for rate in ("0.2", "0.4", "0.6", "0.8")
)
OUTPUT_NAMES = {"TaxiBJ": "TaxiBJ", "BikeNYC": "BikeNYC", "CHAP": "CHAP_Beijing"}


def _test_completed(test_log: Path) -> bool:
    """Accept only a finalized test log, not a file created before testing."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phase",
        choices=("screen", "core6", "all24"),
        default="screen",
        help=(
            "screen: E01 Core-6 plus E02/E03 weight grids on two points; "
            "core6/all24: use one selected weight per experiment"
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("E01", "E02", "E03"),
        default=("E01", "E02", "E03"),
    )
    parser.add_argument(
        "--e02-weight",
        choices=("1e-5", "1e-4", "1e-3"),
        default="1e-4",
        help="Used by core6/all24; screen always evaluates all three.",
    )
    parser.add_argument(
        "--e03-weight",
        choices=("0.01", "0.03", "0.05"),
        default="0.03",
        help="Used by core6/all24; screen always evaluates all three.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _configs(args: argparse.Namespace) -> dict[str, tuple[Path, ...]]:
    result = {
        "E01": (CONFIG_ROOT / "E01_no_gate_penalty.json",),
        "E02": (
            tuple(sorted(CONFIG_ROOT.glob("E02_delta_scale_*.json")))
            if args.phase == "screen"
            else (CONFIG_ROOT / f"E02_delta_scale_{args.e02_weight}.json",)
        ),
        "E03": (
            tuple(sorted(CONFIG_ROOT.glob("E03_rmse_regret_*.json")))
            if args.phase == "screen"
            else (CONFIG_ROOT / f"E03_rmse_regret_{args.e03_weight}.json",)
        ),
    }
    return result


def _points(phase: str, experiment: str) -> tuple[tuple[str, str, str], ...]:
    if phase == "all24":
        return ALL24
    if phase == "screen" and experiment in {"E02", "E03"}:
        return PRESCREEN2
    return CORE6


def _completed(
    dataset: str,
    pattern: str,
    rate: str,
    config_path: Path,
    seed: int,
) -> bool:
    expected_patch = json.loads(config_path.read_text(encoding="utf-8"))
    expected_experiment = expected_patch.get("experiment", {})
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
        saved_config = run_dir / "config.json"
        checkpoint = run_dir / "checkpoints/best.pt"
        if not saved_config.is_file() or not checkpoint.is_file():
            continue
        config = json.loads(saved_config.read_text(encoding="utf-8"))
        if (
            int(config.get("seed", -1)) == seed
            and config.get("experiment") == expected_experiment
        ):
            return True
    return False


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    jobs: list[tuple[str, Path, str, str, str]] = []
    configs = _configs(args)
    for experiment in args.experiments:
        for config_path in configs[experiment]:
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            for dataset, pattern, rate in _points(args.phase, experiment):
                jobs.append((experiment, config_path, dataset, pattern, rate))

    for index, (experiment, config_path, dataset, pattern, rate) in enumerate(jobs, 1):
        label = f"{config_path.stem} {dataset} {pattern}@{rate} seed={args.seed}"
        if (
            not args.rerun_completed
            and _completed(dataset, pattern, rate, config_path, args.seed)
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
            "--seed", str(args.seed),
            "--experiment-config", str(config_path.relative_to(ROOT)),
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
