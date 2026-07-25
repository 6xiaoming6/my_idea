#!/usr/bin/env python3
"""Train one V18 BARP-MoE dataset/pattern/rate experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline_common import (
    ABLATIONS,
    DATASETS,
    RATES,
    ROOT,
    build_resolved_config,
    resolve_path,
)


def _environment(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(cpu_threads)
    return env


def _python(conda_env: str, *arguments: str) -> list[str]:
    conda = shutil.which("conda")
    if conda:
        return [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            *arguments,
        ]
    return [sys.executable, *arguments]


def _git_dirty() -> bool | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        status = subprocess.run(
            [
                git,
                "status",
                "--porcelain",
                "--ignore-submodules=dirty",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(status.stdout.strip())


def _ensure_masks(
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    spec = DATASETS[args.dataset]
    mask_dir = ROOT / spec["mask_root"] / f"{args.mask}_mask" / args.rate
    expected = tuple(mask_dir / f"{split}.csv" for split in ("train", "val", "test"))
    if args.regenerate_masks or not all(path.is_file() for path in expected):
        command = _python(
            args.conda_env,
            "scripts/generate_fixed_masks.py",
            "--train_npz",
            spec["train"],
            "--val_npz",
            spec["val"],
            "--test_npz",
            spec["test"],
            "--pattern",
            args.mask,
            "--mask_rate",
            args.rate,
            "--seed",
            str(args.seed),
            "--output_dir",
            str(mask_dir.relative_to(ROOT)),
        )
        print("[mask]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--mask", choices=("fixed", "random"), required=True)
    parser.add_argument("--rate", choices=RATES, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", choices=("none", *ABLATIONS), default="none")
    parser.add_argument("--training-policy", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--smoke-config",
        action="store_true",
        help="Merge configs/v18-single/smoke.json after the dataset config.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--regenerate-masks", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an uncommitted tree for debugging; do not use for paper results.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    dirty = _git_dirty()
    if dirty is True and not (args.allow_dirty or args.dry_run):
        raise RuntimeError(
            "Formal V18 training requires a clean Git tree. Commit/stash changes "
            "or use --allow-dirty only for non-paper debugging."
        )
    if dirty is None:
        print(
            "[warning] Git status is unavailable; the shared trainer will record "
            "git metadata as unavailable.",
            flush=True,
        )

    spec = DATASETS[args.dataset]
    required = ("base", "model")
    required_data = ("train", "val", "test")
    for key in (*required, *required_data):
        path = ROOT / spec[key]
        if not path.is_file() and not (args.dry_run and key in required_data):
            raise FileNotFoundError(path)

    policy_path = resolve_path(args.training_policy) if args.training_policy else None
    env = _environment(args.gpu, args.cpu_threads)
    _ensure_masks(args, env)
    config = build_resolved_config(
        args.dataset,
        args.mask,
        args.rate,
        args.seed,
        ablation=args.ablation,
        smoke=args.smoke_config,
        epochs=args.epochs,
        training_policy=policy_path,
    )

    if args.run_name:
        run_name = args.run_name
    elif args.ablation == "none":
        run_name = "full"
    else:
        run_name = f"ablation_v18_{args.ablation}"

    with tempfile.TemporaryDirectory(prefix="v18_single_") as directory:
        override = Path(directory) / "resolved_v18.json"
        override.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = _python(
            args.conda_env,
            "scripts/train.py",
            "-c",
            spec["base"],
            "--override_config",
            str(override),
            "--train_npz",
            spec["train"],
            "--val_npz",
            spec["val"],
            "--test_npz",
            spec["test"],
            "--name",
            run_name,
            "--no_plot",
            "--quiet",
        )
        print(
            f"[resolved] {args.dataset} {args.mask}@{args.rate} "
            f"seed={args.seed} epochs={config['train']['epochs']} "
            f"val_epoch={config['train']['val_epoch']} "
            f"ablation={args.ablation}",
            flush=True,
        )
        print("[train]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
