#!/usr/bin/env python3
"""Dataset-aware single-run wrapper for v9-single."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "train.py"
MODEL_CONFIGS = ROOT / "configs" / "v9-single"

DATASETS = {
    "TaxiBJ": {
        "base_config": ROOT / "configs" / "datasets" / "taxibj.json",
        "model_config": MODEL_CONFIGS / "taxibj.json",
        "mask_root": ROOT / "data" / "TaxiBJ",
        "train_npz": ROOT / "data" / "TaxiBJ" / "taxibj_train.npz",
        "val_npz": ROOT / "data" / "TaxiBJ" / "taxibj_val.npz",
        "test_npz": ROOT / "data" / "TaxiBJ" / "taxibj_test.npz",
    },
    "BikeNYC": {
        "base_config": ROOT / "configs" / "datasets" / "bikenyc.json",
        "model_config": MODEL_CONFIGS / "bikenyc.json",
        "mask_root": ROOT / "data" / "BikeNYC",
        "train_npz": ROOT / "data" / "BikeNYC" / "bikenyc_train.npz",
        "val_npz": ROOT / "data" / "BikeNYC" / "bikenyc_val.npz",
        "test_npz": ROOT / "data" / "BikeNYC" / "bikenyc_test.npz",
    },
    "CHAP": {
        "base_config": ROOT / "configs" / "datasets" / "chap_beijing.json",
        "model_config": MODEL_CONFIGS / "chap.json",
        "mask_root": ROOT / "data" / "CHAP" / "beijing",
        "train_npz": ROOT / "data" / "CHAP" / "beijing" / "chap_beijing_train.npz",
        "val_npz": ROOT / "data" / "CHAP" / "beijing" / "chap_beijing_val.npz",
        "test_npz": ROOT / "data" / "CHAP" / "beijing" / "chap_beijing_test.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one v9-single training job.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mask-pattern", choices=("fixed", "random"), default="random")
    parser.add_argument("--mask-rate", choices=("0.2", "0.4", "0.6", "0.8"), default="0.4")
    parser.add_argument("--fixed-seed", type=int, default=42)
    parser.add_argument("--name", default="full")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _deep_update(base: dict, patch: dict) -> dict:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _generate_masks(args: argparse.Namespace, spec: dict) -> None:
    mask_dir = spec["mask_root"] / f"{args.mask_pattern}_mask" / args.mask_rate
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_fixed_masks.py"),
        "--train_npz",
        str(spec["train_npz"]),
        "--val_npz",
        str(spec["val_npz"]),
        "--test_npz",
        str(spec["test_npz"]),
        "--pattern",
        args.mask_pattern,
        "--mask_rate",
        args.mask_rate,
        "--seed",
        str(args.fixed_seed),
        "--output_dir",
        str(mask_dir),
    ]
    print("[run]", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _composed_override(args: argparse.Namespace, spec: dict, directory: Path) -> Path:
    model_cfg = json.loads(spec["model_config"].read_text(encoding="utf-8"))
    mask_dir = spec["mask_root"] / f"{args.mask_pattern}_mask" / args.mask_rate
    mask_cfg = {
        "data": {
            "mask": {
                "pattern": args.mask_pattern,
                "missing_rate": float(args.mask_rate),
                "train_csv": str(mask_dir / "train.csv"),
                "val_csv": str(mask_dir / "val.csv"),
                "test_csv": str(mask_dir / "test.csv"),
            }
        }
    }
    override = _deep_update(model_cfg, mask_cfg)
    path = directory / f"v9_{args.dataset}_{args.mask_pattern}_{args.mask_rate}.json"
    path.write_text(json.dumps(override, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    _generate_masks(args, spec)
    with tempfile.TemporaryDirectory(prefix="v9_single_train_") as temp:
        override_path = _composed_override(args, spec, Path(temp))
        command = [
            sys.executable,
            str(TRAIN),
            "-c",
            str(spec["base_config"]),
            "--override_config",
            str(override_path),
            "--train_npz",
            str(spec["train_npz"]),
            "--val_npz",
            str(spec["val_npz"]),
            "--test_npz",
            str(spec["test_npz"]),
            "-n",
            args.name,
        ]
        if args.no_plot:
            command.append("--no_plot")
        if args.quiet:
            command.append("--quiet")
        print("[run]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
