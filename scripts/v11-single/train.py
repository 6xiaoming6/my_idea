#!/usr/bin/env python3
"""Run one v11 confidence-calibrated MoE experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_NAMES = {"TaxiBJ": "taxibj.json", "BikeNYC": "bikenyc.json", "CHAP": "chap.json"}
ABLATIONS = (
    "gate_only_fallback",
    "topk_calibrated",
    "no_confidence",
    "confidence_no_mask",
    "confidence_no_input_feature",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("TaxiBJ", "BikeNYC", "CHAP"), required=True)
    parser.add_argument("--mask-pattern", choices=("fixed", "random"), required=True)
    parser.add_argument("--mask-rate", choices=("0.2", "0.4", "0.6", "0.8"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument(
        "--training-policy",
        default="configs/policies/full_model_paper.json",
    )
    parser.add_argument("--quick", type=int, choices=(1, 5), default=None)
    parser.add_argument("--ablation", choices=ABLATIONS, default=None)
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


def _run(args: argparse.Namespace, model_config_dir: str, run_name: str) -> None:
    policy = args.training_policy
    if args.quick is not None:
        policy = f"configs/v11-single/policies/quick_{args.quick}epoch.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiments.py"),
        "--dataset",
        args.dataset,
        "--gpu",
        args.gpu,
        "--mask-pattern",
        args.mask_pattern,
        "--mask-rate",
        args.mask_rate,
        "--experiments",
        "full",
        "--full-name",
        run_name,
        "--model-config-dir",
        model_config_dir,
        "--training-policy",
        policy,
        "--conda-env",
        args.conda_env,
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.ablation is None:
        _run(args, "configs/v11-single", "full")
        return

    config_name = DATASET_CONFIG_NAMES[args.dataset]
    base_path = ROOT / "configs" / "v11-single" / config_name
    ablation_path = ROOT / "configs" / "v11-single" / "ablations" / f"{args.ablation}.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    combined = _deep_update(base, ablation)
    with tempfile.TemporaryDirectory(prefix="v11_ablation_") as directory:
        model_dir = Path(directory)
        (model_dir / config_name).write_text(
            json.dumps(combined, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _run(args, str(model_dir), f"ablation_v11_{args.ablation}")


if __name__ == "__main__":
    main()
