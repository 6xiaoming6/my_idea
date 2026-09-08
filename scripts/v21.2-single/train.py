#!/usr/bin/env python3
"""Single-stage V21.2 training; all model knobs live in experiments.json."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v21.2-single/experiments.json"
spec = importlib.util.spec_from_file_location("v14_runner", ROOT / "scripts/v14-single/train.py")
v14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v14)


def load_suite(path):
    path = Path(path).expanduser()
    return json.loads((path if path.is_absolute() else ROOT / path).read_text())


def patch_for(suite, variant, dataset, epochs=None):
    if variant not in suite["variants"]:
        raise ValueError(f"Unknown variant: {variant}")
    patch = v14._deep_update(suite["common"], suite["datasets"][dataset])
    patch = v14._deep_update(patch, suite["variants"][variant])
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        patch = v14._deep_update(patch, {"train": {"epochs": epochs, "val_epoch": 1}})
    # Include inherited dataset settings and source bytes so a changed protocol
    # cannot silently reuse an old complete result.
    effective = v14._deep_update(v14._load(ROOT / v14.DATASETS[dataset]["base"]), v14._load(ROOT / v14.DATASETS[dataset]["model"]))
    effective = v14._deep_update(effective, patch)
    digest = hashlib.sha256(json.dumps(effective, sort_keys=True).encode())
    sources = sorted((ROOT / "src/stmoe_imputer").rglob("*.py"))
    sources += [ROOT / "scripts/train.py", ROOT / "scripts/v14-single/train.py", Path(__file__)]
    for source in sources:
        digest.update(str(source.relative_to(ROOT)).encode())
        digest.update(source.read_bytes())
    patch["experiment_policy"] = {"name": f"v21_2_{variant}", "fingerprint": digest.hexdigest(), "selection": "validation_mae", "risk_is_proxy": True}
    return patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--variant", default="risk")
    parser.add_argument("--dataset", choices=tuple(v14.DATASETS), required=True)
    parser.add_argument("--mask", choices=("fixed", "random"), required=True)
    parser.add_argument("--rate", choices=v14.RATES, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    patch = patch_for(load_suite(args.config), args.variant, args.dataset, args.epochs)
    with tempfile.TemporaryDirectory(prefix="v21_2_") as directory:
        path = Path(directory) / "override.json"
        path.write_text(json.dumps(patch, indent=2))
        command = [sys.executable, str(ROOT / "scripts/v14-single/train.py"),
                   "--dataset", args.dataset, "--mask", args.mask, "--rate", args.rate,
                   "--gpu", args.gpu, "--conda-env", args.conda_env,
                   "--cpu-threads", str(args.cpu_threads), "--seed", str(args.seed),
                   "--experiment-config", str(path), "--run-name", f"ablation_v21_2_{args.variant}"]
        if args.dry_run:
            print(json.dumps(patch, indent=2), flush=True)
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
