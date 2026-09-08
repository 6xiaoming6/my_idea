#!/usr/bin/env python3
"""Train V22 without inheriting V14/V21 architecture or losses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v22/experiments.json"
DATASETS = {
    "TaxiBJ": ("taxibj", "data/TaxiBJ", "taxibj"),
    "BikeNYC": ("bikenyc", "data/BikeNYC", "bikenyc"),
    "CHAP": ("chap_beijing", "data/CHAP/beijing", "chap_beijing"),
}


def merge(base, patch):
    result = dict(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_suite(path=DEFAULT_CONFIG, profile=None):
    suite = json.loads(resolve(path).read_text(encoding="utf-8"))
    profiles = suite.pop("profiles", {})
    if profile is not None:
        if profile not in profiles:
            raise ValueError(f"Unknown experiment profile: {profile}; choices: {list(profiles)}")
        suite = merge(suite, profiles[profile])
    return suite


def build_config(suite, variant, dataset, pattern, rate, seed, epochs=None, smoke=False, cpu=False, world_size=1):
    if variant not in suite["variants"] or dataset not in DATASETS:
        raise ValueError(f"Unknown variant/dataset: {variant}/{dataset}")
    if pattern not in {"fixed", "random"} or str(rate) not in {"0.2", "0.4", "0.6", "0.8"}:
        raise ValueError("Unsupported mask pattern/rate")
    base, folder, prefix = DATASETS[dataset]
    cfg = json.loads((ROOT / f"configs/datasets/{base}.json").read_text())
    for patch in (suite["common"], suite["datasets"][dataset], suite["variants"][variant]):
        cfg = merge(cfg, patch)
    # Dataset presets also contain legacy prediction-MoE knobs. Strip inactive
    # options so config/logs describe the actual V22 model, not a fictional stack.
    cfg["model"] = {k: cfg["model"][k] for k in ("c_in", "version", "architecture", "v22", "aux")}
    cfg["model"]["main"] = {"use_router": False, "use_routed_branch": False}
    cfg["loss"] = {k: cfg["loss"][k] for k in ("type", "lambda_v22_mass", "lambda_v22_balance")}
    if cfg["model"]["aux"].get("enabled"):
        raise ValueError("V22 controlled experiments require aux.enabled=false")
    if cfg["train"].get("early_stopping", {}).get("enabled"):
        raise ValueError("V22 fixed-budget protocol requires early_stopping.enabled=false")
    cfg = merge(cfg, {"seed": seed, "device": "cpu" if cpu else "cuda:0", "data": {"mask": {
        "pattern": pattern, "missing_rate": float(rate),
        **{f"{split}_csv": f"{folder}/{pattern}_mask/{rate}/{split}.csv" for split in ("train", "val", "test")}
    }}})
    if epochs is not None:
        if epochs < 1:
            raise ValueError("--epochs must be positive")
        cfg["train"].update(epochs=epochs, val_epoch=min(cfg["train"]["val_epoch"], epochs))
    if smoke:
        # Synthetic pipeline test only; separate namespace can NEVER satisfy a real job.
        cfg["output_dir"] = "outputs/v22/smoke"
        cfg["train"].update(epochs=1, val_epoch=1, amp=False)
        cfg["data"].update(batch_size=1, num_workers=0, pin_memory=False, drop_last=False)
        cfg["data"]["synthetic"].update(num_train=2, num_val=2)
    if int(cfg["train"]["epochs"]) < 1 or int(cfg["train"]["val_epoch"]) < 1:
        raise ValueError("epochs and val_epoch must be positive")
    if world_size > 1:
        per_rank = int(suite.get("distributed", {}).get("per_rank_batch_size", 4))
        if per_rank < 1:
            raise ValueError("distributed.per_rank_batch_size must be positive")
        cfg["data"]["batch_size"] = 1 if smoke else per_rank
        cfg["distributed"] = {
            "world_size": world_size, "per_rank_batch_size": cfg["data"]["batch_size"],
            "global_batch_size": world_size * cfg["data"]["batch_size"],
            "timeout_seconds": int(suite.get("distributed", {}).get("timeout_seconds", 1800)),
        }
    # Fingerprint actual config, source, and data identities. No dependence on Git availability.
    paths = {s: f"{folder}/{prefix}_{s}.npz" for s in ("train", "val", "test")}
    identity = {}
    if not smoke:
        for path in [*paths.values(), *[cfg["data"]["mask"][f"{s}_csv"] for s in paths]]:
            file = ROOT / path
            identity[path] = [file.stat().st_size, file.stat().st_mtime_ns] if file.is_file() else None
    digest = hashlib.sha256(json.dumps({"config": cfg, "data_identity": identity}, sort_keys=True).encode())
    sources = sorted((ROOT / "src/stmoe_imputer").rglob("*.py"))
    sources += [ROOT / "scripts/train.py", Path(__file__)]
    if world_size > 1:
        sources.append(ROOT / "scripts/v22/train_ddp.py")
    for source in sources:
        digest.update(str(source.relative_to(ROOT)).encode())
        digest.update(source.read_bytes())
    cfg["experiment_policy"] = {
        "name": f"v22_{variant}", "fingerprint": digest.hexdigest(),
        "selection": "validation_mae", "smoke": smoke,
        "data_identity": identity, "normalization": "observed_only_sample_channel",
    }
    return cfg, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", default=None, help="Named screening policy from the JSON config")
    parser.add_argument("--variant", default="moe")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--mask", choices=("fixed", "random"), required=True)
    parser.add_argument("--rate", choices=("0.2", "0.4", "0.6", "0.8"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", default="0")
    gpu_group.add_argument("--gpus", nargs="+", help="Two or more IDs launch one DDP experiment")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--conda-env", default="difftdi", help="Use 'current' to use this Python")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true", help="Synthetic one-epoch train/val/test; NOT real-data evidence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    devices = args.gpus or [args.gpu]
    if args.cpu_threads < 1 or len(set(devices)) != len(devices) or any(not g.isdigit() for g in devices):
        parser.error("Use distinct numeric GPU IDs and positive --cpu-threads")
    cfg, paths = build_config(load_suite(args.config, args.profile), args.variant, args.dataset, args.mask, args.rate,
                              args.seed, args.epochs, args.smoke, args.cpu, world_size=len(devices))
    if not args.dry_run and not args.smoke:
        for path in [*paths.values(), *[cfg["data"]["mask"][f"{s}_csv"] for s in paths]]:
            if not (ROOT / path).is_file():
                raise FileNotFoundError(f"Required real data/mask missing: {ROOT / path}; generate offline masks first")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if args.cpu else ",".join(devices), PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.cpu_threads)
    interpreter = [sys.executable] if args.conda_env == "current" else ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python"]
    with tempfile.TemporaryDirectory(prefix="v22_") as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        entry = ["scripts/train.py"] if len(devices) == 1 else [
            "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={len(devices)}", "--max_restarts=0", "scripts/v22/train_ddp.py"]
        command = interpreter + entry + ["-c", str(path), "--name", f"ablation_v22_{args.variant}", "--no_plot", "--quiet"]
        command += ["--synthetic"] if args.smoke else [a for s, p in paths.items() for a in (f"--{s}_npz", p)]
        print(f"[V22] {args.variant} {args.dataset} {args.mask}@{args.rate}; epochs={cfg['train']['epochs']} val_epoch={cfg['train']['val_epoch']}", flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            print(json.dumps(cfg, indent=2), flush=True)
        else:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
