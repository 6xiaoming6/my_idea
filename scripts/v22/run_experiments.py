#!/usr/bin/env python3
"""Sequential experiments, optionally multi-GPU DDP within each experiment."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys

from train import DEFAULT_CONFIG, ROOT, build_config, load_suite


def completed(cfg, variant):
    mask = cfg["data"]["mask"]
    root = ROOT / cfg["output_dir"] / cfg["data"]["dataset_name"] / "ablation" / f"v22_{variant}" / mask["pattern"] / f"rate{mask['missing_rate']}"
    for path in sorted(root.glob("*/config.json"), reverse=True):
        try:
            saved = json.loads(path.read_text())
            if saved.get("experiment_policy", {}).get("fingerprint") != cfg["experiment_policy"]["fingerprint"]:
                continue
            run = path.parent
            records = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
            epochs = [r for r in records if "train" in r and "epoch" in r]
            tests = [r for r in records if r.get("stage") == "test"]
            best = [r for r in epochs if r.get("is_best") and r.get("val")]
            if not epochs or max(r["epoch"] for r in epochs) < cfg["train"]["epochs"] or not best or len(tests) != 1:
                continue
            if not (run / "checkpoints/best.pt").is_file() or "Testing finished:" not in (run / "logs/test.log").read_text():
                continue
            if "Training finished normally:" not in (run / "logs/train.log").read_text():
                continue
            if tests[0].get("extra", {}).get("best_epoch") != best[-1]["epoch"]:
                continue
            if not math.isfinite(float(best[-1]["val"]["mae"])):
                continue
            if not all(math.isfinite(float(tests[0]["metrics"][k])) for k in ("mae", "rmse")):
                continue
            return {"run": str(run), "best_epoch": best[-1]["epoch"], "val_mae": best[-1]["val"]["mae"],
                    "test_mae": tests[0]["metrics"]["mae"], "test_rmse": tests[0]["metrics"]["rmse"]}
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", default=None, help="Use --profile quick for the small screening study")
    parser.add_argument("--points", choices=("core6", "full24"), default=None)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=("TaxiBJ", "BikeNYC", "CHAP"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", default="0")
    gpu_group.add_argument("--gpus", nargs="+", help="Co-train each experiment with DDP")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Read-only console/JSON summary; does not launch training")
    completion_group = parser.add_mutually_exclusive_group()
    completion_group.add_argument("--skip-completed", dest="rerun_completed", action="store_false",
                                  help="Default: skip matching runs with complete train/val/test and best checkpoint")
    completion_group.add_argument("--rerun-completed", action="store_true",
                                  help="Explicitly rerun even successfully completed experiments")
    parser.set_defaults(rerun_completed=False)
    args = parser.parse_args()
    devices = args.gpus or [args.gpu]
    if len(set(devices)) != len(devices) or any(not g.isdigit() for g in devices):
        parser.error("GPU IDs must be distinct nonnegative integers")
    suite = load_suite(args.config, args.profile)
    variants = list(dict.fromkeys(args.variants or suite["default_variants"]))
    if args.points == "full24":
        points = [(d, m, r) for m in ("fixed", "random") for d in suite["datasets"] for r in ("0.2", "0.4", "0.6", "0.8")]
    elif args.points == "core6":
        points = suite["core6"]
    else:
        points = suite.get("default_points", suite["core6"])
    points = [p for p in points if not args.datasets or p[0] in args.datasets]
    if not points:
        parser.error("No points remain after dataset filtering; quick only includes BikeNYC. Use --points core6 to expand.")
    # Validate the entire queue before starting any expensive job.
    jobs = [(v, d, m, str(r), s, build_config(suite, v, d, m, str(r), s, args.epochs, cpu=args.cpu, world_size=len(devices))[0])
            for d, m, r in points for s in dict.fromkeys(args.seeds) for v in variants]
    rows = []
    existing = [completed(cfg, variant) for variant, _, _, _, _, cfg in jobs]
    if not args.summary:
        skip_count = sum(old is not None for old in existing) if not args.rerun_completed else 0
        pending_epochs = sum(job[-1]['train']['epochs'] for job, old in zip(jobs, existing)
                             if old is None or args.rerun_completed)
        print(f"[plan] profile={args.profile or 'full'} jobs={len(jobs)} world_size={len(devices)} "
              f"completed={sum(old is not None for old in existing)} skip={skip_count} "
              f"pending={len(jobs) - skip_count} pending_epoch_budget={pending_epochs}", flush=True)
        print("[resume] Completed matching train/val/test runs are preserved. "
              "Incomplete runs restart from epoch 1 in a new output directory; no checkpoint resume.", flush=True)
    for i, ((variant, dataset, pattern, rate, seed, cfg), old) in enumerate(zip(jobs, existing), 1):
        label = f"{variant} {dataset} {pattern}@{rate} seed={seed}"
        if args.summary:
            rows.append({"variant": variant, "dataset": dataset, "pattern": pattern, "rate": rate, "seed": seed,
                         "status": "complete" if old else "missing_or_incompatible", **(old or {})})
            continue
        if old and not args.rerun_completed:
            print(f"[{i}/{len(jobs)}] SKIP {label}: {old['run']}", flush=True)
            continue
        command = [sys.executable, str(ROOT / "scripts/v22/train.py"), "--config", str(args.config), "--variant", variant,
                   "--dataset", dataset, "--mask", pattern, "--rate", rate, "--seed", str(seed),
                   "--cpu-threads", str(args.cpu_threads), "--conda-env", args.conda_env]
        command += ["--gpus", *devices] if len(devices) > 1 else ["--gpu", devices[0]]
        if args.profile:
            command += ["--profile", args.profile]
        if args.epochs is not None:
            command += ["--epochs", str(args.epochs)]
        if args.cpu:
            command.append("--cpu")
        if args.dry_run:
            print(f"[{i}/{len(jobs)}] WOULD RUN {label}\n{shlex.join(command)}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] RUN {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    if args.summary:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"Completed {sum(r['status'] == 'complete' for r in rows)}/{len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
