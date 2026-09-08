#!/usr/bin/env python3
"""Sequential single-GPU Core-6; skip only matching, fully completed jobs."""
import argparse
import json
import math
import subprocess
import sys

from train import DEFAULT_CONFIG, ROOT, load_suite, patch_for


def completed(suite, variant, dataset, pattern, rate, seed, epochs):
    patch = patch_for(suite, variant, dataset, epochs)
    folder = "CHAP_Beijing" if dataset == "CHAP" else dataset
    root = ROOT / patch["output_dir"] / folder / "ablation" / f"v21_2_{variant}" / pattern / f"rate{rate}"
    for path in sorted(root.glob("*/config.json"), reverse=True):
        try:
            cfg = json.loads(path.read_text())
            if cfg.get("seed") != seed or cfg.get("experiment_policy", {}).get("fingerprint") != patch["experiment_policy"]["fingerprint"]:
                continue
            run = path.parent
            if not (run / "checkpoints/best.pt").is_file():
                continue
            records = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
            tests = [r for r in records if r.get("stage") == "test"]
            if max(int(r.get("epoch", 0)) for r in records) < patch["train"]["epochs"]:
                continue
            if not any(r.get("is_best") and r.get("val") for r in records):
                continue
            if len(tests) == 1 and all(math.isfinite(float(tests[0]["metrics"][k])) for k in ("mae", "rmse")) and "Testing finished:" in (run / "logs/test.log").read_text():
                return run
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=("TaxiBJ", "BikeNYC", "CHAP"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    suite = load_suite(args.config)
    variants = list(dict.fromkeys(args.variants or suite["default_variants"]))
    points = [p for p in suite["core6"] if not args.datasets or p[0] in args.datasets]
    jobs = [(v, *p) for v in variants for p in points]
    for i, (variant, dataset, pattern, rate) in enumerate(jobs, 1):
        old = completed(suite, variant, dataset, pattern, rate, args.seed, args.epochs)
        label = f"{variant} {dataset} {pattern}@{rate} seed={args.seed}"
        if old and not args.rerun_completed:
            print(f"[{i}/{len(jobs)}] SKIP {label}: {old}", flush=True)
            continue
        cmd = [sys.executable, str(ROOT / "scripts/v21.2-single/train.py"), "--config", args.config,
               "--variant", variant, "--dataset", dataset, "--mask", pattern, "--rate", rate,
               "--gpu", args.gpu, "--conda-env", args.conda_env, "--cpu-threads", str(args.cpu_threads), "--seed", str(args.seed)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[{i}/{len(jobs)}] RUN {label}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
