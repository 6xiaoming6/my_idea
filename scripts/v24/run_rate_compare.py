#!/usr/bin/env python3
"""Run A0/A2 on TaxiBJ random masks at four missing rates, one GPU."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
POLICY = ROOT / "configs/v24/rate_compare_experiments.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def paths(policy: dict, pattern: str, rate: float) -> dict:
    root = ROOT / "data/TaxiBJ" / f"{pattern}_mask" / format(rate, "g")
    result = {split: root / f"{split}.csv" for split in ("train", "val", "test")}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing mask CSVs: " + ", ".join(missing))
    return result


def build_jobs(policy: dict) -> list[dict]:
    base = load(ROOT / policy["base_config"])
    jobs = []
    for pattern in policy["patterns"]:
        for rate in policy["rates"]:
            mask_paths = paths(policy, pattern, rate)
            for variant in policy["variants"]:
                name = f"v24_coe_rate_compare_{variant}_{pattern}_rate{format(rate, 'g')}_seed{policy['seed']}"
                cfg = deep_update(base, policy["variants_config"].get(variant, {}))
                cfg["seed"] = policy["seed"]
                cfg["output_dir"] = policy["output_dir"]
                cfg["data"] = deep_update(cfg["data"], {
                    "batch_size": policy["batch_size"],
                    "mask": {"pattern": pattern, "missing_rate": rate,
                              "train_csv": str(mask_paths["train"]),
                              "val_csv": str(mask_paths["val"]),
                              "test_csv": str(mask_paths["test"])},
                    "train_mask_diversity": None,
                    "eval_mask_diversity": None,
                })
                cfg["train"] = deep_update(cfg["train"], {
                    "epochs": policy["epochs"], "val_epoch": policy["val_epoch"],
                    "save_best_checkpoint": policy["save_best_checkpoint"],
                    "early_stopping": policy["early_stopping"],
                    "scheduler": policy["scheduler"],
                })
                cfg["experiment_plan"] = {
                    "stage": "coe_rate_compare", "variant": variant,
                    "protocol": f"{pattern}_rate{format(rate, 'g')}",
                    "protocol_kind": "legacy_csv", "training_mask_source": "csv",
                    "evaluation_mask_source": "fixed_csv_per_split",
                }
                cfg_path = ROOT / policy["output_dir"] / "configs" / f"{name}.json"
                result_path = ROOT / policy["output_dir"] / "results" / f"{name}.json"
                log_path = ROOT / policy["output_dir"] / "launcher_logs" / f"{name}.log"
                jobs.append({"name": name, "variant": variant, "pattern": pattern,
                             "rate": rate, "config": cfg, "config_path": cfg_path,
                             "result_path": result_path, "log_path": log_path})
    return jobs


def check_gpu_idle(gpu: int) -> None:
    visible = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                             check=True, capture_output=True, text=True)
    if str(gpu) not in visible.stdout.split():
        raise RuntimeError(f"GPU {gpu} does not exist")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                "--format=csv,noheader"], check=True, capture_output=True, text=True)
    if processes.stdout.strip():
        raise RuntimeError("A GPU already has a compute process; queue not started:\n" + processes.stdout)


def write_plan(jobs: list[dict], policy: dict) -> None:
    plan = {"status": "planned_not_executed", "num_runs": len(jobs),
            "epochs": policy["epochs"], "variants": policy["variants"],
            "patterns": policy["patterns"], "rates": policy["rates"],
            "jobs": [{k: j[k] for k in ("name", "variant", "pattern", "rate")} for j in jobs]}
    path = ROOT / policy["output_dir"] / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    policy = load(POLICY)
    jobs = build_jobs(policy)
    write_plan(jobs, policy)
    print(json.dumps({"num_runs": len(jobs), "epochs": policy["epochs"],
                      "order": [j["name"] for j in jobs]}, ensure_ascii=False, indent=2))
    if args.dry_run or args.summary_only:
        return 0
    output = ROOT / policy["output_dir"]
    for sub in ("configs", "results", "launcher_logs"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    with (output / ".single_gpu_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Rate comparison queue already running") from None
        check_gpu_idle(args.gpu)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for index, job in enumerate(jobs, start=1):
            job["config_path"].parent.mkdir(parents=True, exist_ok=True)
            job["config_path"].write_text(json.dumps(job["config"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if job["result_path"].exists():
                print(f"[{index}/{len(jobs)}] skip completed {job['name']}", flush=True)
                continue
            command = [PYTHON, "-u", str(ROOT / "scripts/train.py"), "-c", str(job["config_path"]),
                       "--train_npz", str(ROOT / policy["train_npz"]), "--val_npz", str(ROOT / policy["val_npz"]),
                       "--test_npz", str(ROOT / policy["test_npz"]), "--no_plot", "-n", job["name"],
                       "--result-file", str(job["result_path"])]
            print(f"[{index}/{len(jobs)}] start {job['name']}", flush=True)
            with job["log_path"].open("w", encoding="utf-8") as log:
                code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                raise RuntimeError(f"{job['name']} failed (exit {code}); see {job['log_path']}")
            print(f"[{index}/{len(jobs)}] complete {job['name']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
