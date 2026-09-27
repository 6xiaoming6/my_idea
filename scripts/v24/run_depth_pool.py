#!/usr/bin/env python3
"""Run four CoE depth/expert-pool candidates sequentially on one GPU."""
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
POLICY = ROOT / "configs/v24/depth_pool_experiments.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_jobs(policy: dict, epochs: int | None = None) -> list[dict]:
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        policy = copy.deepcopy(policy)
        policy["epochs"] = epochs
        policy["scheduler"] = copy.deepcopy(policy.get("scheduler", {}))
        policy["scheduler"]["total_epochs"] = epochs
    effective_epochs = int(policy["epochs"])
    base = load(ROOT / policy["base_config"])
    mask_root = ROOT / "data/TaxiBJ/random_mask/0.4"
    masks = {split: mask_root / f"{split}.csv" for split in ("train", "val", "test")}
    missing = [str(p) for p in masks.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing TaxiBJ random0.4 masks: " + ", ".join(missing))
    jobs = []
    for variant in policy["variants"]:
        name = f"v24_{variant['name']}_random_rate0.4_e{effective_epochs}_seed{policy['seed']}"
        cfg = update(base, {"model": {"coe": {
            "num_steps": variant["num_steps"],
            "expert_pool": variant["expert_pool"],
            "top_k": variant["top_k"],
            "fixed_path": None,
            "fixed_expert_steps": [None] * variant["num_steps"],
            "routing_mode": "hard",
            "router_state": "dynamic",
            "expert_state": "dynamic",
            "routing_warmup_epochs": 0,
            "routing_transition_epochs": 0,
        }}})
        cfg["seed"] = policy["seed"]
        cfg["output_dir"] = policy["output_dir"]
        cfg["data"] = update(cfg["data"], {
            "batch_size": policy["batch_size"],
            "mask": {"pattern": "random", "missing_rate": 0.4,
                      "train_csv": str(masks["train"]),
                      "val_csv": str(masks["val"]),
                      "test_csv": str(masks["test"])},
            "train_mask_diversity": None,
            "eval_mask_diversity": None,
        })
        cfg["train"] = update(cfg["train"], {
            "epochs": policy["epochs"], "val_epoch": policy["val_epoch"],
            "save_best_checkpoint": policy["save_best_checkpoint"],
            "early_stopping": policy["early_stopping"],
            "scheduler": policy["scheduler"],
        })
        cfg["experiment_plan"] = {"stage": policy.get("stage", "coe_depth_pool"), "variant": variant["name"],
                                   "protocol": "random_rate0.4", "protocol_kind": "legacy_csv"}
        control = ROOT / policy["control_dir"]
        jobs.append({"name": name, "config": cfg,
                     "config_path": control / "configs" / f"{name}.json",
                     "result_path": control / "results" / f"{name}.json",
                     "log_path": control / "launcher_logs" / f"{name}.log",
                     "num_steps": variant["num_steps"], "top_k": variant["top_k"],
                     "num_experts": len(variant["expert_pool"])})
    return jobs


def check_gpu_idle(gpu: int) -> None:
    visible = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    if str(gpu) not in visible.stdout.split():
        raise RuntimeError(f"GPU {gpu} does not exist")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    if processes.stdout.strip():
        raise RuntimeError("A GPU already has a compute process; depth-pool queue was not started:\n" + processes.stdout)


def main(policy_path: Path = POLICY) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=None, help="Override the epoch count for every queued experiment.")
    args = p.parse_args()
    policy = load(policy_path)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be a positive integer")
        policy["epochs"] = args.epochs
        policy["scheduler"] = copy.deepcopy(policy.get("scheduler", {}))
        policy["scheduler"]["total_epochs"] = args.epochs
    jobs = build_jobs(policy, args.epochs)
    print(json.dumps({"num_runs": len(jobs), "epochs": policy["epochs"],
                      "order": [{"name": j["name"], "steps": j["num_steps"], "top_k": j["top_k"], "experts": j["num_experts"]} for j in jobs]}, ensure_ascii=False, indent=2))
    control = ROOT / policy["control_dir"]
    control.mkdir(parents=True, exist_ok=True)
    (control / "plan.json").write_text(json.dumps({"status": "planned_not_executed", "jobs": [{"name": j["name"], "steps": j["num_steps"], "top_k": j["top_k"], "experts": j["num_experts"]} for j in jobs]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        return 0
    for sub in ("configs", "results", "launcher_logs"):
        (control / sub).mkdir(parents=True, exist_ok=True)
    with (control / ".single_gpu_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Depth-pool queue already running") from None
        check_gpu_idle(args.gpu)
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for index, job in enumerate(jobs, 1):
            job["config_path"].parent.mkdir(parents=True, exist_ok=True)
            job["config_path"].write_text(json.dumps(job["config"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if job["result_path"].exists():
                print(f"[{index}/{len(jobs)}] skip {job['name']}", flush=True); continue
            command = [PYTHON, "-u", str(ROOT / "scripts/train.py"), "-c", str(job["config_path"]),
                       "--train_npz", str(ROOT / policy["train_npz"]), "--val_npz", str(ROOT / policy["val_npz"]),
                       "--test_npz", str(ROOT / policy["test_npz"]), "--no_plot", "-n", job["name"],
                       "--result-file", str(job["result_path"])]
            print(f"[{index}/{len(jobs)}] start {job['name']}", flush=True)
            with job["log_path"].open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, bufsize=0)
                assert process.stdout is not None
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    log.write(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
                code = process.wait()
            if code:
                raise RuntimeError(f"{job['name']} failed (exit {code}); see {job['log_path']}")
            print(f"[{index}/{len(jobs)}] complete {job['name']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
