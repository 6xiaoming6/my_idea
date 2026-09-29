#!/usr/bin/env python3
"""Compare three learning-rate schedules on the direct four-step shared Top-2 MoE."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/v24/coe_main_s4_e8_taxibj_base.json"
VARIANT = ROOT / "configs/v24/experiments/direct_shared_s4_top2.json"
SOURCE_POLICY = ROOT / "configs/v24/coe_direct_baselines_taxibj_experiments.json"
OUTPUT = ROOT / "outputs/v24-COE/experiments/shared_top2_lr_sweep_20260929"
ARMS = (
    ("fixed_3e-4", 3e-4, {"type": "none"}),
    ("cosine_1e-3_to_3e-4", 1e-3, {"type": "cosine", "eta_min": 3e-4}),
    ("fixed_1e-3", 1e-3, {"type": "none"}),
)
FIELDS = ("name", "status", "best_epoch", "best_val_mae", "test_mae", "test_rmse",
          "completed_epochs", "total_time_sec", "run_dir")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def merge(base: dict, patch: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def jobs(epochs: int) -> list[dict]:
    base = merge(read_json(BASE), read_json(VARIANT))
    policy = read_json(SOURCE_POLICY)
    study = policy["studies"][policy["default_study"]]
    cfgs = []
    for name, lr, scheduler in ARMS:
        cfg = copy.deepcopy(base)
        cfg["seed"] = 7
        cfg["data"]["batch_size"] = 32
        cfg["data"]["drop_last"] = False
        cfg["train"] = merge(cfg["train"], policy["training"])
        cfg["train"].update(epochs=epochs, val_epoch=5, lr_main=lr,
                            save_best_checkpoint=True, best_checkpoint_name="best.pth")
        cfg["train"].pop("lr_router", None)
        cfg["train"].pop("lr_aux", None)
        cfg["train"].pop("lr_v14", None)
        cfg["train"]["scheduler"] = {**scheduler, **({"total_epochs": epochs - 1} if scheduler["type"] == "cosine" else {})}
        cfg["experiment_plan"] = {"suite": "shared_top2_lr_sweep_20260929", "name": name,
                                  "baseline": "direct_shared_s4_top2"}
        coe = cfg["model"]["coe"]
        assert (cfg["model"]["architecture"], coe["num_steps"], coe["top_k"],
                coe["expert_sharing"], coe["state_update_mode"], coe["pair_mode"],
                coe["completion_feedback"]) == ("v24_ts_coe", 4, 2, "shared", "direct", "native", False)
        assert cfg["data"]["train_mask_diversity"]["families"] == cfg["data"]["eval_mask_diversity"]["families"]
        assert cfg["data"]["train_mask_diversity"]["rates"] == cfg["data"]["eval_mask_diversity"]["rates"] == [0.4]
        assert cfg["train"]["val_epoch"] == 5 and cfg["data"]["batch_size"] == 32
        cfgs.append({"name": name, "config": cfg, "sources": study["sources"]})
    reference = None
    for job in cfgs:
        normalized = copy.deepcopy(job["config"])
        normalized.pop("experiment_plan")
        normalized["train"].pop("lr_main")
        normalized["train"].pop("scheduler")
        if reference is None:
            reference = normalized
        elif normalized != reference:
            raise ValueError("The LR arms differ outside learning-rate settings")
    return cfgs


def complete(job: dict, directory: Path) -> dict | None:
    for attempt in sorted(directory.glob("attempts/attempt*"), reverse=True):
        config_path, receipt_path = attempt / "input_config.json", attempt / "receipt.json"
        if not config_path.is_file() or not receipt_path.is_file():
            continue
        config, receipt = read_json(config_path), read_json(receipt_path)
        expected = copy.deepcopy(job["config"])
        expected["output_dir"] = str((attempt / "output").relative_to(ROOT))
        checkpoint = Path(receipt.get("run_dir", "")) / "checkpoints/best.pth"
        if (config == expected and receipt.get("status") == "finished"
                and receipt.get("completed_epochs") == expected["train"]["epochs"]
                and receipt.get("config_sha256") == sha(config) and checkpoint.is_file()):
            return receipt
    return None


def summary(plan: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in plan:
        receipt = complete(job, OUTPUT / job["name"])
        rows.append({"name": job["name"], "status": "complete" if receipt else "pending",
                     "best_epoch": receipt.get("best_epoch", "") if receipt else "",
                     "best_val_mae": receipt.get("best_val_mae", "") if receipt else "",
                     "test_mae": receipt.get("test", {}).get("mae", "") if receipt else "",
                     "test_rmse": receipt.get("test", {}).get("rmse", "") if receipt else "",
                     "completed_epochs": receipt.get("completed_epochs", "") if receipt else "",
                     "total_time_sec": receipt.get("total_time_sec", "") if receipt else "",
                     "run_dir": receipt.get("run_dir", "") if receipt else ""})
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(job: dict, gpu: int) -> None:
    directory = OUTPUT / job["name"]
    attempts = directory / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.removeprefix("attempt")) for path in attempts.glob("attempt[0-9]*")]
    attempt = attempts / f"attempt{max(numbers, default=0) + 1}"
    attempt.mkdir()
    config = copy.deepcopy(job["config"])
    config["output_dir"] = str((attempt / "output").relative_to(ROOT))
    config_path, receipt_path = attempt / "input_config.json", attempt / "receipt.json"
    write_json(config_path, config)
    command = [sys.executable, "-u", str(ROOT / "scripts/train.py"), "--config", str(config_path),
               "--name", f"shared_top2_lr_{job['name']}_seed7", "--no_plot",
               "--result-file", str(receipt_path)]
    for split in ("train", "val", "test"):
        command.extend((f"--{split}_npz", str(ROOT / job["sources"][split])))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    print(f"Starting {job['name']} on GPU {gpu}", flush=True)
    with (attempt / "launcher.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        while chunk := os.read(process.stdout.fileno(), 4096):
            log.write(chunk)
            log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"{job['name']} exited {code}; inspect {attempt / 'launcher.log'}")
    if complete(job, directory) is None:
        raise RuntimeError(f"{job['name']} did not produce a complete receipt and best.pth")
    print(f"Completed {job['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="Single physical GPU, default 0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or args.epochs < 2:
        parser.error("--gpu must be nonnegative and --epochs must be at least 2")
    plan = jobs(args.epochs)
    if args.dry_run:
        for job in plan:
            train = job["config"]["train"]
            print(job["name"], "lr_main=", train["lr_main"], "scheduler=", train["scheduler"],
                  "epochs=", train["epochs"], "val_epoch=", train["val_epoch"])
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary(plan)
        if args.summary_only:
            print(OUTPUT / "summary.csv")
            return
        for job in plan:
            if complete(job, OUTPUT / job["name"]):
                print(f"Skipping complete {job['name']}", flush=True)
                continue
            try:
                run(job, args.gpu)
            finally:
                summary(plan)
    print(OUTPUT / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
