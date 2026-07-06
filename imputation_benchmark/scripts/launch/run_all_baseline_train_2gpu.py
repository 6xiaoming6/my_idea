#!/usr/bin/env python3
"""Two-GPU paper-comparison launcher for every adapted mainline baseline.

Defaults reproduce the main experiment grid: 13 baselines x 3 datasets x
2 mask patterns x 4 missing rates = 312 full training jobs, using seed 42.
One job is assigned to each GPU at a time.  Job failures are recorded and do
not stop the remaining queue; rerun with --resume-run to continue.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


BENCH = Path(__file__).resolve().parents[2]
PROJECT = BENCH.parent
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
MASKS = ("fixed", "random")
RATES = (0.2, 0.4, 0.6, 0.8)
DEFAULT_POLICY = BENCH / "configs" / "policies" / "baseline_paper.json"
# Main-paper models implemented from st_imputation_baseline_survey.md.
# EXTRA_BASELINES.md models remain individually runnable but are deliberately
# absent from every default training matrix.
MODELS = (
    "MeanFill", "HistoricalAverage", "LATC", "BRITS", "GAIN", "CSDI",
    "SAITS", "GRIN", "PriSTI", "ImputeFormer", "STCPA", "STAMImputer", "PAST",
)
SCRIPTS = {
    "AGCRN": "train_agcrn.py", "ASTGNN": "train_astgnn.py",
    "BRITS": "train_brits.py", "CSDI": "train_csdi.py",
    "E2GAN": "train_e2gan.py", "GAIN": "train_gain.py",
    "GCASTN": "train_gcastn.py", "IGNNK": "train_ignnk.py",
    "ImputeFormer": "train_imputeformer.py", "LAST": "train_last.py",
    "LATC": "train_latc.py", "mTAN": "train_mtan.py",
    "PriSTI": "train_pristi.py", "SSTBAN": "train_sstban.py",
    "SAITS": "train_saits.py", "GRIN": "train_grin.py",
    "STCPA": "train_stcpa.py", "STAMImputer": "train_stamimputer.py",
    "PAST": "train_past.py", "MeanFill": "train_meanfill.py",
    "HistoricalAverage": "train_historical_average.py",
}


def rate_label(rate: float) -> str:
    return format(rate, "g")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=("0", "1"), metavar="GPU",
                        help="One or more GPU IDs; use one ID when launching fixed/random in separate terminals")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=tuple(SCRIPTS), default=list(MODELS))
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--channel", default="0", help="all or a zero-based channel index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=0, help="Per model-stage timeout; 0 disables it")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--policy-json", default=str(DEFAULT_POLICY),
                        help="Unified JSON policy controlling epochs, batches, patience, and validation intervals")
    parser.add_argument("--run-root", default=str(BENCH / "artifacts" / "runs" / "paper"))
    parser.add_argument("--resume-run", help="Existing launcher run directory to continue")
    parser.add_argument("--skip-prepare", action="store_true", help="Require and reuse existing adapted data/configs")
    parser.add_argument("--rebuild-data", action="store_true", help="Regenerate adapted data even when complete")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the job matrix without preparing or training")
    parser.add_argument("--quiet-console", action="store_true", help="Do not tee model progress to the console")
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Delete the best checkpoint after testing; default retains exactly one best checkpoint per job",
    )
    return parser.parse_args()


def job_id(job: dict[str, Any]) -> str:
    return "__".join((job["dataset"], job["model"], job["mask"],
                       f"rate{rate_label(job['rate'])}", f"channel{job['channel']}",
                       f"seed{job['seed']}"))


def data_complete(job: dict[str, Any]) -> bool:
    rate = rate_label(job["rate"])
    root = BENCH / "data" / "adapted" / job["dataset"] / f"{job['mask']}_{rate}" / f"channel_{job['channel']}"
    required = (
        root / f"true_data_{job['mask']}_{rate}_v2.npz",
        root / f"miss_data_{job['mask']}_{rate}_v2.npz",
        root / "split" / f"true_data_{job['mask']}_{rate}_v2.npz",
        root / "split" / f"miss_data_{job['mask']}_{rate}_v2.npz",
        root / "grid_edges.csv", root / "manifest.json",
    )
    return all(path.exists() for path in required)


def prepare_protocol(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    rate = rate_label(protocol["rate"])
    if args.rebuild_data or not data_complete(protocol):
        if args.skip_prepare:
            raise FileNotFoundError(f"Incomplete adapted data for {job_id({**protocol, 'model': 'DATA'})}")
        subprocess.run([
            args.python, str(BENCH / "scripts" / "data" / "prepare_grid_dataset.py"),
            "--dataset", protocol["dataset"], "--mask", protocol["mask"],
            "--rate", rate, "--channel", protocol["channel"], "--legacy-stream",
            "--output-root", str(BENCH / "data" / "adapted"),
        ], cwd=PROJECT, check=True)
    subprocess.run([
        args.python, str(BENCH / "scripts" / "config" / "generate_train_configs.py"),
        "--dataset", protocol["dataset"], "--mask", protocol["mask"],
        "--rate", rate, "--channel", protocol["channel"], "--seed", str(args.seed),
        "--policy-json", str(args.policy_path),
    ], cwd=PROJECT, check=True)


def write_summary(run_dir: Path, states: dict[str, dict[str, Any]]) -> None:
    rows = sorted(states.values(), key=lambda item: item["id"])
    lines = ["# Two-GPU baseline training summary", "",
             "| Dataset | Model | Mask | Rate | GPU | Status | MAE | RMSE | MAPE | Seconds | Log |",
             "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for item in rows:
        test = item.get("test", {})
        lines.append(
            f"| {item['dataset']} | {item['model']} | {item['mask']} | {item['rate']} | "
            f"{item.get('gpu', '')} | {item['status']} | {test.get('mae', '')} | "
            f"{test.get('rmse', '')} | {test.get('mape', '')} | {item.get('seconds', 0):.1f} | `{item.get('log', '')}` |"
        )
    passed = sum(item["status"] == "passed" for item in rows)
    lines += ["", f"Passed: **{passed}/{len(rows)}**", ""]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.policy_path = Path(args.policy_json).expanduser().resolve()
    if not args.policy_path.is_file():
        raise FileNotFoundError(f"Training policy not found: {args.policy_path}")
    policy = json.loads(args.policy_path.read_text(encoding="utf-8"))
    policy_name = policy.get("name", args.policy_path.stem)
    disabled_models = {
        model for model, settings in policy.get("models", {}).items()
        if settings.get("enabled") is False
    }
    if disabled_models:
        requested_disabled = disabled_models.intersection(args.models)
        if requested_disabled:
            print(
                "Policy-disabled models excluded: " + ", ".join(sorted(requested_disabled)),
                flush=True,
            )
        args.models = [model for model in args.models if model not in disabled_models]
    if not args.models:
        raise ValueError("No enabled models remain after applying the training policy")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus must not contain duplicate GPU IDs")
    if any(not 0 < rate < 1 for rate in args.rates):
        raise ValueError("Every --rates value must be between 0 and 1")
    protocols = [
        {"dataset": dataset, "mask": mask, "rate": rate, "channel": args.channel, "seed": args.seed}
        for dataset in args.datasets for mask in args.masks for rate in args.rates
    ]
    jobs = [{**protocol, "model": model} for protocol in protocols for model in args.models]
    print(f"Protocols: {len(protocols)}, models: {len(args.models)}, total jobs: {len(jobs)}")
    if args.dry_run:
        for job in jobs:
            print(job_id(job))
        return

    if args.resume_run:
        run_dir = Path(args.resume_run).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{policy_name}_{'-'.join(args.masks)}_gpu{'-'.join(args.gpus)}"
        run_dir = PROJECT / args.run_root / f"{stamp}_{label}"
        suffix = 1
        while run_dir.exists():
            run_dir = PROJECT / args.run_root / f"{stamp}_{label}_{suffix}"
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launcher_logs").mkdir(exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": args.datasets, "models": args.models, "masks": args.masks,
        "rates": args.rates, "channel": args.channel, "seed": args.seed,
        "gpus": list(args.gpus), "jobs": len(jobs),
        "checkpoint_policy": "none" if args.no_checkpoint else "best_only",
        "training_policy": {"name": policy_name, "path": str(args.policy_path), "content": policy},
        "protocol": {
            "split": "original train/val/test NPZ split",
            "selection": "each baseline's validation criterion; test evaluated once after selection",
            "max_epochs": max(int(item.get("epochs", 0)) for item in policy["models"].values()),
            "metrics": ["MAE", "RMSE", "MAPE (CHAP only recommended for reporting)"],
            "model_structure": "unchanged baseline source architecture",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for index, protocol in enumerate(protocols, 1):
        print(f"[prepare {index}/{len(protocols)}] {protocol['dataset']} {protocol['mask']} {protocol['rate']}", flush=True)
        prepare_protocol(args, {**protocol, "model": "DATA"})
    if args.prepare_only:
        print(f"Preparation complete: {run_dir}")
        return

    states: dict[str, dict[str, Any]] = {}
    old_summary = run_dir / "summary.json"
    if old_summary.exists():
        for item in json.loads(old_summary.read_text(encoding="utf-8")):
            states[item["id"]] = item
    pending = queue.Queue()
    for job in jobs:
        identifier = job_id(job)
        if states.get(identifier, {}).get("status") == "passed":
            continue
        pending.put(job)
    lock = threading.Lock()
    console_lock = threading.Lock()
    # Native baselines write checkpoints below model-specific trees. Serializing
    # jobs of the same model prevents one worker from pruning another's files.
    model_locks = {model: threading.Lock() for model in SCRIPTS}

    def worker(gpu: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            identifier = job_id(job)
            log = run_dir / "launcher_logs" / f"{identifier}.log"
            if log.exists():
                attempt = 2
                while (run_dir / "launcher_logs" / f"{identifier}.attempt{attempt}.log").exists():
                    attempt += 1
                log = run_dir / "launcher_logs" / f"{identifier}.attempt{attempt}.log"
            command = [
                args.python, str(BENCH / "scripts" / "train" / SCRIPTS[job["model"]]),
                "--dataset", job["dataset"], "--mask", job["mask"],
                "--rate", rate_label(job["rate"]), "--channel", job["channel"],
                "--gpu", gpu, "--seed", str(job["seed"]), "--python", args.python,
                "--timeout", str(args.timeout), "--output-root", args.output_root,
                "--policy-json", str(args.policy_path), "--no-prepare",
            ]
            if args.no_checkpoint:
                command.append("--no-checkpoint")
            with model_locks[job["model"]]:
                print(f"[{gpu}] START {identifier}", flush=True)
                start = time.monotonic()
                with log.open("w", encoding="utf-8") as stream:
                    stream.write("$ " + " ".join(command) + "\n")
                    stream.flush()
                    process = subprocess.Popen(
                        command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors="replace", bufsize=1,
                    )
                    assert process.stdout is not None
                    short = f"GPU{gpu} {job['dataset']}/{job['model']} {job['mask']}@{rate_label(job['rate'])}"
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        if not args.quiet_console:
                            with console_lock:
                                print(f"[{short}] {line}", end="", flush=True)
                    returncode = process.wait()
                    result = subprocess.CompletedProcess(command, returncode)
            model_result = None
            launcher_text = log.read_text(encoding="utf-8", errors="replace")
            match = re.findall(r"standardized baseline logs saved to (.+)", launcher_text)
            if match:
                result_path = Path(match[-1].strip()).parent / "result.json"
                if result_path.is_file():
                    model_result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.returncode == 0:
                job_status = "passed"
            elif result.returncode == 124:
                job_status = "timeout"
            elif result.returncode == 130:
                job_status = "interrupted"
            else:
                job_status = "failed"
            state = {**job, "id": identifier, "gpu": gpu,
                     "status": job_status,
                     "returncode": result.returncode, "seconds": time.monotonic() - start,
                     "log": str(log.relative_to(PROJECT)),
                     "result_json": str(result_path) if match else None,
                     "test": model_result.get("test", {}) if model_result else {}}
            with lock:
                states[identifier] = state
                write_summary(run_dir, states)
            print(f"[{gpu}] {state['status'].upper()} {identifier} ({state['seconds']:.1f}s)", flush=True)
            if result.returncode != 0:
                tail = "\n".join(launcher_text.splitlines()[-12:])
                with console_lock:
                    print(f"[{gpu}] FAILURE DETAIL {identifier}: returncode={result.returncode}\n{tail}", flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_summary(run_dir, states)
    failures = sum(item["status"] != "passed" for item in states.values())
    print(f"Summary: {run_dir / 'summary.md'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
