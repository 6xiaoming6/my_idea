#!/usr/bin/env python3
"""Run one train epoch, one validation pass, and one test pass per overview baseline.

Only methods marked implemented in ``baselines/st_imputation_baseline_survey.md``
are included. Appendix-only methods from ``EXTRA_BASELINES.md`` cannot be
selected. Jobs run sequentially on one GPU to keep power and memory bounded.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


BENCH = Path(__file__).resolve().parents[2]
PROJECT = BENCH.parent
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
MODELS = (
    "MeanFill", "HistoricalAverage", "LATC", "BRITS", "GAIN", "SAITS", "GRIN",
    "ImputeFormer", "STCPA", "STAMImputer", "PAST", "PriSTI", "CSDI",
)
NEURAL_MODELS = set(MODELS) - {"MeanFill", "HistoricalAverage", "LATC"}
SCRIPTS = {
    "MeanFill": "train_meanfill.py", "HistoricalAverage": "train_historical_average.py",
    "LATC": "train_latc.py", "BRITS": "train_brits.py", "GAIN": "train_gain.py",
    "SAITS": "train_saits.py", "GRIN": "train_grin.py",
    "ImputeFormer": "train_imputeformer.py", "STCPA": "train_stcpa.py",
    "STAMImputer": "train_stamimputer.py", "PAST": "train_past.py",
    "PriSTI": "train_pristi.py", "CSDI": "train_csdi.py",
}
DEFAULT_POLICY = BENCH / "configs" / "policies" / "overview_1epoch_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--mask", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--windows", type=int, default=4,
                        help="Source windows retained in each train/val/test split")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per model; 0 disables it")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--policy-json", default=str(DEFAULT_POLICY))
    parser.add_argument("--data-root", default=str(BENCH / "data" / "testing" / "overview_1epoch"))
    parser.add_argument("--config-root", default=str(BENCH / "configs" / "testing" / "overview_1epoch"))
    parser.add_argument("--output-root", default="outputs/overview_1epoch_test")
    parser.add_argument("--run-root", default=str(BENCH / "artifacts" / "runs" / "overview_1epoch"))
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--quiet-console", action="store_true")
    return parser.parse_args()


def rate_label(rate: float) -> str:
    return format(rate, "g")


def prepare(args: argparse.Namespace, dataset: str) -> None:
    if args.skip_prepare:
        return
    subprocess.run([
        args.python, str(BENCH / "scripts" / "data" / "prepare_grid_dataset.py"), "--dataset", dataset,
        "--mask", args.mask, "--rate", rate_label(args.rate), "--channel", args.channel,
        "--legacy-stream", "--max-windows-per-split", str(args.windows),
        "--output-root", str(Path(args.data_root).resolve()),
    ], cwd=PROJECT, check=True)
    subprocess.run([
        args.python, str(BENCH / "scripts" / "config" / "generate_train_configs.py"), "--dataset", dataset,
        "--mask", args.mask, "--rate", rate_label(args.rate), "--channel", args.channel,
        "--policy-json", str(Path(args.policy_json).resolve()),
        "--adapted-root", str(Path(args.data_root).resolve()),
        "--output-root", str(Path(args.config_root).resolve()),
    ], cwd=PROJECT, check=True)


def validate_result(model: str, result: dict) -> tuple[bool, str]:
    test = result.get("test", {})
    if result.get("status") != "finished" or result.get("returncode") != 0:
        return False, f"runner status={result.get('status')} returncode={result.get('returncode')}"
    if test.get("mae") is None or test.get("rmse") is None:
        return False, "test MAE/RMSE missing"
    completed = int(result.get("completed_epochs", 0))
    best_epoch = result.get("best_epoch")
    if model in NEURAL_MODELS and (completed != 1 or best_epoch != 1):
        return False, f"expected train_epoch=1 and val_epoch=1, got completed={completed}, best={best_epoch}"
    if model == "LATC" and completed != 1:
        return False, f"expected one LATC iteration, got {completed}"
    if model in {"MeanFill", "HistoricalAverage"} and best_epoch != 0:
        return False, f"expected deterministic validation pass at epoch 0, got {best_epoch}"
    return True, "train/validation/test contract passed"


def write_summary(run_dir: Path, results: list[dict]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Overview baseline one-epoch test", "",
        "| Dataset | Model | Status | Train epoch | Val epoch | Test MAE | Test RMSE | Seconds | Log |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['dataset']} | {item['model']} | {item['status']} | {item.get('completed_epochs', '')} | "
            f"{item.get('best_epoch', '')} | {item.get('test', {}).get('mae', '')} | "
            f"{item.get('test', {}).get('rmse', '')} | {item['seconds']:.1f} | `{item['launcher_log']}` |"
        )
    passed = sum(item["status"] == "passed" for item in results)
    lines += ["", f"Passed: **{passed}/{len(results)}**", ""]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.windows < 1:
        raise ValueError("--windows must be positive")
    if not 0 < args.rate < 1:
        raise ValueError("--rate must be between 0 and 1")
    policy = json.loads(Path(args.policy_json).read_text(encoding="utf-8"))
    active = {name for name, cfg in policy["models"].items() if cfg.get("enabled", True)}
    if set(args.models) - active:
        raise ValueError(f"Selected models are disabled by one-epoch policy: {sorted(set(args.models) - active)}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT / args.run_root / f"{stamp}_{args.mask}_rate{rate_label(args.rate)}_gpu{args.gpu}"
    (run_dir / "launcher_logs").mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "gpu": args.gpu,
        "datasets": args.datasets, "models": args.models, "mask": args.mask, "rate": args.rate,
        "windows_per_split": args.windows, "policy": str(Path(args.policy_json).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "config_root": str(Path(args.config_root).resolve()),
        "contract": "one train epoch -> one validation pass -> load best -> one test pass",
        "checkpoint_policy": "best checkpoint used for test, then deleted",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for dataset in args.datasets:
        print(f"[prepare] {dataset} {args.mask}@{rate_label(args.rate)} windows={args.windows}", flush=True)
        prepare(args, dataset)

    results: list[dict] = []
    jobs = [(dataset, model) for dataset in args.datasets for model in args.models]
    for index, (dataset, model) in enumerate(jobs, 1):
        identifier = f"{dataset}__{model}__{args.mask}__rate{rate_label(args.rate)}"
        launcher_log = run_dir / "launcher_logs" / f"{identifier}.log"
        command = [
            args.python, str(BENCH / "scripts" / "train" / SCRIPTS[model]),
            "--dataset", dataset, "--mask", args.mask, "--rate", rate_label(args.rate),
            "--channel", args.channel, "--gpu", args.gpu, "--python", args.python,
            "--timeout", str(args.timeout), "--policy-json", str(Path(args.policy_json).resolve()),
            "--data-root", str(Path(args.data_root).resolve()), "--output-root", args.output_root,
            "--config-root", str(Path(args.config_root).resolve()),
            "--no-prepare", "--no-checkpoint",
        ]
        print(f"[{index}/{len(jobs)}] START {dataset}/{model}", flush=True)
        start = time.monotonic()
        captured = []
        with launcher_log.open("w", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace", bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line); stream.flush(); captured.append(line)
                if not args.quiet_console:
                    print(f"[{dataset}/{model}] {line}", end="", flush=True)
            returncode = process.wait()
        elapsed = time.monotonic() - start
        output = "".join(captured)
        result_match = re.findall(r"standardized baseline logs saved to (.+)", output)
        model_result = None
        if result_match:
            candidate = Path(result_match[-1].strip()).parent / "result.json"
            if candidate.is_file():
                model_result = json.loads(candidate.read_text(encoding="utf-8"))
        if returncode == 0 and model_result is not None:
            passed, detail = validate_result(model, model_result)
        else:
            passed, detail = False, f"process returncode={returncode}; result.json={'found' if model_result else 'missing'}"
        item = {
            "dataset": dataset, "model": model, "status": "passed" if passed else "failed",
            "detail": detail, "process_returncode": returncode, "seconds": elapsed,
            "launcher_log": str(launcher_log.relative_to(PROJECT)),
            "completed_epochs": model_result.get("completed_epochs") if model_result else None,
            "best_epoch": model_result.get("best_epoch") if model_result else None,
            "test": model_result.get("test", {}) if model_result else {},
            "standardized_logs": model_result.get("logs") if model_result else None,
        }
        results.append(item); write_summary(run_dir, results)
        print(f"[{index}/{len(jobs)}] {item['status'].upper()} {dataset}/{model} ({elapsed:.1f}s): {detail}", flush=True)
    print(f"Summary: {run_dir / 'summary.md'}", flush=True)
    if any(item["status"] != "passed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
