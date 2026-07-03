#!/usr/bin/env python3
"""Call every model-specific smoke script for TaxiBJ, BikeNYC and CHAP.

Run this from the project root with the Python environment used for the
benchmark.  Model commands and setup live in smoke_tests/run_*_smoke.py; this
file only prepares shared data, invokes those scripts, and aggregates logs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


BENCH = Path(__file__).resolve().parent
PROJECT = BENCH.parent
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
MODELS = (
    "AGCRN", "ASTGNN", "BRITS", "CSDI", "E2GAN", "GAIN", "GCASTN",
    "IGNNK", "ImputeFormer", "LAST", "LATC", "mTAN", "PriSTI", "SSTBAN",
)
MODEL_SCRIPTS = {
    "AGCRN": "run_agcrn_smoke.py",
    "ASTGNN": "run_astgnn_smoke.py",
    "BRITS": "run_brits_smoke.py",
    "CSDI": "run_csdi_smoke.py",
    "E2GAN": "run_e2gan_smoke.py",
    "GAIN": "run_gain_smoke.py",
    "GCASTN": "run_gcastn_smoke.py",
    "IGNNK": "run_ignnk_smoke.py",
    "ImputeFormer": "run_imputeformer_smoke.py",
    "LAST": "run_last_smoke.py",
    "LATC": "run_latc_smoke.py",
    "mTAN": "run_mtan_smoke.py",
    "PriSTI": "run_pristi_smoke.py",
    "SSTBAN": "run_sstban_smoke.py",
}


@dataclass
class Result:
    dataset: str
    model: str
    status: str
    returncode: int | None
    seconds: float
    log: str
    command: list[str]
    tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable for every baseline.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--windows", type=int, default=8, help="Original windows retained per split.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per process timeout in seconds.")
    parser.add_argument("--logs", default="imputation_benchmark/smoke_runs")
    parser.add_argument("--no-prepare", action="store_true", help="Reuse existing smoke data/configs.")
    return parser.parse_args()


def command_for(args: argparse.Namespace, model: str, dataset: str) -> list[str]:
    script = BENCH / "smoke_tests" / MODEL_SCRIPTS[model]
    return [
        args.python, str(script), "--dataset", dataset, "--python", args.python,
        "--gpu", args.gpu, "--timeout", str(args.timeout), "--windows",
        str(args.windows), "--no-prepare",
    ]


def prepare(args: argparse.Namespace) -> None:
    if args.windows < 1:
        raise ValueError("--windows must be positive")
    subprocess.run([args.python, str(BENCH / "generate_smoke_configs.py")], cwd=PROJECT, check=True)
    for dataset in args.datasets:
        subprocess.run([
            args.python, str(BENCH / "prepare_grid_dataset.py"), "--dataset", dataset,
            "--mask", "fixed", "--rate", "0.2", "--channel", "0", "--legacy-stream",
            "--max-windows-per-split", str(args.windows), "--output-root",
            str(BENCH / "data/smoke"),
        ], cwd=PROJECT, check=True)


def run_one(args: argparse.Namespace, dataset: str, model: str, log: Path) -> Result:
    command = command_for(args, model, dataset)
    start = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {' '.join(command)}\n")
        stream.flush()
        proc = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
    code = proc.returncode
    status = "passed" if code == 0 else "timeout" if code == 124 else "failed"
    text = log.read_text(encoding="utf-8", errors="replace")
    return Result(dataset, model, status, code, time.monotonic() - start,
                  str(log.relative_to(PROJECT)), command, "\n".join(text.splitlines()[-30:]))


def write_summary(root: Path, results: list[Result]) -> None:
    (root / "summary.json").write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n")
    lines = ["# Baseline smoke summary", "", "| Dataset | Model | Status | Exit | Seconds | Log |", "|---|---|---:|---:|---:|---|"]
    for r in results:
        lines.append(f"| {r.dataset} | {r.model} | {r.status} | {r.returncode} | {r.seconds:.1f} | `{r.log}` |")
    passed = sum(r.status == "passed" for r in results)
    lines += ["", f"Passed: **{passed}/{len(results)}**", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.no_prepare:
        prepare(args)
    run_root = PROJECT / args.logs / datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[Result] = []
    total = len(args.datasets) * len(args.models)
    for index, (dataset, model) in enumerate(((d, m) for d in args.datasets for m in args.models), 1):
        print(f"[{index}/{total}] {dataset} / {model}", flush=True)
        result = run_one(args, dataset, model, run_root / f"{dataset}_{model}.log")
        results.append(result)
        write_summary(run_root, results)
        print(f"  {result.status}, exit={result.returncode}, {result.seconds:.1f}s", flush=True)
    print(f"Summary: {run_root / 'summary.md'}")
    if any(r.status != "passed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
