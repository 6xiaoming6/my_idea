#!/usr/bin/env python3
"""Rebuild concise human-readable baseline logs from preserved raw output."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from baseline_logger import _human_time, _records


MAE_SELECTED = {"CSDI", "E2GAN", "GAIN", "IGNNK", "PriSTI"}


def fmt(value: float | None, width: int, precision: int) -> str:
    return f"{value:>{width}.{precision}f}" if value is not None else f"{'n/a':>{width}}"


def seed_from(result: dict, run_dir: Path) -> str:
    if result.get("seed") is not None:
        return str(result["seed"])
    match = re.search(r"_seed([^_]+)", run_dir.name)
    return match.group(1) if match else "n/a"


def compact(run_dir: Path, dry_run: bool = False) -> bool:
    result_path = run_dir / "result.json"
    raw_path = run_dir / "logs" / "raw.log"
    if not result_path.is_file() or not raw_path.is_file():
        return False

    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = str(result.get("variant") or run_dir.parents[2].name)
    records, parsed_test = _records(model, raw_path.read_text(encoding="utf-8", errors="replace"))
    stored_test = result.get("test", {})
    test = {
        name: parsed_test.get(name) if parsed_test.get(name) is not None else stored_test.get(name)
        for name in ("mae", "rmse", "mape", "time_sec")
    }

    best_epoch = None
    best_metric = None
    best_metric_name = None
    mae_selected = model in MAE_SELECTED
    for item in records:
        va = item["val"]
        metric = va["mae"] if mae_selected else va["loss"]
        if metric is None and mae_selected:
            metric = va["loss"]
        if item["is_best"] and metric is not None:
            best_epoch = item["epoch"]
            best_metric = metric
            best_metric_name = "val_mae" if mae_selected and va["mae"] is not None else "val_loss"

    dataset = str(result.get("dataset", "n/a"))
    mask = str(result.get("mask_pattern", "n/a"))
    rate = result.get("mask_rate_config", "n/a")
    channel = result.get("channel", "n/a")
    seed = seed_from(result, run_dir)
    batch = result.get("batch_size", "n/a")
    device = result.get("device", "n/a")
    status = str(result.get("status", "unknown"))
    returncode = result.get("returncode", "n/a")
    total_time = float(result.get("total_time_sec") or 0.0)
    config_path = result.get("config_path") or "n/a"
    identity = (f"dataset={dataset}  model={model}  mask={mask}  rate={rate}  "
                f"channel={channel}  seed={seed}")

    t_header = (f"{'epoch':>6}  {'train_loss':>12}  {'val_loss':>11}  {'val_mae':>10}  "
                f"{'val_rmse':>10}  {'best':>5}")
    train = [f"Baseline training | {identity}  batch={batch}  device={device}",
             f"config: {config_path}", "-" * len(t_header), t_header, "-" * len(t_header)]
    v_header = (f"{'epoch':>6}  {'val_loss':>11}  {'mae':>10}  {'rmse':>10}  "
                f"{'mape':>10}  {'best':>5}")
    val = [f"Baseline validation | {identity}",
           "metrics: original scale, artificially missing entries only",
           "-" * len(v_header), v_header, "-" * len(v_header)]
    metric_lines = []
    for item in records:
        epoch, tr, va = item["epoch"], item["train"], item["val"]
        best = "*" if item["is_best"] else ""
        train.append(
            f"{epoch:>6}  {fmt(tr['loss'],12,5)}  {fmt(va['loss'],11,5)}  "
            f"{fmt(va['mae'],10,4)}  {fmt(va['rmse'],10,4)}  {best:>5}"
        )
        if any(va[name] is not None for name in ("loss", "mae", "rmse", "mape")):
            val.append(
                f"{epoch:>6}  {fmt(va['loss'],11,5)}  {fmt(va['mae'],10,4)}  "
                f"{fmt(va['rmse'],10,4)}  {fmt(va['mape'],10,4)}  {best:>5}"
            )
        metric_lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    summary = (f"best: epoch={best_epoch}  {best_metric_name}={best_metric:.6f}"
               if best_epoch is not None else "best: n/a")
    completed_epochs = max((item["epoch"] for item in records), default=0)
    train += ["-" * len(t_header), summary,
              f"epochs={completed_epochs}  time={_human_time(total_time)}  status={status}  returncode={returncode}"]
    val += ["-" * len(v_header), summary]
    test_log = [f"Baseline test | {identity}",
                "metrics: original scale, artificially missing entries only",
                (f"checkpoint: epoch={best_epoch if best_epoch is not None else 'n/a'}  "
                 f"criterion={best_metric_name or 'n/a'}  "
                 f"value={best_metric if best_metric is not None else 'n/a'}"),
                (f"MAE={test.get('mae') if test.get('mae') is not None else 'n/a'}  "
                 f"RMSE={test.get('rmse') if test.get('rmse') is not None else 'n/a'}  "
                 f"MAPE={test.get('mape') if test.get('mape') is not None else 'n/a'}"),
                f"time={_human_time(total_time)}  status={status}  returncode={returncode}"]

    if dry_run:
        print(run_dir)
        return True
    log_dir = run_dir / "logs"
    (log_dir / "train.log").write_text("\n".join(train) + "\n", encoding="utf-8")
    (log_dir / "val.log").write_text("\n".join(val) + "\n", encoding="utf-8")
    (log_dir / "test.log").write_text("\n".join(test_log) + "\n", encoding="utf-8")
    (log_dir / "metrics.jsonl").write_text("\n".join(metric_lines) + ("\n" if metric_lines else ""), encoding="utf-8")
    result.update({
        "seed": seed,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
        "completed_epochs": completed_epochs,
        "test": test,
    })
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).expanduser().resolve()
    runs = sorted({path.parent for path in root.glob("*/baseline/*/*/rate*/*/result.json")})
    count = sum(compact(run, args.dry_run) for run in runs)
    print(f"Compacted {count} baseline run(s) under {root}")


if __name__ == "__main__":
    main()
