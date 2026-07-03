#!/usr/bin/env python3
"""Convert heterogeneous baseline stdout into the project's log layout."""
from __future__ import annotations

import configparser
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean(text: str) -> str:
    return ANSI.sub("", text.replace("\r", "\n"))


def _last(pattern: str, text: str, flags: int = re.I) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def _blank(epoch: int) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train": {"loss": None, "mae": None, "rmse": None, "mape": None, "lr": None},
        "val": {"loss": None, "mae": None, "rmse": None, "mape": None},
        "perf": {"train_time_sec": None, "val_time_sec": None, "epoch_time_sec": None,
                 "peak_memory_gb": None},
        "is_best": False,
    }


def _records(model: str, raw: str) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    text = _clean(raw)
    records: dict[int, dict[str, Any]] = {}

    def record(epoch: int) -> dict[str, Any]:
        return records.setdefault(epoch, _blank(epoch))

    if model == "AGCRN":
        for match in re.finditer(rf"Train Epoch (\d+): averaged Loss: ({NUMBER})", text, re.I):
            record(int(match[1]))["train"]["loss"] = _number(match[2])
        for match in re.finditer(rf"Val Epoch (\d+): average Loss: ({NUMBER})", text, re.I):
            record(int(match[1]))["val"]["loss"] = _number(match[2])

    elif model in {"ASTGNN", "GCASTN"}:
        for match in re.finditer(rf"epoch:\s*(\d+), train time every whole data:({NUMBER})s", text, re.I):
            item = record(int(match[1]) + 1)
            item["perf"]["train_time_sec"] = _number(match[2])
        for match in re.finditer(rf"epoch:\s*(\d+), total time:({NUMBER})s", text, re.I):
            item = record(int(match[1]) + 1)
            item["perf"]["epoch_time_sec"] = _number(match[2])
        # These models print validation batches but not an epoch aggregate.
        losses = [_number(m[1]) for m in re.finditer(rf"validation batch 1 / \d+, loss: ({NUMBER})", text, re.I)]
        for item, loss in zip(records.values(), losses):
            item["val"]["loss"] = loss

    elif model == "BRITS":
        for match in re.finditer(r"epoch\s+(\d+)\s+train spend\s+([\d.]+)\s+seconds", text, re.I):
            item = record(int(match[1]) + 1)
            item["perf"]["train_time_sec"] = _number(match[2])
        val_losses = [_number(m[1]) for m in re.finditer(rf"Validation loss decrease from {NUMBER} (?:to|ot) ({NUMBER})", text, re.I)]
        for item, loss in zip(records.values(), val_losses):
            item["val"]["loss"] = loss

    elif model == "CSDI":
        grouped: dict[int, list[float]] = {}
        for match in re.finditer(rf"Train Epoch:\s*(\d+)\s+Batch:\s*\d+\s+Loss:\s*({NUMBER})", text, re.I):
            grouped.setdefault(int(match[1]) + 1, []).append(float(match[2]))
        for epoch, values in grouped.items():
            record(epoch)["train"]["loss"] = sum(values) / len(values)

    elif model in {"E2GAN", "GAIN"}:
        for match in re.finditer(rf"epoch(\d+).*?(?:^|\n)train\s+Generator_losss:\s*({NUMBER}).*?val\s+mae:\s*({NUMBER})\s+rmse:\s*({NUMBER})\s+mape:\s*({NUMBER})", text, re.I | re.S | re.M):
            item = record(int(match[1]) + 1)
            item["train"]["loss"] = _number(match[2])
            item["val"]["mae"] = _number(match[3])
            item["val"]["rmse"] = _number(match[4])
            item["val"]["mape"] = _number(match[5])

    elif model == "IGNNK":
        for match in re.finditer(rf"epoch(\d+) train loss:({NUMBER}).*?epoch\1 val\s+mae:({NUMBER})\s+RMSE:({NUMBER})\s+MAPE:({NUMBER})", text, re.I | re.S):
            item = record(int(match[1]) + 1)
            item["train"]["loss"] = _number(match[2])
            item["val"]["mae"] = _number(match[3])
            item["val"]["rmse"] = _number(match[4])
            item["val"]["mape"] = _number(match[5])

    elif model == "ImputeFormer":
        for match in re.finditer(rf"Epoch\s+(\d+)\s+- training loss \([^)]*\):\s*({NUMBER}), validation MSE:\s*({NUMBER})", text, re.I):
            item = record(int(match[1]))
            item["train"]["loss"] = _number(match[2])
            item["val"]["loss"] = _number(match[3])
            item["val"]["rmse"] = math.sqrt(float(match[3]))

    elif model == "LATC":
        for match in re.finditer(rf"Iter:\s*(\d+).*?Imputation MAE:\s*({NUMBER}).*?Imputation RMSE:\s*({NUMBER})", text, re.I | re.S):
            item = record(int(match[1]))
            item["val"]["mae"] = _number(match[2])
            item["val"]["rmse"] = _number(match[3])

    elif model == "mTAN":
        for match in re.finditer(rf"Iter:\s*(\d+),Train avg elbo:\s*({NUMBER}).*?Iter:\s*\1,Val loss:\s*({NUMBER})", text, re.I | re.S):
            item = record(int(match[1]) + 1)
            item["train"]["loss"] = _number(match[2])
            item["val"]["loss"] = _number(match[3])

    elif model == "PriSTI":
        for match in re.finditer(rf"avg_epoch_loss=({NUMBER}), epoch=(\d+)", text, re.I):
            record(int(match[2]) + 1)["train"]["loss"] = _number(match[1])

    elif model == "SSTBAN":
        for match in re.finditer(rf"epoch:\s*(\d+)/(\d+), training time:\s*({NUMBER})s, validation time:\s*({NUMBER})s\s*\ntrain loss:\s*({NUMBER}), val_loss:\s*({NUMBER})", text, re.I):
            item = record(int(match[1]))
            item["perf"]["train_time_sec"] = _number(match[3])
            item["perf"]["val_time_sec"] = _number(match[4])
            item["perf"]["epoch_time_sec"] = float(match[3]) + float(match[4])
            item["train"]["loss"] = _number(match[5])
            item["val"]["loss"] = _number(match[6])

    # Data-adaptation wrappers print one common aggregate line whenever a
    # baseline completes validation. This fills val.log without fabricating
    # metrics for epochs where validation was intentionally skipped.
    for match in re.finditer(rf"Validation Epoch\s+(\d+):\s*average Loss:\s*({NUMBER})", text, re.I):
        record(int(match[1]))["val"]["loss"] = _number(match[2])

    # Every adapted baseline uses this exact line for metrics calculated only
    # at artificially hidden validation entries after inverse scaling.  Keep
    # this model-independent so newly added baselines can join the same log
    # contract without adding another parser branch.
    for match in re.finditer(
        rf"Validation Metrics Epoch\s+(\d+):\s*MAE:\s*({NUMBER})\s+"
        rf"RMSE:\s*({NUMBER})\s+MAPE:\s*({NUMBER})",
        text,
        re.I,
    ):
        item = record(int(match[1]))
        item["val"].update({
            "mae": _number(match[2]),
            "rmse": _number(match[3]),
            "mape": _number(match[4]),
        })

    # Validation metric lines contain the same MAE/RMSE/MAPE token sequence as
    # several native test reports. Remove them before test parsing so a job
    # that crashes before testing cannot accidentally report validation values
    # as its final test result.
    test_text = re.sub(r"^.*Validation Metrics Epoch.*$", "", text, flags=re.I | re.M)

    # LAST is non-iterative, so it intentionally has no fabricated epoch row.
    test: dict[str, float | None] = {"mae": None, "rmse": None, "mape": None, "time_sec": None}
    patterns = [
        rf"test\s+mae:\s*({NUMBER})\s+rmse:\s*({NUMBER})\s+mape:\s*({NUMBER})",
        rf"TEST\s+MAE:\s*({NUMBER}),\s*RMSE:\s*({NUMBER}),\s*MAPE:\s*({NUMBER})",
        rf"Test:\s*MAE:({NUMBER})\s+RMSE:({NUMBER})\s+MAPE:({NUMBER})",
        rf"all MAE:\s*({NUMBER}).*?all RMSE:\s*({NUMBER}).*?all MAPE:\s*({NUMBER})",
        rf"MAE:({NUMBER})\s+RMSE:({NUMBER})\s+MAPE:({NUMBER})",
        rf"RMSE:\s*({NUMBER})\s*\nMAE:\s*({NUMBER})\s*\nMAPE:\s*({NUMBER})",
        rf"Imputation MAE:\s*({NUMBER}).*?Imputation RMSE:\s*({NUMBER}).*?Imputation MAPE:\s*({NUMBER})",
    ]
    for index, pattern in enumerate(patterns):
        match = _last(pattern, test_text, re.I | re.S)
        if match:
            if index == 5:
                test.update({"rmse": _number(match[1]), "mae": _number(match[2]), "mape": _number(match[3])})
            else:
                test.update({"mae": _number(match[1]), "rmse": _number(match[2]), "mape": _number(match[3])})
            break
    if model == "ImputeFormer":
        match = _last(rf"^\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*$", test_text, re.M)
        if match:
            test.update({"mae": _number(match[1]), "rmse": _number(match[2]), "mape": _number(match[3])})
    if model == "AGCRN":
        match = _last(rf"Average Horizon, MAE:\s*({NUMBER}), RMSE:\s*({NUMBER}), MAPE:\s*({NUMBER})%", test_text)
        if match:
            test.update({"mae": _number(match[1]), "rmse": _number(match[2]), "mape": _number(match[3])})
    if model == "IGNNK":
        match = _last(rf"best epoch:\s*\d+, best_MAE:\s*({NUMBER}), best_RMSE:\s*({NUMBER}), best_MAPE:\s*({NUMBER})", test_text)
        if match:
            test.update({"mae": _number(match[1]), "rmse": _number(match[2]), "mape": _number(match[3])})
    if model == "LAST":
        block = _last(rf"Completing the test set together.*?MAE:\s*({NUMBER}).*?RMSE:\s*({NUMBER}).*?MAPE:\s*({NUMBER})", test_text, re.I | re.S)
        if block:
            test.update({"mae": _number(block[1]), "rmse": _number(block[2]), "mape": _number(block[3])})
    if model == "SSTBAN":
        match = _last(rf"^test\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})%", test_text, re.I | re.M)
        if match:
            test.update({"mae": _number(match[2]), "rmse": _number(match[3]), "mape": _number(match[4])})

    ordered = [records[key] for key in sorted(records)]
    best_value = float("inf")
    for item in ordered:
        candidate = item["val"]["mae"]
        if candidate is None:
            candidate = item["val"]["loss"]
        if candidate is not None and candidate < best_value:
            item["is_best"] = True
            best_value = candidate
    return ordered, test


def _load_config(commands: list[list[str]]) -> tuple[str | None, dict[str, Any]]:
    # Multi-stage models such as BRITS put the actual training config last.
    for command in reversed(commands):
        if "--config" not in command:
            continue
        path = Path(command[command.index("--config") + 1]).resolve()
        if not path.is_file():
            continue
        if path.suffix in {".yaml", ".yml"}:
            return str(path), yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = configparser.ConfigParser()
        cfg.read(path)
        return str(path), {section: dict(cfg[section]) for section in cfg.sections()}
    return None, {}


def _find_key(data: Any, names: set[str]) -> Any:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower().replace("-", "_") in names:
                return value
        for value in data.values():
            found = _find_key(value, names)
            if found is not None:
                return found
    return None


def _human_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _command_output(command: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=3, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip().splitlines()
    return value[0] if result.returncode == 0 and value else "unknown"


class BaselineLogAdapter:
    def __init__(self, project: Path, model: str, args: Any, commands: list[list[str]]) -> None:
        self.project = project
        self.model = model
        self.args = args
        self.config_path, self.config = _load_config(commands)
        batch = _find_key(self.config, {"batch_size"})
        if batch is None:
            for command in reversed(commands):
                if "--batch_size" in command:
                    batch = command[command.index("--batch_size") + 1]
                    break
        seed = getattr(args, "seed", None) or _find_key(self.config, {"seed"})
        batch = str(batch if batch is not None else "na")
        self.batch = batch
        seed = str(seed if seed not in {None, ""} else 42)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(args.output_root)
        if not root.is_absolute():
            root = project / root
        base = root / args.dataset / "baseline" / model / args.mask / f"rate{format(args.rate, 'g')}"
        run_id = f"{stamp}_seed{seed}_bs{batch}"
        self.run_dir = base / run_id
        suffix = 1
        while self.run_dir.exists():
            self.run_dir = base / f"{run_id}_{suffix}"
            suffix += 1
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.raw = (self.log_dir / "raw.log").open("w", encoding="utf-8", buffering=1)
        self.started = datetime.now()
        self.commands = commands

    def write_raw(self, text: str) -> None:
        self.raw.write(text)

    @staticmethod
    def _fmt(value: float | None, width: int, precision: int) -> str:
        return f"{value:>{width}.{precision}f}" if value is not None else f"{'n/a':>{width}}"

    def finalize(self, status: str, returncode: int, total_time: float) -> None:
        self.raw.flush()
        self.raw.close()
        raw_text = (self.log_dir / "raw.log").read_text(encoding="utf-8", errors="replace")
        records, test = _records(self.model, raw_text)
        configured_lr = _number(_find_key(self.config, {"learning_rate", "lr"}))
        for item in records:
            if item["train"].get("lr") is None:
                item["train"]["lr"] = configured_lr
        train_path, val_path = self.log_dir / "train.log", self.log_dir / "val.log"
        test_path = self.log_dir / "test.log"
        metrics_path = self.log_dir / "metrics.jsonl"
        started = self.started.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path = (self.project / "imputation_benchmark" / "data" / "adapted" /
                         self.args.dataset / f"{self.args.mask}_{format(self.args.rate, 'g')}" /
                         f"channel_{self.args.channel}" / "manifest.json")
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = manifest.get("source_samples", {})
        source_paths = manifest.get("source_npz", {})
        source_shape = manifest.get("source_window_shape", [])
        selected_channels = manifest.get("selected_channels", [])
        channels = len(selected_channels) if selected_channels else 1
        sample_shape = None
        if len(source_shape) == 4:
            sample_shape = (channels, *source_shape[1:])
        actual_missing = manifest.get("actual_missing_rate")
        batch_number = int(self.batch) if self.batch.isdigit() else None

        def steps(split: str) -> int | str:
            count = samples.get(split)
            if not isinstance(count, int) or not batch_number:
                return "n/a"
            return math.ceil(count / batch_number)

        def mask_description(split: str) -> str:
            count = samples.get(split)
            if not isinstance(count, int) or sample_shape is None or actual_missing is None:
                return "n/a"
            return (f"shape={(count, *sample_shape)}, observed_rate={1 - float(actual_missing):.4f}, "
                    f"missing_rate={float(actual_missing):.4f}")

        device_name = _command_output(
            ["nvidia-smi", f"--id={self.args.gpu}", "--query-gpu=name", "--format=csv,noheader"],
            self.project,
        )
        metadata = {
            "run_dir": str(self.run_dir),
            "command": " | ".join(" ".join(c) for c in self.commands),
            "git_commit": _command_output(["git", "rev-parse", "--short", "HEAD"], self.project),
            "python": sys.version.split()[0],
            "train_npz": source_paths.get("train", "n/a"),
            "val_npz": source_paths.get("val", "n/a"),
            "dataset": self.args.dataset,
            "experiment_type": "baseline",
            "variant": self.model,
            "train_samples": samples.get("train", "n/a"),
            "val_samples": samples.get("val", "n/a"),
            "train_steps_per_epoch": steps("train"),
            "val_steps_per_epoch": steps("val"),
            "batch_size": self.batch,
            "mask_pattern": self.args.mask,
            "mask_rate_config": self.args.rate,
            "metric_scale": "original data range (after inverse transform)",
            "metric_scope": "artificially missing entries only",
            "train_mask": mask_description("train"),
            "val_mask": mask_description("val"),
            "data_shape": (samples.get("train"), *sample_shape) if sample_shape and samples.get("train") else "n/a",
            "channel": self.args.channel,
            "device": f"cuda:{self.args.gpu}",
            "device_name": device_name,
            "total_params": "n/a (original baseline does not report it)",
            "trainable_params": "n/a (original baseline does not report it)",
            "config_path": self.config_path,
            "best_checkpoint": getattr(self.args, "best_checkpoint", None) or "n/a",
        }
        t_header = (f"{'epoch':>6}  {'loss':>11}  {'mae':>10}  {'rmse':>11}  {'val_mae':>10}  "
                    f"{'lr':>10}  {'train_s':>8}  {'val_s':>7}  {'epoch_s':>8}  {'mem_gb':>7}  {'best':>5}")
        v_header = (f"{'epoch':>6}  {'loss':>11}  {'mae':>10}  {'rmse':>11}  {'mape':>10}  "
                    f"{'train_mae':>10}  {'epoch_s':>8}  {'best':>5}")
        with train_path.open("w", encoding="utf-8") as train, val_path.open("w", encoding="utf-8") as val, metrics_path.open("w", encoding="utf-8") as metrics:
            train.write(f"Training started: {started}\n" + "-" * 96 + "\nRun\n")
            for key, value in metadata.items():
                train.write(f"  {key}: {value}\n")
            train.write("Config\n" + json.dumps(self.config, indent=2, ensure_ascii=False) + "\n" + "-" * 96 + "\n")
            train.write("-" * len(t_header) + "\n" + t_header + "\n" + "-" * len(t_header) + "\n")
            val.write("Metric scale: original data range (after inverse transform)\n")
            val.write("Metric scope: artificially missing entries only\n")
            val.write("-" * len(v_header) + "\n" + v_header + "\n" + "-" * len(v_header) + "\n")
            best_epoch, best_metric, best_metric_name = None, float("inf"), "val_metric"
            for item in records:
                epoch, tr, va, perf = item["epoch"], item["train"], item["val"], item["perf"]
                best = "*" if item["is_best"] else ""
                lr_text = f"{tr['lr']:>10.2e}" if tr["lr"] is not None else f"{'n/a':>10}"
                train.write(
                    f"{epoch:>6}  {self._fmt(tr['loss'],11,5)}  {self._fmt(tr['mae'],10,4)}  "
                    f"{self._fmt(tr['rmse'],11,4)}  {self._fmt(va['mae'],10,4)}  {lr_text}  "
                    f"{self._fmt(perf['train_time_sec'],8,1)}  {self._fmt(perf['val_time_sec'],7,1)}  "
                    f"{self._fmt(perf['epoch_time_sec'],8,1)}  {self._fmt(perf['peak_memory_gb'],7,2)}  {best:>5}\n"
                )
                val.write(
                    f"{epoch:>6}  {self._fmt(va['loss'],11,5)}  {self._fmt(va['mae'],10,4)}  "
                    f"{self._fmt(va['rmse'],11,4)}  {self._fmt(va['mape'],10,4)}  "
                    f"{self._fmt(tr['mae'],10,4)}  "
                    f"{self._fmt(perf['epoch_time_sec'],8,1)}  {best:>5}\n"
                )
                metric = va["mae"] if va["mae"] is not None else va["loss"]
                if item["is_best"] and metric is not None:
                    best_epoch, best_metric = epoch, metric
                    best_metric_name = "val_mae" if va["mae"] is not None else "val_loss"
                    line = f"Best model at epoch {epoch} ({best_metric_name}={metric:.4f})\n"
                    train.write(line); val.write(line)
                metrics.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            final = records[-1] if records else _blank(0)
            epoch_times = [item["perf"]["epoch_time_sec"] for item in records
                           if item["perf"]["epoch_time_sec"] is not None]
            train_times = [item["perf"]["train_time_sec"] for item in records
                           if item["perf"]["train_time_sec"] is not None]
            val_times = [item["perf"]["val_time_sec"] for item in records
                         if item["perf"]["val_time_sec"] is not None]
            memories = [item["perf"]["peak_memory_gb"] for item in records
                        if item["perf"]["peak_memory_gb"] is not None]

            def average(values: list[float | None]) -> str:
                clean = [float(value) for value in values if value is not None]
                return f"{sum(clean) / len(clean):.2f}" if clean else "n/a"

            def summary_value(value: float | None, precision: int = 6) -> str:
                return f"{value:.{precision}f}" if value is not None else "n/a"

            train_steps = steps("train")
            val_steps = steps("val")
            train_per_step = ([value / train_steps for value in train_times]
                              if isinstance(train_steps, int) and train_steps else [])
            val_per_step = ([value / val_steps for value in val_times]
                            if isinstance(val_steps, int) and val_steps else [])
            finish_text = "finished normally" if status == "finished" else status
            footer = ["", "-" * 96,
                      f"Training {finish_text}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      "Summary", f"  completed_epochs: {len(records)}",
                      f"  best_epoch: {best_epoch if best_epoch is not None else 'n/a'}",
                      f"  best_{best_metric_name}: {summary_value(best_metric) if best_epoch is not None else 'n/a'}",
                      f"  final_train_loss: {summary_value(final['train']['loss'])}",
                      f"  final_train_mae: {summary_value(final['train']['mae'])}",
                      f"  final_val_loss: {summary_value(final['val']['loss'])}",
                      f"  final_val_mae: {summary_value(final['val']['mae'])}",
                      f"  final_val_rmse: {summary_value(final['val']['rmse'])}",
                      f"  final_val_mape: {summary_value(final['val']['mape'])}",
                      f"  total_time: {_human_time(total_time)}",
                      f"  avg_epoch_time_sec: {average(epoch_times)}",
                      f"  avg_train_sec_per_step: {average(train_per_step)}",
                      f"  avg_val_sec_per_step: {average(val_per_step)}",
                      f"  peak_memory_gb: {max(memories):.2f}" if memories else "  peak_memory_gb: n/a",
                      f"  metrics_jsonl: {metrics_path}",
                      f"  returncode: {returncode}",
                      f"  raw_log: {self.log_dir / 'raw.log'}", "-" * 96]
            block = "\n".join(footer) + "\n"
            train.write(block); val.write(block)
        with test_path.open("w", encoding="utf-8") as test_log:
            test_log.write(f"Testing started: {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "-" * 96 + "\nRun\n")
            for key, value in metadata.items():
                test_log.write(f"  {key}: {value}\n")
            test_log.write("-" * 96 + "\nResults\n")
            test_log.write(f"  selected_best_epoch: {best_epoch if best_epoch is not None else 'n/a'}\n")
            test_log.write(f"  selected_best_metric: {best_metric if best_epoch is not None else 'n/a'}\n")
            test_log.write(f"  test_mae: {test['mae'] if test['mae'] is not None else 'n/a'}\n")
            test_log.write(f"  test_rmse: {test['rmse'] if test['rmse'] is not None else 'n/a'}\n")
            test_log.write(f"  test_mape: {test['mape'] if test['mape'] is not None else 'n/a'}\n")
            test_log.write(f"  status: {status}\n  returncode: {returncode}\n")
            test_log.write(f"Testing record finished: {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "-" * 96 + "\n")
        (self.run_dir / "config.json").write_text(json.dumps(self.config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result = {
            **metadata, "status": status, "returncode": returncode,
            "completed_epochs": len(records),
            "best_epoch": best_epoch,
            "best_metric_name": best_metric_name if best_epoch is not None else None,
            "best_metric": best_metric if best_epoch is not None else None,
            "test": test, "total_time_sec": total_time,
            "logs": str(self.log_dir),
        }
        (self.run_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[info] standardized baseline logs saved to {self.log_dir}")
