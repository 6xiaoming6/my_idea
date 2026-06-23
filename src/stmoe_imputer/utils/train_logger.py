"""Training logger that writes readable logs and machine-readable metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TrainLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._train_path = log_dir / "train.log"
        self._val_path = log_dir / "val.log"
        self._metrics_path = log_dir / "metrics.jsonl"
        self._train_f = self._train_path.open("w", encoding="utf-8", buffering=1)
        self._val_f = self._val_path.open("w", encoding="utf-8", buffering=1)
        self._metrics_f = self._metrics_path.open("w", encoding="utf-8", buffering=1)

    # ── header / footer ──────────────────────────────────────────────

    def log_header(self, cfg: dict, extra: dict | None = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"Training started: {ts}", "-" * 96]
        if extra:
            lines.append("Run")
            for k, v in extra.items():
                lines.append(f"  {k}: {v}")
        lines += ["Config", json.dumps(cfg, indent=2, ensure_ascii=False), "-" * 96]
        for line in lines:
            self._train_f.write(line + "\n")

    def log_footer(self, summary: dict[str, Any] | None = None, status: str = "finished") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "Training finished normally" if status == "finished" else f"Training {status}"
        lines = ["", "-" * 96, f"{title}: {ts}"]
        if summary:
            lines.append("Summary")
            for k, v in summary.items():
                lines.append(f"  {k}: {v}")
        lines.append("-" * 96)
        for line in lines:
            self._train_f.write(line + "\n")
            self._val_f.write(line + "\n")

    # ── table headers ────────────────────────────────────────────────

    def log_table_headers(self) -> None:
        t_header = (
            f"{'epoch':>6}  {'loss':>11}  {'mae':>10}  {'rmse':>11}  "
            f"{'val_mae':>10}  {'lr':>10}  {'train_s':>8}  {'val_s':>7}  "
            f"{'epoch_s':>8}  {'mem_gb':>7}  {'best':>5}"
        )
        v_header = (
            f"{'epoch':>6}  {'loss':>11}  {'mae':>10}  {'rmse':>11}  "
            f"{'train_mae':>10}  {'epoch_s':>8}  {'best':>5}"
        )
        t_sep = "-" * len(t_header)
        v_sep = "-" * len(v_header)
        for f, h, s in [(self._train_f, t_header, t_sep), (self._val_f, v_header, v_sep)]:
            f.write(s + "\n")
            f.write(h + "\n")
            f.write(s + "\n")

    # ── per-epoch ───────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch: int,
        train: dict[str, float],
        val: dict[str, float],
        perf: dict[str, float] | None = None,
        is_best: bool = False,
    ) -> None:
        perf = perf or {}
        lr = train.get("lr", perf.get("lr", 0.0))
        train_s = perf.get("train_time_sec", 0.0)
        val_s = perf.get("val_time_sec", 0.0)
        epoch_s = perf.get("epoch_time_sec", train_s + val_s)
        mem_gb = perf.get("peak_memory_gb", 0.0)
        best_mark = "*" if is_best else ""
        t_line = (
            f"{epoch:>6}  {train['loss']:>11.5f}  {train['mae']:>10.4f}  {train['rmse']:>11.4f}  "
            f"{val['mae']:>10.4f}  {lr:>10.2e}  {train_s:>8.1f}  {val_s:>7.1f}  "
            f"{epoch_s:>8.1f}  {mem_gb:>7.2f}  {best_mark:>5}"
        )
        v_line = (
            f"{epoch:>6}  {val['loss']:>11.5f}  {val['mae']:>10.4f}  {val['rmse']:>11.4f}  "
            f"{train['mae']:>10.4f}  {epoch_s:>8.1f}  {best_mark:>5}"
        )
        self._train_f.write(t_line + "\n")
        self._val_f.write(v_line + "\n")
        self._metrics_f.write(
            json.dumps(
                {"epoch": epoch, "train": train, "val": val, "perf": perf, "is_best": is_best},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    def log_best(self, epoch: int, val_mae: float) -> None:
        line = f"Best model at epoch {epoch} (val_mae={val_mae:.4f})"
        self._train_f.write(line + "\n")
        self._val_f.write(line + "\n")

    # ── lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        self._train_f.close()
        self._val_f.close()
        self._metrics_f.close()

    @property
    def log_dir(self) -> Path:
        return self._train_path.parent
