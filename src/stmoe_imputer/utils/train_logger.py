"""Training logger that writes train/val metrics to separate files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class TrainLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._train_path = log_dir / "train.log"
        self._val_path = log_dir / "val.log"
        self._train_f = self._train_path.open("w", encoding="utf-8", buffering=1)
        self._val_f = self._val_path.open("w", encoding="utf-8", buffering=1)

    # ── header / footer ──────────────────────────────────────────────

    def log_header(self, cfg: dict, extra: dict | None = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"Training started: {ts}", "─" * 64]
        if extra:
            lines.append("─ Extra ─")
            for k, v in extra.items():
                lines.append(f"  {k}: {v}")
        lines += ["─ Config ─", json.dumps(cfg, indent=2, ensure_ascii=False), "─" * 64]
        for line in lines:
            self._train_f.write(line + "\n")

    def log_footer(self) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer = f"Training finished: {ts}"
        self._train_f.write(footer + "\n")
        self._val_f.write(footer + "\n")

    # ── table headers ────────────────────────────────────────────────

    def log_table_headers(self) -> None:
        t_header = f"{'epoch':>6}  {'loss':>11}  {'mae':>10}  {'rmse':>11}"
        v_header = f"{'epoch':>6}  {'loss':>11}  {'mae':>10}  {'rmse':>11}"
        t_sep = "─" * len(t_header)
        v_sep = "─" * len(v_header)
        for f, h, s in [(self._train_f, t_header, t_sep), (self._val_f, v_header, v_sep)]:
            f.write(s + "\n")
            f.write(h + "\n")
            f.write(s + "\n")

    # ── per-epoch ───────────────────────────────────────────────────

    def log_epoch(self, epoch: int, train: dict[str, float], val: dict[str, float]) -> None:
        t_line = f"{epoch:>6}  {train['loss']:>11.5f}  {train['mae']:>10.4f}  {train['rmse']:>11.4f}"
        v_line = f"{epoch:>6}  {val['loss']:>11.5f}  {val['mae']:>10.4f}  {val['rmse']:>11.4f}"
        self._train_f.write(t_line + "\n")
        self._val_f.write(v_line + "\n")
        self._train_f.write(
            "metrics " + json.dumps({"epoch": epoch, **train}, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._val_f.write(
            "metrics " + json.dumps({"epoch": epoch, **val}, ensure_ascii=False, sort_keys=True) + "\n"
        )

    def log_best(self, epoch: int, val_mae: float) -> None:
        line = f"Best model at epoch {epoch} (val_mae={val_mae:.4f})"
        self._train_f.write(line + "\n")
        self._val_f.write(line + "\n")

    # ── lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        self._train_f.close()
        self._val_f.close()

    @property
    def log_dir(self) -> Path:
        return self._train_path.parent
