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
        self._test_path = log_dir / "test.log"
        self._metrics_path = log_dir / "metrics.jsonl"
        self._train_f = self._train_path.open("w", encoding="utf-8", buffering=1)
        self._val_f = self._val_path.open("w", encoding="utf-8", buffering=1)
        self._test_f = self._test_path.open("w", encoding="utf-8", buffering=1)
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
        val: dict[str, float] | None,
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
        val_mae = f"{val['mae']:.4f}" if val is not None else "-"
        t_line = (
            f"{epoch:>6}  {train['loss']:>11.5f}  {train['mae']:>10.4f}  {train['rmse']:>11.4f}  "
            f"{val_mae:>10}  {lr:>10.2e}  {train_s:>8.1f}  {val_s:>7.1f}  "
            f"{epoch_s:>8.1f}  {mem_gb:>7.2f}  {best_mark:>5}"
        )
        self._train_f.write(t_line + "\n")
        self._log_dual_moe(self._train_f, train)
        if val is not None:
            v_line = (
                f"{epoch:>6}  {val['loss']:>11.5f}  {val['mae']:>10.4f}  {val['rmse']:>11.4f}  "
                f"{train['mae']:>10.4f}  {epoch_s:>8.1f}  {best_mark:>5}"
            )
            self._val_f.write(v_line + "\n")
            self._log_dual_moe(self._val_f, val)
        self._metrics_f.write(
            json.dumps(
                {"epoch": epoch, "train": train, "val": val, "perf": perf, "is_best": is_best},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    @staticmethod
    def _log_dual_moe(stream, metrics):
        if "mae_expert_fine" not in metrics:
            return
        recovery = {k: v for k, v in metrics.items() if k.startswith(('recovery_', 'l_recoverability'))}
        if recovery:
            stream.write('  recoverability (model-relative, not calibrated): '+json.dumps(recovery, sort_keys=True)+'\n')
        diagnostics = {k: v for k, v in metrics.items() if k.startswith('backend_diag_')}
        if diagnostics:
            # Opt-in only; full machine-readable diagnostics also in .jsonl.
            stream.write('  backend diagnostics: '+json.dumps(diagnostics, sort_keys=True)+'\n')
        def fmt(keys):
            return "/".join(f"{metrics[k]:.4f}" if k in metrics else "n/a" for k in keys)
        def expert_order(key):
            return int(key.rsplit('_e', 1)[1].split('_', 1)[0])
        domain = 'missing' if 'aggregation_mid_missing_count' in metrics else 'observed'
        front = " ".join(f"{s}="+fmt(sorted((k for k in metrics if k.startswith(f"aggregation_{s}_{domain}_e") and k.endswith("_mean")), key=expert_order)) for s in ("mid", "coarse"))
        completion_keys = sorted((k for k in metrics if k.startswith('completion_missing_e') and k.endswith('_mean')), key=expert_order)
        labels = 'routed experts' if completion_keys else 'f/m/c'
        back = fmt(completion_keys or [f"completion_missing_{s}_mean" for s in ("fine", "mid", "coarse")])
        stream.write(f"  aggregation({domain},components): {front}; completion(missing,{labels}): {back}\n")
        if 'completion_shared_always_active' in metrics:
            stream.write('  completion shared(always on, excluded from Top-K/balance) MAE/RMSE: '+fmt(['mae_completion_shared','rmse_completion_shared'])+'\n')
            routed = ' '.join(f'e{i}='+fmt([f'mae_completion_e{i}_with_shared', f'rmse_completion_e{i}_with_shared']) for i in range(len(completion_keys)))
            stream.write('  completion routed+shared MAE/RMSE: '+routed+'\n')
        errors = " ".join(f"{s}="+fmt([f"mae_expert_{s}", f"rmse_expert_{s}"]) for s in ("fine", "mid", "coarse"))
        stream.write(f"  expert MAE/RMSE: {errors}\n")
        regions = " ".join(f"{s}="+fmt([f"aggregation_{s}_assignment_entropy", f"aggregation_{s}_effective_regions"]) for s in ("mid", "coarse"))
        stream.write(f"  regions(assignment entropy/effective count): {regions}\n")
        if 'normalized_correction_abs_mean' in metrics:
            stream.write(f"  bounded correction(abs mean/saturation): {fmt(['normalized_correction_abs_mean', 'residual_saturation_fraction'])}\n")
        if 'l_balance_aggregation' in metrics or 'l_balance_completion' in metrics:
            stream.write(f"  load balance(front/back): {fmt(['l_balance_aggregation', 'l_balance_completion'])}; weighted: {fmt(['l_balance_aggregation_weighted', 'l_balance_completion_weighted'])}\n")
            for name in ('aggregation_mid', 'aggregation_coarse', 'completion'):
                keys = sorted((k for k in metrics if k.startswith(f'topk_{name}_e') and k.endswith('_load')), key=expert_order)
                if keys:
                    stream.write(f"  topk {name}: load={fmt(keys)}; selected/token={fmt([f'topk_{name}_selected_per_token'])}\n")

    def log_test(self, metrics: dict[str, float] | None, extra: dict[str, Any] | None = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._test_f.write(f"Testing started: {ts}\n" + "-" * 96 + "\n")
        if extra:
            self._test_f.write("Run\n")
            for key, value in extra.items():
                self._test_f.write(f"  {key}: {value}\n")
        self._test_f.write("-" * 96 + "\nResults\n")
        if metrics:
            for key, value in metrics.items():
                text = f"{value:.6f}" if isinstance(value, float) else str(value)
                self._test_f.write(f"  {key}: {text}\n")
        else:
            self._test_f.write("  status: skipped (no test dataset)\n")
        self._metrics_f.write(json.dumps({"stage": "test", "metrics": metrics, "extra": extra},
                                         ensure_ascii=False, sort_keys=True) + "\n")
        self._test_f.write(f"Testing finished: {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "-" * 96 + "\n")

    def log_best(self, epoch: int, val_mae: float) -> None:
        line = f"Best model at epoch {epoch} (val_mae={val_mae:.4f})"
        self._train_f.write(line + "\n")
        self._val_f.write(line + "\n")

    # ── lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        self._train_f.close()
        self._val_f.close()
        self._test_f.close()
        self._metrics_f.close()

    @property
    def log_dir(self) -> Path:
        return self._train_path.parent
