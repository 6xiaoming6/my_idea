from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.config import deep_update, load_config, save_config
from stmoe_imputer.data import build_datasets, build_loader
from stmoe_imputer.engine import build_optimizer, build_scheduler, evaluate, train_one_epoch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils import get_device, set_seed
from stmoe_imputer.utils.checkpoint import save_checkpoint
from stmoe_imputer.utils.train_logger import TrainLogger


def _run_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return str(device)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    return 0.0


def _dataset_shape(dataset) -> str:
    arrays = getattr(dataset, "arrays", None)
    x_key = getattr(dataset, "x_key", None)
    if arrays is not None and x_key in arrays:
        return str(tuple(arrays[x_key].shape))
    t = getattr(dataset, "t", None)
    h = getattr(dataset, "h", None)
    w = getattr(dataset, "w", None)
    c = getattr(dataset, "c_in", None)
    if None not in (t, h, w, c):
        return f"(N,{c},{t},{h},{w})"
    return "unknown"


def _mask_summary(dataset) -> str:
    masks = getattr(dataset, "loaded_masks", None)
    if masks is None:
        masks = getattr(dataset, "masks", None)
    if masks is None:
        return "unknown"
    observed = float(masks.float().mean().item())
    return (
        f"shape={tuple(masks.shape)}, observed_rate={observed:.4f}, "
        f"missing_rate={1.0 - observed:.4f}"
    )


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    seconds_i = int(round(seconds))
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _safe_path_part(value: object) -> str:
    text = str(value).strip().replace(" ", "_")
    safe = []
    for char in text:
        if char.isalnum() or char in {"_", "-", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "unknown"


def _experiment_parts(name: str) -> tuple[str, str]:
    safe_name = _safe_path_part(name)
    if safe_name == "full":
        return "full", "model"
    if safe_name.startswith("ablation_"):
        return "ablation", safe_name.removeprefix("ablation_")
    if safe_name in {"smoke", "debug", "test"} or safe_name.startswith(("smoke_", "debug_", "test_")):
        return "debug", safe_name
    return "custom", safe_name


def _rate_part(rate: object) -> str:
    try:
        rate_value = float(rate)
    except (TypeError, ValueError):
        return f"rate{_safe_path_part(rate)}"
    return f"rate{rate_value:g}"


def _unique_run_dir(base_dir: Path, run_id: str) -> Path:
    candidate = base_dir / run_id
    if not candidate.exists():
        return candidate
    for idx in range(2, 1000):
        candidate = base_dir / f"{run_id}_{idx:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot create a unique run directory under {base_dir}")


def _append_experiment_index(index_path: Path, row: dict[str, object]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "started_at",
        "finished_at",
        "status",
        "run_dir",
        "dataset",
        "experiment_type",
        "variant",
        "mask_pattern",
        "missing_rate",
        "seed",
        "batch_size",
        "epochs_config",
        "completed_epochs",
        "best_epoch",
        "best_val_mae",
        "final_train_mae",
        "final_val_mae",
        "total_time_sec",
        "avg_epoch_time_sec",
        "avg_train_sec_per_step",
        "avg_val_sec_per_step",
        "peak_memory_gb",
        "git_commit",
    ]
    exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config","-c", default="configs/presets/default.json")
    parser.add_argument("--override_config", default=None, help="Optional JSON patch for ablations.")
    parser.add_argument("--train_npz", default=None)
    parser.add_argument("--val_npz", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting training curves.")
    parser.add_argument("--name", "-n", default="run", help="Run name for the output directory.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal console output (compact one-line per epoch).")
    return parser.parse_args()


def _plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss Curve")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["train_mae"], "b-", label="Train MAE")
    ax2.plot(epochs, history["val_mae"], "r-", label="Val MAE")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE"); ax2.set_title("MAE Curve")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = output_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[info] curves saved to {save_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.override_config:
        cfg = deep_update(cfg, load_config(args.override_config))
    set_seed(cfg.get("seed", 42))

    dataset_name = cfg["data"].get("dataset_name", "unknown")
    mask_cfg = cfg["data"].get("mask", {})
    mask_pattern = mask_cfg.get("pattern", "random")
    mask_rate = mask_cfg.get("missing_rate", mask_cfg.get("mask_rate", 0.0))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_type, variant = _experiment_parts(args.name)
    run_id = f"{ts}_seed{_safe_path_part(cfg.get('seed', 42))}_bs{_safe_path_part(cfg['data']['batch_size'])}"
    run_base_dir = (
        ROOT
        / cfg["output_dir"]
        / _safe_path_part(dataset_name)
        / experiment_type
        / variant
        / _safe_path_part(mask_pattern)
        / _rate_part(mask_rate)
    )
    run_dir = _unique_run_dir(run_base_dir, run_id)
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.json")

    device = get_device(cfg.get("device", "auto"))
    train_ds, val_ds = build_datasets(cfg, args.train_npz, args.val_npz, synthetic=args.synthetic)
    train_loader = build_loader(train_ds, cfg, shuffle=True)
    val_loader = build_loader(val_ds, cfg, shuffle=False)

    model = DualBranchSTImputer.from_config(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    git_commit = _run_git_commit()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = TrainLogger(log_dir)
    logger.log_header(cfg, extra={
        "run_dir": str(run_dir),
        "command": " ".join(sys.argv),
        "git_commit": git_commit,
        "python": platform.python_version(),
        "train_npz": args.train_npz or "(synthetic)",
        "val_npz": args.val_npz or "(synthetic)",
        "dataset": dataset_name,
        "experiment_type": experiment_type,
        "variant": variant,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_steps_per_epoch": len(train_loader),
        "val_steps_per_epoch": len(val_loader),
        "batch_size": cfg["data"]["batch_size"],
        "mask_pattern": mask_pattern,
        "mask_rate_config": mask_rate,
        "train_mask": _mask_summary(train_ds),
        "val_mask": _mask_summary(val_ds),
        "data_shape": _dataset_shape(train_ds),
        "device": str(device),
        "device_name": _device_name(device),
        "total_params": f"{total_params:,}",
        "trainable_params": f"{trainable_params:,}",
    })
    logger.log_table_headers()

    best_mae = float("inf")
    best_epoch = 0
    last_improved_epoch = 0
    early_cfg = cfg["train"].get("early_stopping", {})
    early_mode = early_cfg.get("mode", "min")
    early_best = float("inf") if early_mode == "min" else -float("inf")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    epoch_perfs: list[dict[str, float]] = []
    completed_epochs = 0
    total_start = time.perf_counter()
    status = "finished"
    try:
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            epoch_start = time.perf_counter()
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            _sync_device(device)
            train_start = time.perf_counter()
            train_logs = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
            _sync_device(device)
            train_time = time.perf_counter() - train_start

            val_start = time.perf_counter()
            val_logs = evaluate(model, val_loader, device, cfg, desc=f"val epoch {epoch}", epoch=epoch)
            _sync_device(device)
            val_time = time.perf_counter() - val_start

            epoch_time = time.perf_counter() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            train_logs["lr"] = current_lr
            perf = {
                "train_time_sec": train_time,
                "val_time_sec": val_time,
                "epoch_time_sec": epoch_time,
                "train_sec_per_step": train_time / max(1, len(train_loader)),
                "val_sec_per_step": val_time / max(1, len(val_loader)),
                "train_samples_per_sec": len(train_ds) / train_time if train_time > 0 else 0.0,
                "val_samples_per_sec": len(val_ds) / val_time if val_time > 0 else 0.0,
                "peak_memory_gb": _peak_memory_gb(device),
            }
            is_best = val_logs["mae"] < best_mae
            if args.quiet:
                print(f"[E {epoch:3d}/{cfg['train']['epochs']}] "
                      f"loss={train_logs['loss']:7.2f} mae={val_logs['mae']:7.2f} "
                      f"rmse={val_logs['rmse']:7.2f} lr={current_lr:.2e} "
                      f"time={epoch_time:.1f}s mem={perf['peak_memory_gb']:.2f}GB")
            else:
                print(
                    f"epoch={epoch} lr={current_lr:.2e} "
                    f"time={epoch_time:.1f}s mem={perf['peak_memory_gb']:.2f}GB "
                    f"train={train_logs} val={val_logs}"
                )
            if scheduler is not None:
                scheduler.step()

            logger.log_epoch(epoch, train_logs, val_logs, perf=perf, is_best=is_best)
            completed_epochs = epoch
            epoch_perfs.append(perf)
            history["train_loss"].append(float(train_logs["loss"]))
            history["val_loss"].append(float(val_logs["loss"]))
            history["train_mae"].append(float(train_logs["mae"]))
            history["val_mae"].append(float(val_logs["mae"]))

            metrics = {f"train_{key}": value for key, value in train_logs.items()}
            metrics.update({f"val_{key}": value for key, value in val_logs.items()})
            # checkpoint saving disabled — re-enable when needed:
            # save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, metrics, cfg)
            # if epoch % cfg["train"].get("save_every", 1) == 0:
            #     save_checkpoint(ckpt_dir / f"epoch_{epoch}.pt", model, optimizer, epoch, metrics, cfg)
            if is_best:
                best_mae = val_logs["mae"]
                best_epoch = epoch
                last_improved_epoch = epoch
                # save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, metrics, cfg)
                logger.log_best(epoch, best_mae)

            if early_cfg.get("enabled", False):
                monitor = early_cfg.get("monitor", "val_mae")
                patience = int(early_cfg.get("patience", 20))
                metric_key = monitor[4:] if monitor.startswith("val_") else monitor
                current = float(val_logs[metric_key])
                improved = current < early_best if early_mode == "min" else current > early_best
                if improved:
                    early_best = current
                    last_improved_epoch = epoch
                elif epoch - last_improved_epoch >= patience:
                    print(f"[info] early stopping at epoch {epoch} ({monitor}={current:.6f})")
                    break
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        total_time = time.perf_counter() - total_start
        avg_epoch_time = sum(p["epoch_time_sec"] for p in epoch_perfs) / max(1, len(epoch_perfs))
        avg_train_step = sum(p["train_sec_per_step"] for p in epoch_perfs) / max(1, len(epoch_perfs))
        avg_val_step = sum(p["val_sec_per_step"] for p in epoch_perfs) / max(1, len(epoch_perfs))
        max_mem = max((p["peak_memory_gb"] for p in epoch_perfs), default=0.0)
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = {
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch or "n/a",
            "best_val_mae": f"{best_mae:.6f}" if best_epoch else "n/a",
            "final_train_loss": f"{history['train_loss'][-1]:.6f}" if history["train_loss"] else "n/a",
            "final_train_mae": f"{history['train_mae'][-1]:.6f}" if history["train_mae"] else "n/a",
            "final_val_loss": f"{history['val_loss'][-1]:.6f}" if history["val_loss"] else "n/a",
            "final_val_mae": f"{history['val_mae'][-1]:.6f}" if history["val_mae"] else "n/a",
            "total_time": _fmt_seconds(total_time),
            "avg_epoch_time_sec": f"{avg_epoch_time:.2f}",
            "avg_train_sec_per_step": f"{avg_train_step:.4f}",
            "avg_val_sec_per_step": f"{avg_val_step:.4f}",
            "peak_memory_gb": f"{max_mem:.2f}",
            "metrics_jsonl": str(log_dir / "metrics.jsonl"),
        }
        logger.log_footer(summary=summary, status=status)
        logger.close()
        _append_experiment_index(
            ROOT / cfg["output_dir"] / "summary" / "experiment_index.csv",
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "status": status,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "dataset": dataset_name,
                "experiment_type": experiment_type,
                "variant": variant,
                "mask_pattern": mask_pattern,
                "missing_rate": mask_rate,
                "seed": cfg.get("seed", 42),
                "batch_size": cfg["data"]["batch_size"],
                "epochs_config": cfg["train"]["epochs"],
                "completed_epochs": completed_epochs,
                "best_epoch": best_epoch or "",
                "best_val_mae": f"{best_mae:.6f}" if best_epoch else "",
                "final_train_mae": f"{history['train_mae'][-1]:.6f}" if history["train_mae"] else "",
                "final_val_mae": f"{history['val_mae'][-1]:.6f}" if history["val_mae"] else "",
                "total_time_sec": f"{total_time:.2f}",
                "avg_epoch_time_sec": f"{avg_epoch_time:.2f}",
                "avg_train_sec_per_step": f"{avg_train_step:.4f}",
                "avg_val_sec_per_step": f"{avg_val_step:.4f}",
                "peak_memory_gb": f"{max_mem:.2f}",
                "git_commit": git_commit,
            },
        )
    print(f"[info] logs saved to {log_dir}")
    if not args.no_plot:
        _plot_history(history, run_dir)


if __name__ == "__main__":
    main()
