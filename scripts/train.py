from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.config import deep_update, load_config, save_config
from stmoe_imputer.data import build_datasets, build_loader
from stmoe_imputer.engine import build_optimizer, build_scheduler, evaluate, train_one_epoch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils import get_device, set_seed
from stmoe_imputer.utils.checkpoint import save_checkpoint
from stmoe_imputer.utils.train_logger import TrainLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config","-c", default="configs/default.json")
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
    mask_suffix = f"{mask_pattern}{mask_rate}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / cfg["output_dir"] / dataset_name / f"{args.name}_{mask_suffix}_{ts}"
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
    logger = TrainLogger(log_dir)
    logger.log_header(cfg, extra={
        "train_npz": args.train_npz or "(synthetic)",
        "val_npz": args.val_npz or "(synthetic)",
        "device": str(device),
        "total_params": f"{total_params:,}",
        "trainable_params": f"{trainable_params:,}",
    })
    logger.log_table_headers()

    best_mae = float("inf")
    last_improved_epoch = 0
    early_cfg = cfg["train"].get("early_stopping", {})
    early_mode = early_cfg.get("mode", "min")
    early_best = float("inf") if early_mode == "min" else -float("inf")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_logs = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        val_logs = evaluate(model, val_loader, device, cfg, desc=f"val epoch {epoch}", epoch=epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        train_logs["lr"] = current_lr
        if args.quiet:
            print(f"[E {epoch:3d}/{cfg['train']['epochs']}] "
                  f"loss={train_logs['loss']:7.2f} mae={val_logs['mae']:7.2f} "
                  f"rmse={val_logs['rmse']:7.2f} lr={current_lr:.2e}")
        else:
            print(f"epoch={epoch} lr={current_lr:.2e} train={train_logs} val={val_logs}")
        if scheduler is not None:
            scheduler.step()

        logger.log_epoch(epoch, train_logs, val_logs)
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
        if val_logs["mae"] < best_mae:
            best_mae = val_logs["mae"]
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

    logger.log_footer()
    logger.close()
    print(f"[info] logs saved to {log_dir}")
    if not args.no_plot:
        _plot_history(history, run_dir)


if __name__ == "__main__":
    main()
