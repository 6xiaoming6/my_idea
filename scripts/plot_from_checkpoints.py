"""Extract metrics from saved checkpoints and plot training curves.
Supports both old flat layout and new checkpoints/ subdirectory layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch


def _find_checkpoints(root: Path) -> list[Path]:
    """Look for epoch_*.pt files in root or root/checkpoints/."""
    direct = sorted(root.glob("epoch_*.pt"))
    if direct:
        return direct
    nested = sorted((root / "checkpoints").glob("epoch_*.pt")) if (root / "checkpoints").is_dir() else []
    return nested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="outputs/taxibj",
                        help="Path to a run directory (e.g. outputs/taxibj/TaxiBJ/20260527_120000_run)")
    args = parser.parse_args()

    run_dir = Path(args.output_dir)
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parents[1] / run_dir

    ckpt_files = _find_checkpoints(run_dir)
    if not ckpt_files:
        print(f"No epoch_*.pt files found in {run_dir}")
        return

    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_mae: list[float] = []
    val_mae: list[float] = []

    for path in ckpt_files:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ckpt["metrics"]
        epochs.append(ckpt["epoch"])
        train_loss.append(m["train_loss"])
        val_loss.append(m["val_loss"])
        train_mae.append(m["train_mae"])
        val_mae.append(m["val_mae"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, "bo-", label="Train Loss")
    ax1.plot(epochs, val_loss, "ro-", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss Curve")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_mae, "bo-", label="Train MAE")
    ax2.plot(epochs, val_mae, "ro-", label="Val MAE")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE"); ax2.set_title("MAE Curve")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = run_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plotted {len(epochs)} checkpoints → {save_path}")


if __name__ == "__main__":
    main()
