#!/usr/bin/env python3
"""Pre-generate fixed binary masks as CSV files for reproducibility.

Each point is missing independently (Bernoulli), matching what --mask_pattern random
produces on-the-fly during training.

Usage:
  python scripts/generate_fixed_masks.py \
      --train_npz data/TaxiBJ/taxibj_train.npz \
      --val_npz data/TaxiBJ/taxibj_val.npz \
      --mask_rate 0.4 --seed 42

Output:
  data/<Dataset>/fixed_masks/train_rate0p4_seed42.csv
  data/<Dataset>/fixed_masks/val_rate0p4_seed42.csv
  data/<Dataset>/fixed_masks/train_rate0p4_seed42.json  (metadata)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.data.masks import random_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed masks as CSV files")
    parser.add_argument("--train_npz", required=True, help="Path to training NPZ file")
    parser.add_argument("--val_npz", required=True, help="Path to validation NPZ file")
    parser.add_argument("--mask_rate", "-r", type=float, default=0.4, help="Missing rate (default: 0.4)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Base seed for reproducibility (default: 42)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save CSVs (default: <npz_dir>/fixed_masks/)")
    return parser.parse_args()


def _infer_output_dir(train_npz: str) -> Path:
    npz_path = Path(train_npz)
    return npz_path.parent / "fixed_masks"


def generate_split(npz_path: str, mask_rate: float, seed: int,
                   output_dir: Path, split: str) -> Path:
    """Generate fixed masks for one data split and save as CSV.

    Uses the identical seed logic as FlowNPZDataset (random mode):
        generator = torch.Generator().manual_seed(seed + idx)
    """
    with np.load(npz_path) as data:
        x_key = "x_f_gt" if "x_f_gt" in data else "x_f"
        x = data[x_key]
        n_samples, _, t, h, w = x.shape

    flat_len = t * h * w
    all_masks = np.empty((n_samples, flat_len), dtype=np.int8)

    for idx in range(n_samples):
        generator = torch.Generator().manual_seed(seed + idx)
        mask = random_mask(t, h, w, mask_rate, generator)  # [1, T, H, W]
        all_masks[idx] = mask.reshape(-1).numpy().astype(np.int8)

    output_dir.mkdir(parents=True, exist_ok=True)
    rate_str = str(mask_rate).replace(".", "p")
    stem = f"{split}_rate{rate_str}_seed{seed}"
    csv_path = output_dir / f"{stem}.csv"
    np.savetxt(str(csv_path), all_masks, delimiter=",", fmt="%d")
    print(f"[{split}] saved {n_samples} masks ({n_samples}×{flat_len}) → {csv_path} "
          f"({os.path.getsize(csv_path) / 1024 / 1024:.1f} MB)")

    actual_rate = 1.0 - all_masks.mean()
    print(f"        actual missing rate = {actual_rate:.4f} (target = {mask_rate})")

    meta_path = output_dir / f"{stem}.json"
    meta = {
        "n_samples": n_samples,
        "t": t, "h": h, "w": w,
        "mask_rate": mask_rate,
        "actual_missing_rate": float(actual_rate),
        "seed": seed,
        "created_at": datetime.now().isoformat(),
        "npz_source": str(npz_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"        metadata → {meta_path}")

    return csv_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _infer_output_dir(args.train_npz)

    print(f"Generating fixed masks:")
    print(f"  mask_rate  = {args.mask_rate}")
    print(f"  seed       = {args.seed}")
    print(f"  output_dir = {output_dir}")
    print()

    train_csv = generate_split(args.train_npz, args.mask_rate, args.seed, output_dir, "train")
    val_csv = generate_split(args.val_npz, args.mask_rate, args.seed, output_dir, "val")

    print()
    print("Done. Use with run_all_ablations.sh:")
    suffix = train_csv.stem.replace("train_", "")
    print(f"  bash scripts/run_all_ablations.sh --dataset <DS> --gpu 0 \\")
    print(f"    --mask_pattern fixed --fixed_masks_suffix {suffix}")
    print()
    print("Or directly:")
    print(f'  "mask": {{')
    print(f'    "pattern": "fixed",')
    print(f'    "fixed_train_csv": "{train_csv}",')
    print(f'    "fixed_val_csv": "{val_csv}"')
    print(f'  }}')


if __name__ == "__main__":
    main()
