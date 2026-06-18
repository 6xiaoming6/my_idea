#!/usr/bin/env python3
"""Generate offline fixed/random masks following model_designs/fixed和random缺失模式策略.md.

Fixed:
  one H*W mask row per split file; all samples and all time steps share it.

Random:
  N H*W mask rows per split file; sample i always uses row i.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline mask CSV files")
    parser.add_argument("--train_npz", required=True, help="Path to training NPZ file")
    parser.add_argument("--val_npz", required=True, help="Path to validation NPZ file")
    parser.add_argument("--test_npz", default=None, help="Optional test NPZ file")
    parser.add_argument(
        "--pattern",
        "-p",
        choices=("fixed", "random"),
        default="fixed",
        help="Mask pattern to generate",
    )
    parser.add_argument("--mask_rate", "-r", type=float, default=0.4, help="Missing rate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save CSVs (default: <npz_dir>/<pattern>_mask/<rate>/)",
    )
    return parser.parse_args()


def _rate_str(mask_rate: float) -> str:
    return str(mask_rate).replace(".", "p")


def _infer_output_dir(train_npz: str, pattern: str, mask_rate: float) -> Path:
    return Path(train_npz).parent / f"{pattern}_mask" / str(mask_rate)


def _load_shape(npz_path: str) -> tuple[int, int, int, int]:
    with np.load(npz_path) as data:
        x_key = "x_f_gt" if "x_f_gt" in data else "x_f"
        x = data[x_key]
        n_samples, _, t, h, w = x.shape
    return int(n_samples), int(t), int(h), int(w)


def _one_spatial_mask(h: int, w: int, missing_rate: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
    num_positions = h * w
    num_observed = int(math.ceil(num_positions * (1.0 - missing_rate)))
    mask = np.zeros(num_positions, dtype=np.int8)
    if num_observed > 0:
        chosen = rng.choice(num_positions, size=num_observed, replace=False)
        mask[chosen] = 1
    return mask


def _generate_fixed(h: int, w: int, missing_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _one_spatial_mask(h, w, missing_rate, rng).reshape(1, -1)


def _generate_random(
    n_samples: int,
    h: int,
    w: int,
    missing_rate: float,
    seed: int,
) -> np.ndarray:
    masks = np.empty((n_samples, h * w), dtype=np.int8)
    for idx in range(n_samples):
        rng = np.random.default_rng(seed + idx)
        masks[idx] = _one_spatial_mask(h, w, missing_rate, rng)
    return masks


def _save_split(
    npz_path: str,
    output_dir: Path,
    split: str,
    pattern: str,
    mask_rate: float,
    seed: int,
    generation_seed: int,
    fixed_mask: np.ndarray | None = None,
) -> Path:
    n_samples, t, h, w = _load_shape(npz_path)
    if pattern == "fixed":
        masks = fixed_mask if fixed_mask is not None else _generate_fixed(h, w, mask_rate, seed)
    else:
        masks = _generate_random(n_samples, h, w, mask_rate, generation_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = split
    csv_path = output_dir / f"{stem}.csv"
    np.savetxt(str(csv_path), masks, delimiter=",", fmt="%d")

    actual_missing_rate = 1.0 - float(masks.mean())
    meta = {
        "pattern": pattern,
        "split": split,
        "n_samples": n_samples,
        "rows": int(masks.shape[0]),
        "t": t,
        "h": h,
        "w": w,
        "columns": int(masks.shape[1]),
        "columns_semantics": "H*W; broadcast to all T steps at load time",
        "mask_rate": mask_rate,
        "actual_missing_rate": actual_missing_rate,
        "seed": seed,
        "generation_seed": generation_seed,
        "created_at": datetime.now().isoformat(),
        "npz_source": str(npz_path),
    }
    with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f"[{split}] {pattern}: saved {masks.shape[0]}x{masks.shape[1]} -> {csv_path} "
        f"(actual missing={actual_missing_rate:.4f})"
    )
    return csv_path


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _infer_output_dir(args.train_npz, args.pattern, args.mask_rate)
    )

    print("Generating offline masks:")
    print(f"  pattern    = {args.pattern}")
    print(f"  mask_rate  = {args.mask_rate}")
    print(f"  seed       = {args.seed}")
    print(f"  output_dir = {output_dir}")
    print()

    fixed_mask = None
    if args.pattern == "fixed":
        _, _, h, w = _load_shape(args.train_npz)
        fixed_mask = _generate_fixed(h, w, args.mask_rate, args.seed)

    split_seed = {"train": args.seed, "val": args.seed, "test": args.seed}
    if args.pattern == "random":
        split_seed = {"train": args.seed, "val": args.seed + 100000, "test": args.seed + 200000}

    train_csv = _save_split(
        args.train_npz,
        output_dir,
        "train",
        args.pattern,
        args.mask_rate,
        args.seed,
        split_seed["train"],
        fixed_mask,
    )
    val_csv = _save_split(
        args.val_npz,
        output_dir,
        "val",
        args.pattern,
        args.mask_rate,
        args.seed,
        split_seed["val"],
        fixed_mask,
    )
    test_csv = None
    if args.test_npz:
        test_csv = _save_split(
            args.test_npz,
            output_dir,
            "test",
            args.pattern,
            args.mask_rate,
            args.seed,
            split_seed["test"],
            fixed_mask,
        )

    print()
    print("Config patch:")
    print('  "mask": {')
    print(f'    "pattern": "{args.pattern}",')
    print(f'    "missing_rate": {args.mask_rate},')
    print(f'    "train_csv": "{train_csv}",')
    suffix = "," if test_csv is not None else ""
    print(f'    "val_csv": "{val_csv}"{suffix}')
    if test_csv is not None:
        print(f'    "test_csv": "{test_csv}"')
    print("  }")


if __name__ == "__main__":
    main()
