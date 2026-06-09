from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/TaxiBJ")
    parser.add_argument("--out", default="data/taxibj_windows.npz")
    parser.add_argument("--years", nargs="+", type=int, default=[2013, 2014, 2015, 2016])
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--w", type=int, default=32)
    return parser.parse_args()


def read_year(path: Path, h: int, w: int) -> np.ndarray:
    grid = pd.read_csv(path)
    times = sorted(grid["time"].unique())
    time_to_idx = {time: idx for idx, time in enumerate(times)}
    dense = np.zeros((len(times), 2, h, w), dtype=np.float32)
    t_idx = grid["time"].map(time_to_idx).to_numpy()
    rows = grid["row_id"].to_numpy(dtype=np.int64)
    cols = grid["column_id"].to_numpy(dtype=np.int64)
    dense[t_idx, 0, rows, cols] = grid["inflow"].to_numpy(dtype=np.float32)
    dense[t_idx, 1, rows, cols] = grid["outflow"].to_numpy(dtype=np.float32)
    return dense


def make_windows(x_all: np.ndarray, window: int, stride: int) -> np.ndarray:
    starts = range(0, x_all.shape[0] - window + 1, stride)
    return np.stack([x_all[start : start + window] for start in starts], axis=0)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    arrays = [read_year(data_dir / f"TAXIBJ{year}.grid", args.h, args.w) for year in args.years]
    x_all = np.concatenate(arrays, axis=0)
    windows = make_windows(x_all, args.window, args.stride)
    windows = np.transpose(windows, (0, 2, 1, 3, 4))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x_f_gt=windows)
    print(f"saved {out} with x_f_gt shape {windows.shape}")


if __name__ == "__main__":
    main()
