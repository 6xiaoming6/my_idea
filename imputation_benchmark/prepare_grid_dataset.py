#!/usr/bin/env python3
"""Adapt this project's grid datasets for the unmodified benchmark baselines.

The benchmark expects a traffic tensor shaped ``[time, nodes, features]`` and
two files named ``true_data_<type>_<rate>_v2.npz`` and
``miss_data_<type>_<rate>_v2.npz``.  Our training data are sliding windows
shaped ``[samples, channels, time, height, width]``.  This program only
converts data and masks; it deliberately does not import or alter a model.

Examples
--------
python prepare_grid_dataset.py --dataset TaxiBJ --mask fixed --rate 0.4
python prepare_grid_dataset.py --dataset CHAP --mask SR-TR --rate 0.4
python prepare_grid_dataset.py --dataset BikeNYC --mask fixed --rate 0.4 --channel 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DatasetProfile:
    train: str
    val: str
    test: str
    mask_root: str


PROFILES = {
    "TaxiBJ": DatasetProfile(
        "my_idea/data/TaxiBJ/taxibj_train.npz",
        "my_idea/data/TaxiBJ/taxibj_val.npz",
        "my_idea/data/TaxiBJ/taxibj_test.npz",
        "my_idea/data/TaxiBJ",
    ),
    "BikeNYC": DatasetProfile(
        "my_idea/data/BikeNYC/bikenyc_train.npz",
        "my_idea/data/BikeNYC/bikenyc_val.npz",
        "my_idea/data/BikeNYC/bikenyc_test.npz",
        "my_idea/data/BikeNYC",
    ),
    "CHAP": DatasetProfile(
        "my_idea/data/CHAP/beijing/chap_beijing_train.npz",
        "my_idea/data/CHAP/beijing/chap_beijing_val.npz",
        "my_idea/data/CHAP/beijing/chap_beijing_test.npz",
        "my_idea/data/CHAP/beijing",
    ),
}


def _load_windows(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        key = "x_f_gt" if "x_f_gt" in npz else "x_f"
        if key not in npz:
            raise KeyError(f"{path} must contain x_f_gt or x_f.")
        windows = np.asarray(npz[key], dtype=np.float32)
    if windows.ndim != 5:
        raise ValueError(f"{path}: expected [samples,C,T,H,W], got {windows.shape}.")
    return windows


def _restore_series(windows: np.ndarray, stride: int, source: Path) -> np.ndarray:
    """Undo a sliding-window view while checking that its overlap is genuine."""
    n, channels, length, height, width = windows.shape
    if stride < 1 or stride > length:
        raise ValueError(f"--stride must be in [1, {length}], got {stride}.")
    expected = (n - 1) * stride + length
    series = np.empty((expected, channels, height, width), dtype=np.float32)
    filled = np.zeros(expected, dtype=bool)
    for index, window in enumerate(windows):
        window = window.transpose(1, 0, 2, 3)  # [C,T,H,W] -> [T,C,H,W]
        start = index * stride
        end = start + length
        overlap = filled[start:end]
        if overlap.any() and not np.array_equal(series[start:end][overlap], window[overlap]):
            raise ValueError(
                f"{source} is not a stride-{stride} sliding-window dataset: "
                f"sample {index} disagrees with an earlier overlap."
            )
        series[start:end] = window
        filled[start:end] = True
    return series


def _load_mask_csv(path: Path, expected_rows: int, cells: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask CSV not found: {path}")
    array = np.loadtxt(path, delimiter=",", dtype=np.float32)
    array = np.atleast_2d(array)
    if array.shape[0] != expected_rows or array.shape[1] != cells:
        raise ValueError(
            f"{path} has shape {array.shape}; expected ({expected_rows}, {cells})."
        )
    if not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{path} must contain only 0/1 masks.")
    return array


def _flatten_windows(windows: np.ndarray, selected_channels: list[int]) -> np.ndarray:
    """[B,C,T,H,W] -> [B,T,C*H*W], preserving every supplied sample."""
    selected = windows[:, selected_channels]
    return selected.transpose(0, 2, 1, 3, 4).reshape(selected.shape[0], selected.shape[2], -1)


def _grid_mask(kind: str, shape: tuple[int, int, int, int], rate: float, rng: np.random.Generator) -> np.ndarray:
    """Create traffic-benchmark masks on a grid, shared by all channels."""
    time, channels, height, width = shape
    cells = height * width
    if kind == "SR-TR":
        spatial = rng.random((time, height, width)) >= rate
    elif kind == "SR-TC":
        patch = max(1, min(16, time))
        blocks = rng.random((int(np.ceil(time / patch)), height, width)) >= rate
        spatial = np.repeat(blocks, patch, axis=0)[:time]
    elif kind in {"SC-TR", "SC-TC"}:
        # Select a contiguous row-major block.  It is a true spatial block and
        # its exact realised rate is written to the manifest.
        block = max(1, int(round(cells * rate)))
        starts = rng.integers(0, cells, size=time if kind == "SC-TR" else max(1, int(np.ceil(time / 16))))
        spatial = np.ones((time, cells), dtype=np.float32)
        for idx, start in enumerate(starts):
            frame_indices = [idx] if kind == "SC-TR" else range(idx * 16, min((idx + 1) * 16, time))
            missing = (np.arange(block) + int(start)) % cells
            for frame in frame_indices:
                spatial[frame, missing] = 0.0
        spatial = spatial.reshape(time, height, width)
    else:
        raise ValueError(f"Unsupported generated mask type: {kind}")
    return np.broadcast_to(spatial[:, None], (time, channels, height, width)).astype(np.float32).copy()


def _write_grid_edges(path: Path, channels: int, height: int, width: int, selected_channels: list[int]) -> None:
    """Write a directed 4-neighbour grid graph plus same-cell channel links."""
    nodes_per_channel = height * width
    selected = {channel: index for index, channel in enumerate(selected_channels)}
    lines = ["from,to,distance\n"]
    for old_channel, new_channel in selected.items():
        offset = new_channel * nodes_per_channel
        for row in range(height):
            for col in range(width):
                node = offset + row * width + col
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        lines.append(f"{node},{offset + nr * width + nc},1.0\n")
                for other_old, other_new in selected.items():
                    if other_old != old_channel:
                        lines.append(f"{node},{other_new * nodes_per_channel + row * width + col},1.0\n")
    path.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(PROFILES), required=True)
    parser.add_argument("--train-npz")
    parser.add_argument("--val-npz")
    parser.add_argument("--test-npz")
    parser.add_argument("--mask-root", help="Root containing fixed_mask/random_mask directories.")
    parser.add_argument("--output-root", default="imputation_benchmark/data/adapted")
    parser.add_argument("--mask", choices=("fixed", "random", "SR-TR", "SR-TC", "SC-TR", "SC-TC"), required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--channel", default="all", help="all (default) or one zero-based channel index.")
    parser.add_argument("--stride", type=int, default=1, help="Recorded in metadata for source reproducibility.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-windows-per-split", type=int, default=None,
        help="Smoke-test only: retain at most this many original windows in each split.",
    )
    parser.add_argument(
        "--legacy-stream",
        action="store_true",
        help="Also write duplicated [time,nodes,1] true/miss arrays for legacy non-split-aware loaders.",
    )
    parser.add_argument("--compress", action="store_true", help="Use compressed NPZ output (slower and memory-intensive).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.rate < 1.0:
        raise ValueError("--rate must be between 0 and 1.")
    profile = PROFILES[args.dataset]
    paths = {
        "train": Path(args.train_npz or profile.train),
        "val": Path(args.val_npz or profile.val),
        "test": Path(args.test_npz or profile.test),
    }
    windows = {name: _load_windows(path) for name, path in paths.items()}
    if args.max_windows_per_split is not None:
        if args.max_windows_per_split < 1:
            raise ValueError("--max-windows-per-split must be positive.")
        windows = {name: array[:args.max_windows_per_split] for name, array in windows.items()}
    first_shape = windows["train"].shape[1:]
    if any(array.shape[1:] != first_shape for array in windows.values()):
        raise ValueError("train/val/test NPZ shapes must agree except for sample count.")
    channels, window_length, height, width = first_shape
    if args.channel == "all":
        selected_channels = list(range(channels))
        channel_label = "all"
    else:
        channel = int(args.channel)
        if not 0 <= channel < channels:
            raise ValueError(f"--channel must be all or in [0, {channels - 1}].")
        selected_channels = [channel]
        channel_label = str(channel)

    # Preserve the project split exactly.  TaxiBJ and BikeNYC windows are not
    # chronologically ordered, so reconstructing a single time series from
    # overlapping samples would silently corrupt the data.  The 3-D `data`
    # array below is a compatibility stream; the split_* arrays are canonical
    # and must be used by adapters that accept batched windows (such as CSDI).
    split_true = {name: _flatten_windows(windows[name], selected_channels) for name in windows}
    split_lengths = {name: array.shape[0] * array.shape[1] for name, array in split_true.items()}
    stream_true = np.concatenate([split_true[name] for name in ("train", "val", "test")], axis=0)
    stream_true = stream_true.reshape(-1, stream_true.shape[-1])
    series = stream_true.reshape(stream_true.shape[0], len(selected_channels), height, width)
    source_samples = {name: int(windows[name].shape[0]) for name in windows}
    # [time,C,H,W] -> [time,C*H*W,1]; feature dimension stays one so legacy
    # baselines that use data[:, :, 0] retain every original channel.
    true_data = stream_true[..., None]

    mask_note: str | None = None
    if args.mask in {"fixed", "random"}:
        mask_root = Path(args.mask_root or profile.mask_root)
        cells = height * width
        csvs = [
            _load_mask_csv(
                mask_root / f"{args.mask}_mask" / str(args.rate) / f"{name}.csv",
                1 if args.mask == "fixed" else windows[name].shape[0],
                cells,
            )
            for name in ("train", "val", "test")
        ]
        if args.mask == "fixed":
            if not all(np.array_equal(csv[0], csvs[0][0]) for csv in csvs):
                raise ValueError("Fixed-mask CSVs differ across splits; cannot represent them as one global traffic mask.")
            spatial = np.broadcast_to(csvs[0][0], (series.shape[0], cells)).reshape(series.shape[0], height, width)
        else:
            spatial = np.concatenate(
                [np.repeat(csv, window_length, axis=0) for csv in csvs], axis=0
            ).reshape(series.shape[0], height, width)
            mask_note = "Each source window mask is repeated over that window's time steps; split_* arrays preserve it exactly."
        mask_grid = np.broadcast_to(spatial[:, None], series.shape).astype(np.float32).copy()
    else:
        mask_grid = _grid_mask(args.mask, series.shape, args.rate, np.random.default_rng(args.seed))

    mask = mask_grid.reshape(series.shape[0], -1, 1)
    if mask.shape != true_data.shape:
        raise AssertionError(f"mask shape {mask.shape} != data shape {true_data.shape}")

    output = Path(args.output_root) / args.dataset / f"{args.mask}_{args.rate}" / f"channel_{channel_label}"
    output.mkdir(parents=True, exist_ok=True)
    true_path = output / f"true_data_{args.mask}_{args.rate}_v2.npz"
    miss_path = output / f"miss_data_{args.mask}_{args.rate}_v2.npz"
    # Store canonical data exactly once.  The matching miss file is a compact
    # format marker; split-aware loaders derive zero-filled observations as
    # `data * mask` without materialising a second several-hundred-MB array.
    true_payload = {"adapter_format": np.array("grid_windows_v1")}
    miss_payload = {"adapter_format": np.array("grid_windows_v1")}
    cursor = 0
    for name in ("train", "val", "test"):
        length = split_lengths[name]
        split_mask = mask[cursor : cursor + length, :, 0].reshape(split_true[name].shape)
        true_payload[f"{name}_data"] = split_true[name].astype(np.float32)
        true_payload[f"{name}_mask"] = split_mask.astype(np.float32)
        cursor += length
    saver = np.savez_compressed if args.compress else np.savez
    if args.legacy_stream:
        # Legacy implementations expect exactly these filenames and a traffic
        # stream [T,N,1].  A separate output root keeps this compatibility view
        # from being confused with the split-aware CSDI/GAIN/etc. view.
        saver(true_path, data=true_data.astype(np.float32), mask=mask.astype(np.float32))
        saver(miss_path, data=(true_data * mask).astype(np.float32), mask=mask.astype(np.float32))
        saver(output / f"window_true_data_{args.mask}_{args.rate}_v2.npz", **true_payload)
        saver(output / f"window_miss_data_{args.mask}_{args.rate}_v2.npz", **miss_payload)
    else:
        saver(true_path, **true_payload)
        saver(miss_path, **miss_payload)
    _write_grid_edges(output / "grid_edges.csv", channels, height, width, selected_channels)

    total_samples = sum(source_samples.values())
    manifest = {
        "dataset": args.dataset,
        "source_npz": {name: str(path) for name, path in paths.items()},
        "source_window_shape": list(windows["train"].shape[1:]),
        "source_samples": source_samples,
        "stride": args.stride,
        "max_windows_per_split": args.max_windows_per_split,
        "baseline_data_shape": list(true_data.shape),
        "canonical_window_layout": "split_* arrays are [samples, time, nodes] and preserve the project's original split.",
        "layout": "data is [time, nodes, features] where features=1 and nodes=selected_channels*H*W",
        "selected_channels": selected_channels,
        "nodes": int(true_data.shape[1]),
        "feature_dim": 1,
        "grid": {"height": height, "width": width},
        "mask": args.mask,
        "requested_missing_rate": args.rate,
        "actual_missing_rate": float(1.0 - mask.mean()),
        "seed": args.seed,
        "split_ratios_for_baselines": {
            "train": split_lengths["train"] / sum(split_lengths.values()),
            "val": split_lengths["val"] / sum(split_lengths.values()),
            "test": split_lengths["test"] / sum(split_lengths.values()),
        },
        "files": {
            "true": true_path.name,
            "missing": miss_path.name,
            "grid_edges": "grid_edges.csv",
            "window_true": f"window_true_data_{args.mask}_{args.rate}_v2.npz" if args.legacy_stream else None,
            "window_missing": f"window_miss_data_{args.mask}_{args.rate}_v2.npz" if args.legacy_stream else None,
        },
        "note": mask_note or "The compatibility stream concatenates complete source windows; use split_* arrays to avoid cross-window samples.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"  canonical split data: [samples, {window_length}, {true_data.shape[1]}]; missing rate: {manifest['actual_missing_rate']:.6f}")
    print(f"  baseline split ratios: {manifest['split_ratios_for_baselines']}")


if __name__ == "__main__":
    main()
