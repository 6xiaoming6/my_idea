#!/usr/bin/env python3
"""Generate fixed, model-independent window masks for the v24 protocol.

Only NPZ array headers are read. CSV columns are flattened in T/H/W order;
the same geometric mask applies to every channel. Use data.mask.pattern=random
in the existing dataset loader, with the generated per-split CSV paths.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Sequence
import zipfile

import numpy as np


BASE_PATTERNS = ("random_point", "node_contiguous", "spatial_region", "spatiotemporal_block")
PATTERNS = (*BASE_PATTERNS, "mixed")
SPLITS = ("train", "val", "test")
DEFINITIONS = {
    "random_point": "Uniformly sample round(rate*T*H*W) distinct cells in each window.",
    "node_contiguous": "Random nodes lose the full window; at most one further node loses a contiguous time segment to meet the exact rounded cell count.",
    "spatial_region": "The round(rate*H*W) cells nearest a random spatial center form one compact connected region, repeated at every time step without wrapping.",
    "spatiotemporal_block": "One nonwrapped contiguous cuboid whose integer volume is nearest rate*T*H*W; ties favor balanced fractions of the three axis lengths, then smaller volume.",
    "mixed": "Assign the four base patterns to whole windows in a seeded shuffled order, with component counts differing by at most one; no within-window mixing.",
}


def inspect_npz(path: str | Path, layout: str = "auto") -> dict:
    """Infer exactly the layout used by data.transforms.to_bcthw, from headers."""
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for key in ("m_f", "target_mask"):
            if f"{key}.npy" in names:
                raise ValueError(f"{path} contains {key}; embedded masks override or alter the CSV protocol")
        key = "x_f_gt" if "x_f_gt.npy" in names else "x_f"
        member = f"{key}.npy"
        if member not in names:
            raise ValueError(f"{path} must contain x_f_gt or x_f")
        if names.count(member) != 1:
            raise ValueError(f"{path} has duplicate {member} entries")
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY header version {version} in {path}")
        has_available_mask = "available_mask.npy" in names
    if len(shape) != 5 or any(size < 1 for size in shape):
        raise ValueError(f"{path}: expected a nonempty 5D source array, got {shape}")
    if dtype.hasobject or not np.issubdtype(dtype, np.number):
        raise ValueError(f"{path}: source array must have a numeric non-object dtype")
    if shape[1] <= 8 and (shape[-1] > 8 or shape[1] <= shape[-1]):
        inferred = "NCTHW"
        ncthw = shape
    elif shape[-1] <= 8:
        inferred = "NTHWC"
        ncthw = (shape[0], shape[4], shape[1], shape[2], shape[3])
    else:
        raise ValueError(f"{path}: the dataset loader cannot infer layout from {shape}")
    if layout not in {"auto", "NCTHW", "NTHWC"}:
        raise ValueError("layout must be auto, NCTHW, or NTHWC")
    if layout != "auto" and layout != inferred:
        raise ValueError(
            f"{path}: requested {layout}, but the current dataset loader infers {inferred}; "
            "repack the source array into an unambiguous supported layout first"
        )
    return {
        "source_npz": str(path), "source_key": key, "source_layout": inferred,
        "shape_ncthw": list(map(int, ncthw)), "has_available_mask": has_available_mask,
    }


def split_rng(seed: int, pattern: str, rate: float, split: str) -> np.random.Generator:
    # Stable IDs and IEEE-754 words are independent of Python hash randomization,
    # CLI ordering, model training seeds, and output directory names.
    rate_words = struct.unpack(">II", struct.pack(">d", float(rate)))
    sequence = np.random.SeedSequence([seed, PATTERNS.index(pattern) + 1, SPLITS.index(split) + 1, *rate_words])
    return np.random.default_rng(sequence)


@lru_cache(maxsize=128)
def block_dimensions(shape: tuple[int, int, int], rate: float) -> tuple[int, int, int]:
    """Find the closest achievable cuboid volume without enumerating all voxels."""
    target = float(rate) * math.prod(shape)
    if target == 0:
        return (0, 0, 0)
    # Enumerate the two shortest axes; only floor/ceil are needed on the third.
    axes = sorted(range(3), key=lambda axis: shape[axis])
    left, middle, last = (shape[axis] for axis in axes)
    best = (0, 0, 0)
    best_score = (target, 0.0, 0, best)
    for first in range(1, left + 1):
        for second in range(1, middle + 1):
            ideal = target / (first * second)
            candidates = {max(1, min(last, math.floor(ideal))), max(1, min(last, math.ceil(ideal)))}
            for third in candidates:
                dims = [0, 0, 0]
                for axis, value in zip(axes, (first, second, third)):
                    dims[axis] = value
                dimensions = tuple(dims)
                volume = math.prod(dimensions)
                fractions = [dimensions[axis] / shape[axis] for axis in range(3)]
                score = (abs(volume - target), max(fractions) - min(fractions), volume, dimensions)
                if score < best_score:
                    best, best_score = dimensions, score
    return best


def make_mask(shape: tuple[int, int, int], rate: float, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """Return one uint8 observed mask [T,H,W]; allocation is one window only."""
    if pattern not in BASE_PATTERNS:
        raise ValueError(f"make_mask requires a base pattern, got {pattern}")
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("shape must contain positive T/H/W dimensions")
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be finite and between 0 and 1")
    t, h, w = shape
    count = int(round(rate * t * h * w))
    mask = np.ones(shape, dtype=np.uint8)
    if pattern == "random_point":
        mask.reshape(-1)[rng.choice(mask.size, size=count, replace=False)] = 0
    elif pattern == "node_contiguous":
        full_nodes, remainder = divmod(count, t)
        nodes = rng.choice(h * w, size=full_nodes + bool(remainder), replace=False)
        temporal_nodes = mask.reshape(t, h * w)
        temporal_nodes[:, nodes[:full_nodes]] = 0
        if remainder:
            start = int(rng.integers(0, t - remainder + 1))
            temporal_nodes[start:start + remainder, nodes[-1]] = 0
    elif pattern == "spatial_region":
        spatial_count = int(round(rate * h * w))
        center_h, center_w = int(rng.integers(h)), int(rng.integers(w))
        rows, columns = np.indices((h, w))
        distance = ((rows - center_h) ** 2 + (columns - center_w) ** 2).reshape(-1)
        # Every selected noncentral cell has an axis-adjacent cell of smaller
        # distance, ensuring connectivity even with randomized distance ties.
        order = np.lexsort((rng.random(h * w), distance))
        mask.reshape(t, h * w)[:, order[:spatial_count]] = 0
    else:
        dimensions = block_dimensions(tuple(shape), float(rate))
        if math.prod(dimensions):
            starts = [int(rng.integers(0, axis - length + 1)) for axis, length in zip(shape, dimensions)]
            mask[tuple(slice(start, start + length) for start, length in zip(starts, dimensions))] = 0
    return mask


def _csv_payload(mask: np.ndarray) -> bytes:
    # One-byte values with commas and a newline, without a large Python list.
    flat = mask.reshape(-1)
    encoded = np.empty(flat.size * 2, dtype=np.uint8)
    encoded[0::2] = flat + ord("0")
    encoded[1::2] = ord(",")
    encoded[-1] = ord("\n")
    return encoded.tobytes()


def generate_protocols(
    sources: dict[str, str | Path], output_dir: str | Path,
    patterns: Sequence[str] = PATTERNS, rates: Sequence[float] = (0.2, 0.4, 0.6),
    seed: int = 2026, layout: str = "auto",
) -> list[Path]:
    if set(sources) != set(SPLITS):
        raise ValueError("sources must provide train, val, and test NPZ files")
    if not patterns or len(set(patterns)) != len(patterns) or any(p not in PATTERNS for p in patterns):
        raise ValueError(f"patterns must be unique choices from {PATTERNS}")
    if not rates or any(not math.isfinite(rate) or not 0 <= rate <= 1 for rate in rates):
        raise ValueError("rates must be finite values between 0 and 1")
    rate_labels = [format(rate, "g") for rate in rates]
    if len(set(rate_labels)) != len(rate_labels):
        raise ValueError("rates must have distinct directory names under format(rate, 'g')")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    headers = {split: inspect_npz(sources[split], layout=layout) for split in SPLITS}
    output_dir = Path(output_dir).resolve()
    targets = [output_dir / pattern / label for pattern in patterns for label in rate_labels]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing mask protocol: {target}")
    metadata_paths = []
    for pattern in patterns:
        for rate, label in zip(rates, rate_labels):
            directory = output_dir / pattern / label
            directory.mkdir(parents=True, exist_ok=False)
            metadata = {
                "schema_version": 1, "pattern": pattern, "requested_missing_rate": float(rate),
                "mask_seed": seed, "definition": DEFINITIONS[pattern],
                "rounding": "Python round uses nearest integer with ties to even; cuboid chooses nearest achievable volume.",
                "csv_semantics": "0=missing, 1=observed; one row per window; columns flattened T/H/W in C order; mask shared across channels",
                "loader_pattern": "random",
                "rate_scope": "Raw geometric mask rates before intersection with source availability or finite target values; effective supervised Q rates must be reported separately.",
                "splits": {},
            }
            if pattern == "mixed":
                metadata["component_definitions"] = {key: DEFINITIONS[key] for key in BASE_PATTERNS}
            for split in SPLITS:
                header = headers[split]
                n, _, t, h, w = header["shape_ncthw"]
                rng = split_rng(seed, pattern, float(rate), split)
                components = None
                if pattern == "mixed":
                    indices = np.arange(n, dtype=np.int64) % len(BASE_PATTERNS)
                    rng.shuffle(indices)
                    components = [BASE_PATTERNS[index] for index in indices]
                path = directory / f"{split}.csv"
                digest = hashlib.sha256()
                rate_sum, rate_min, rate_max = 0.0, 1.0, 0.0
                with path.open("xb") as handle:
                    for row in range(n):
                        component = components[row] if components is not None else pattern
                        mask = make_mask((t, h, w), float(rate), component, rng)
                        actual = 1.0 - float(mask.sum(dtype=np.int64)) / mask.size
                        rate_sum += actual
                        rate_min, rate_max = min(rate_min, actual), max(rate_max, actual)
                        payload = _csv_payload(mask)
                        handle.write(payload)
                        digest.update(payload)
                split_metadata = {
                    **header, "rows": n, "columns": t * h * w, "csv": str(path),
                    "actual_missing_rate_min": rate_min, "actual_missing_rate_mean": rate_sum / n,
                    "actual_missing_rate_max": rate_max, "sha256": digest.hexdigest(),
                    "actual_minus_requested_missing_rate_min": rate_min - float(rate),
                    "actual_minus_requested_missing_rate_mean": rate_sum / n - float(rate),
                    "actual_minus_requested_missing_rate_max": rate_max - float(rate),
                }
                if components is not None:
                    split_metadata["components"] = components
                metadata["splits"][split] = split_metadata
            metadata_path = directory / "metadata.json"
            with metadata_path.open("x", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            metadata_paths.append(metadata_path)
    return metadata_paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f"--{split}-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS))
    parser.add_argument("--rates", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--layout", choices=("auto", "NCTHW", "NTHWC"), default="auto", help="Assert the source layout; a value incompatible with the dataset loader is rejected")
    args = parser.parse_args(argv)
    try:
        paths = generate_protocols(
            {split: getattr(args, f"{split}_npz") for split in SPLITS},
            args.output_dir, args.patterns, args.rates, args.seed, args.layout,
        )
    except (ValueError, FileExistsError, OSError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
