from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from stmoe_imputer.data.npz_dataset import FlowNPZDataset


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v24/generate_structured_masks.py"
SPEC = importlib.util.spec_from_file_location("v24_generate_structured_masks", SCRIPT)
masks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(masks)


class V24StructuredMasksTest(unittest.TestCase):
    def assert_spatial_connected(self, missing: np.ndarray) -> None:
        locations = set(map(tuple, np.argwhere(missing)))
        if not locations:
            return
        reached = {next(iter(locations))}
        pending = list(reached)
        while pending:
            row, column = pending.pop()
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if neighbor in locations and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        self.assertEqual(reached, locations)

    def assert_cuboid(self, mask: np.ndarray) -> None:
        missing = np.argwhere(mask == 0)
        if not len(missing):
            return
        low, high = missing.min(axis=0), missing.max(axis=0) + 1
        expected = np.ones_like(mask)
        expected[tuple(slice(start, stop) for start, stop in zip(low, high))] = 0
        np.testing.assert_array_equal(mask, expected)

    def write_sources(self, directory: Path, shape=(6, 2, 5, 3, 4), channels_last=False, **extras):
        sources = {}
        target = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        if channels_last:
            target = target.transpose(0, 2, 3, 4, 1)
        for split in masks.SPLITS:
            path = directory / f"{split}.npz"
            np.savez_compressed(path, x_f_gt=target, **extras)
            sources[split] = path
        return sources

    def test_random_points_and_node_outages_meet_exact_rounded_counts(self) -> None:
        shape = (7, 4, 5)
        for rate in (0.0, 0.013, 0.37, 0.6, 1.0):
            for pattern in ("random_point", "node_contiguous"):
                with self.subTest(rate=rate, pattern=pattern):
                    mask = masks.make_mask(shape, rate, pattern, np.random.default_rng(41))
                    self.assertEqual(int((mask == 0).sum()), round(rate * np.prod(shape)))
                    self.assertEqual(mask.dtype, np.uint8)
                    if pattern == "node_contiguous":
                        durations = (mask == 0).sum(axis=0)
                        partial = np.argwhere((durations > 0) & (durations < shape[0]))
                        self.assertLessEqual(len(partial), 1)
                        for row, column in partial:
                            times = np.flatnonzero(mask[:, row, column] == 0)
                            np.testing.assert_array_equal(times, np.arange(times[0], times[-1] + 1))

    def test_spatial_regions_are_compact_connected_and_synchronous(self) -> None:
        shape = (5, 7, 9)
        for seed in range(8):
            for rate in (0.0, 0.17, 0.4, 0.93, 1.0):
                with self.subTest(seed=seed, rate=rate):
                    mask = masks.make_mask(shape, rate, "spatial_region", np.random.default_rng(seed))
                    np.testing.assert_array_equal(mask, np.broadcast_to(mask[0], shape))
                    missing = mask[0] == 0
                    self.assertEqual(int(missing.sum()), round(rate * shape[1] * shape[2]))
                    self.assert_spatial_connected(missing)
                    # A nearest-center region is hole-free: no observed point
                    # has missing neighbors on all four sides.
                    holes = (~missing[1:-1, 1:-1] & missing[:-2, 1:-1] & missing[2:, 1:-1]
                             & missing[1:-1, :-2] & missing[1:-1, 2:])
                    self.assertFalse(holes.any())

    def test_spacetime_blocks_have_globally_closest_achievable_volume(self) -> None:
        shape = (5, 7, 9)
        volumes = {0, *(t * h * w for t in range(1, 6) for h in range(1, 8) for w in range(1, 10))}
        for rate in (0.0, 0.001, 0.017, 0.37, 0.6, 0.997, 1.0):
            with self.subTest(rate=rate):
                mask = masks.make_mask(shape, rate, "spatiotemporal_block", np.random.default_rng(43))
                self.assert_cuboid(mask)
                target = rate * np.prod(shape)
                actual = int((mask == 0).sum())
                self.assertAlmostEqual(abs(actual - target), min(abs(volume - target) for volume in volumes))

    def test_header_inspection_does_not_read_array_values_and_checks_loader_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "header_only.npz"
            header = io.BytesIO()
            # Deliberately omit the enormous array payload. Reading values would
            # fail, whereas header inspection remains bounded and succeeds.
            np.lib.format.write_array_header_1_0(header, {
                "shape": (1_000_000, 2, 12, 128, 128), "fortran_order": False, "descr": "<f4",
            })
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("x_f_gt.npy", header.getvalue())
            actual = masks.inspect_npz(path)
            self.assertEqual(actual["shape_ncthw"], [1_000_000, 2, 12, 128, 128])
            self.assertEqual(actual["source_layout"], "NCTHW")
            with self.assertRaisesRegex(ValueError, "dataset loader infers NCTHW"):
                masks.inspect_npz(path, "NTHWC")

    def test_protocols_are_seeded_replicable_split_independent_and_fully_described(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sources = self.write_sources(directory, shape=(10, 2, 5, 3, 4))
            first = masks.generate_protocols(sources, directory / "first", rates=[0.2, 0.4], seed=2026)
            masks.generate_protocols(sources, directory / "repeat", patterns=list(reversed(masks.PATTERNS)), rates=[0.4, 0.2], seed=2026)
            self.assertEqual(len(first), 10)
            for path in first:
                metadata = json.loads(path.read_text())
                self.assertEqual(metadata["schema_version"], 1)
                self.assertEqual(metadata["mask_seed"], 2026)
                self.assertEqual(metadata["loader_pattern"], "random")
                for split, record in metadata["splits"].items():
                    csv = Path(record["csv"])
                    repeat = directory / "repeat" / csv.relative_to(directory / "first")
                    self.assertEqual(csv.read_bytes(), repeat.read_bytes())
                    self.assertEqual(record["sha256"], hashlib.sha256(csv.read_bytes()).hexdigest())
                    array = np.loadtxt(csv, delimiter=",", ndmin=2)
                    self.assertEqual(array.shape, (10, 5 * 3 * 4))
                    actual = 1 - array.mean(axis=1)
                    for summary, function in (("min", np.min), ("mean", np.mean), ("max", np.max)):
                        self.assertAlmostEqual(record[f"actual_missing_rate_{summary}"], float(function(actual)))
                    self.assertEqual(record["shape_ncthw"], [10, 2, 5, 3, 4])
                    self.assertEqual(record["source_npz"], str(sources[split].resolve()))
                    if metadata["pattern"] == "mixed":
                        components = record["components"]
                        self.assertEqual(len(components), 10)
                        counts = Counter(components)
                        self.assertEqual(set(counts), set(masks.BASE_PATTERNS))
                        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                        for row, component in zip(array.reshape(10, 5, 3, 4), components):
                            if component == "spatial_region":
                                np.testing.assert_array_equal(row, np.broadcast_to(row[0], row.shape))
                                self.assert_spatial_connected(row[0] == 0)
                            elif component == "spatiotemporal_block":
                                self.assert_cuboid(row)
                if metadata["pattern"] == "random_point":
                    self.assertEqual(len({entry["sha256"] for entry in metadata["splits"].values()}), 3)

    def test_csv_loader_retains_time_axis_for_both_supported_source_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for channels_last in (False, True):
                with self.subTest(channels_last=channels_last):
                    directory = root / str(channels_last)
                    directory.mkdir()
                    sources = self.write_sources(directory, channels_last=channels_last)
                    metadata_path, = masks.generate_protocols(
                        sources, directory / "masks", patterns=["random_point"], rates=[0.4],
                        layout="NTHWC" if channels_last else "NCTHW",
                    )
                    record = json.loads(metadata_path.read_text())["splits"]["train"]
                    dataset = FlowNPZDataset(sources["train"], mask_csv=record["csv"], mask_cfg={"pattern": "random"}, multiscale=False)
                    expected = torch.from_numpy(np.loadtxt(record["csv"], delimiter=",", dtype=np.float32)[0].reshape(1, 5, 3, 4))
                    actual = dataset[0]["m_f"]
                    self.assertEqual(actual.shape, (2, 5, 3, 4))
                    torch.testing.assert_close(actual, expected.expand_as(actual))
                    self.assertFalse(torch.equal(actual[:, 0], actual[:, 1]))

    def test_rejects_embedded_masks_and_overwrites_before_creating_other_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for key in ("m_f", "target_mask"):
                sources = self.write_sources(directory, **{key: np.ones((6, 1, 5, 3, 4), dtype=np.uint8)})
                with self.assertRaisesRegex(ValueError, key):
                    masks.generate_protocols(sources, directory / "rejected", patterns=["random_point"], rates=[0.4])
                self.assertFalse((directory / "rejected").exists())
            sources = self.write_sources(directory, available_mask=np.ones((6, 1, 5, 3, 4), dtype=np.uint8))
            metadata_path, = masks.generate_protocols(sources, directory / "accepted", patterns=["random_point"], rates=[0.4])
            metadata = json.loads(metadata_path.read_text())
            self.assertTrue(metadata["splits"]["train"]["has_available_mask"])
            self.assertIn("before intersection", metadata["rate_scope"])
            original = metadata_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                masks.generate_protocols(sources, directory / "accepted", patterns=["node_contiguous", "random_point"], rates=[0.4])
            self.assertFalse((directory / "accepted/node_contiguous").exists())
            self.assertEqual(metadata_path.read_bytes(), original)

    def test_cli_generates_only_requested_protocols_and_rejects_rate_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sources = self.write_sources(directory, shape=(2, 2, 3, 3, 4))
            arguments = [part for split, path in sources.items() for part in (f"--{split}-npz", str(path))]
            output = io.StringIO()
            with redirect_stdout(output):
                masks.main([*arguments, "--output-dir", str(directory / "cli"), "--patterns", "mixed", "--rates", "0.4", "--seed", "17"])
            self.assertEqual(len(list((directory / "cli").glob("*/*/metadata.json"))), 1)
            metadata = json.loads((directory / "cli/mixed/0.4/metadata.json").read_text())
            self.assertEqual(metadata["mask_seed"], 17)
            with self.assertRaisesRegex(ValueError, "distinct directory names"):
                masks.generate_protocols(sources, directory / "collision", rates=[0.2, 0.20000001])
            with self.assertRaisesRegex(ValueError, "finite"):
                masks.generate_protocols(sources, directory / "invalid", rates=[float("nan")])


if __name__ == "__main__":
    unittest.main()
