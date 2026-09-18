#!/usr/bin/env python3
"""Rebuild provenance from raw frames; filter discontinuous/leaking windows.

Never overwrites inputs or an existing clean dataset. The original mask row
remains paired with every retained window. Training-only statistics are saved
with the split/frame IDs and full source hashes for independent verification.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def frame_hash(frame):
    return hashlib.sha256(np.ascontiguousarray(frame, dtype=np.float32).tobytes()).digest()


def retained_windows(frame_ids, times_ns, previous_split_frames):
    retained, rejected = [], []
    for index, ids in enumerate(frame_ids):
        if not np.all(np.diff(times_ns[ids]) == 1800 * 10**9):
            rejected.append({'index': index, 'reason': 'time_discontinuity'})
        elif previous_split_frames.intersection(ids.tolist()):
            rejected.append({'index': index, 'reason': 'cross_split_overlap'})
        else:
            retained.append(index)
    return np.asarray(retained, dtype=np.int64), rejected


def prepare(data_dir, mask_dir, output):
    data_dir, mask_dir, output = map(lambda p: Path(p).resolve(), (data_dir, mask_dir, output))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    lookup, timestamps, hashes = collections.defaultdict(list), [], {}
    for year in (2013, 2014, 2015, 2016):
        path = data_dir / f'TAXIBJ{year}.grid'
        hashes[str(path)] = sha(path)
        frame = pd.read_csv(path, usecols=['time', 'row_id', 'column_id', 'inflow', 'outflow'])
        times = sorted(frame.time.unique())
        rows = frame.row_id.to_numpy(dtype=np.int64)
        cols = frame.column_id.to_numpy(dtype=np.int64)
        if np.any((rows < 0) | (rows >= 32) | (cols < 0) | (cols >= 32)):
            raise ValueError('Raw grid coordinates out of range')
        ti = frame.time.map({t:i for i,t in enumerate(times)}).to_numpy()
        dense = np.zeros((len(times), 2, 32, 32), dtype=np.float32)
        counts = np.zeros((len(times), 32, 32), dtype=np.int32)
        np.add.at(counts, (ti, rows, cols), 1)
        if not np.all(counts == 1):
            raise ValueError('Raw grid has duplicate/missing cells; availability requires explicit handling')
        dense[ti, 0, rows, cols] = frame.inflow.to_numpy()
        dense[ti, 1, rows, cols] = frame.outflow.to_numpy()
        if not np.isfinite(dense).all():
            raise ValueError('Raw source contains nonfinite values')
        for i, values in enumerate(dense):
            lookup[frame_hash(values)].append(len(timestamps) + i)
        timestamps.extend(times)
        del frame, dense, counts
    times_ns = pd.to_datetime(timestamps, utc=True).asi8
    if not np.all(np.diff(times_ns) > 0):
        raise ValueError('Raw chronology must be strictly increasing')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.building_', dir=output.parent))
    old_metadata = json.loads((mask_dir / 'metadata.json').read_text())
    metadata = dict(old_metadata, splits={}, clean_protocol=True)
    source_frames, normalization = set(), None
    manifest = {'schema_version': 1, 'source_sha256': hashes, 'splits': {},
                'mask_seed': old_metadata['mask_seed'], 'policy': 'preserve split order; remove time gaps and cross-split overlapping windows',
                'raw_timestamps': timestamps}
    try:
        for split in ('train', 'val', 'test'):
            path = data_dir / f'taxibj_{split}.npz'; hashes[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                if z.files != ['x_f_gt']:
                    raise ValueError('Expected only x_f_gt in audited original NPZ')
                values = z['x_f_gt']
            ids = []
            for window in values:
                matches = [lookup[frame_hash(f)] for f in window.transpose(1, 0, 2, 3)]
                if any(len(m) != 1 for m in matches):
                    raise ValueError('Every NPZ frame must uniquely match a raw timestamp')
                ids.append([m[0] for m in matches])
            ids = np.asarray(ids, dtype=np.int64)
            selected, rejected = retained_windows(ids, times_ns, source_frames)
            clean = values[selected]
            frames = set(ids[selected].reshape(-1).tolist())
            assert not frames.intersection(source_frames)
            source_frames.update(frames)
            dest = staging / f'taxibj_{split}.npz'
            np.savez_compressed(dest, x_f_gt=clean)
            if split == 'train':
                normalization = {'mean': clean.mean((0,2,3,4), dtype=np.float64).tolist(),
                                 'std': clean.std((0,2,3,4), dtype=np.float64).tolist(),
                                 'fit_split': 'train', 'weighting': 'retained training windows'}
            original_csv = mask_dir / f'{split}.csv'; hashes[str(original_csv)] = sha(original_csv)
            mask = np.loadtxt(original_csv, delimiter=',', dtype=np.uint8, ndmin=2)
            if mask.shape != (len(values), int(np.prod(values.shape[2:]))) or not np.isin(mask,[0,1]).all():
                raise ValueError('Original mask dimensions/values do not match windows')
            mask = mask[selected]
            relative = Path('masks/random_point/0.4') / f'{split}.csv'
            target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(target, mask, delimiter=',', fmt='%d')
            rates = 1 - mask.mean(1)
            record = dict(old_metadata['splits'][split], source_npz=str(output / dest.name),
                          shape_ncthw=list(clean.shape), rows=len(clean), csv=str(output / relative), sha256=sha(target),
                          actual_missing_rate_min=float(rates.min()), actual_missing_rate_mean=float(rates.mean()), actual_missing_rate_max=float(rates.max()))
            metadata['splits'][split] = record
            manifest['splits'][split] = {'retained_indices': selected.tolist(), 'rejected': rejected,
                                        'frame_ids': ids[selected].tolist(), 'shape':list(clean.shape), 'npz_sha256':sha(dest)}
            del values, clean, mask
        hashes[str(mask_dir / 'metadata.json')] = sha(mask_dir / 'metadata.json')
        manifest['normalization'] = normalization
        manifest['cross_split_frame_intersections'] = 0
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (staging / 'masks/random_point/0.4/metadata.json').write_text(json.dumps(metadata, indent=2))
        for path, expected in hashes.items():
            if sha(path) != expected:
                raise RuntimeError(f'Source changed during preparation: {path}')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--mask-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=prepare(args.data_dir,args.mask_dir,args.output)
    print(json.dumps({'output':str(Path(args.output).resolve()),'samples':{s:m['shape'][0] for s,m in result['splits'].items()},'normalization':result['normalization']},indent=2))

if __name__ == '__main__':
    main()
