#!/usr/bin/env python3
"""Prepare physical M/C sums from already split, complete F windows (N,C,T,H,W)."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stmoe_imputer.data.observed_multiscale import aggregate_sum
from release_utils import write, sha256


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve(): raise ValueError('Use a separate destination')
    paths = {s: source / f'taxibj_{s}.npz' for s in ('train', 'val', 'test')}
    targets = {s: destination / path.name for s, path in paths.items()}
    if any(p.exists() for p in [*targets.values(), destination / 'manifest.json']):
        raise FileExistsError('Destination already contains prepared data; refusing to overwrite')
    # Validate all splits before writing anything. No re-splitting or timeline reconstruction.
    arrays = {}
    for s, path in paths.items():
        with np.load(path, allow_pickle=False) as z: f = np.array(z['x_f_gt'], dtype=np.float32)
        if f.ndim != 5 or f.shape[1:] != (2, 12, 32, 32):
            raise ValueError(f'{s}: I37 TaxiBJ expects (N,2,12,32,32), got {f.shape}')
        if not np.isfinite(f).all() or (f < 0).any():
            raise ValueError('Expected complete, finite, nonnegative physical flow counts')
        arrays[s] = f
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'aggregation': 'disjoint per-channel spatial sums; M=2x2, C=4x4', 'splits': {}}
    for s, f in arrays.items():
        m, c = aggregate_sum(f, 2), aggregate_sum(f, 4)
        np.testing.assert_allclose(aggregate_sum(m, 2), c, rtol=1e-6, atol=1e-5)
        tmp = targets[s].with_suffix('.tmp')
        with tmp.open('wb') as out: np.savez_compressed(out, x_f_gt=f, x_m_gt=m, x_c_gt=c)
        tmp.replace(targets[s])
        manifest['splits'][s] = {'source_sha256': sha256(paths[s]), 'sha256': sha256(targets[s]),
                                  'shapes': {k: list(v.shape) for k, v in [('F', f), ('M', m), ('C', c)]}}
    write(destination / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args(); prepare(a.source, a.destination)
