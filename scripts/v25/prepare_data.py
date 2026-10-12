#!/usr/bin/env python3
"""Precompute complete physical M/C sums, before drawing any observation masks."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from stmoe_imputer.data.observed_multiscale import aggregate_sum


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(source, destination):
    source,destination=Path(source),Path(destination)
    files={s:source/f'taxibj_{s}.npz' for s in ('train','val','test')}
    provenance={'schema':'v25_physical_sum_v1','sources':{s:{'path':str(p.resolve()),'sha256':sha(p)} for s,p in files.items()},
                'aggregation':'per-variable per-time disjoint spatial sums; factors 2 and 4',
                'source_split_manifest_sha256':sha(source/'manifest.json') if (source/'manifest.json').exists() else None}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:16]
    root=destination/fingerprint;root.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['provenance']!=provenance:raise ValueError('Data provenance mismatch')
        for row in saved['outputs'].values():
            if sha(row['path'])!=row['sha256']:raise ValueError('Prepared dataset changed')
        return root
    outputs={}
    for split,path in files.items():
        dest=root/f'taxibj_{split}.npz'
        if dest.exists():raise FileExistsError(f'Uncommitted prepared file; inspect before retry: {dest}')
        with np.load(path,allow_pickle=False) as z:f=np.array(z['x_f_gt'],dtype=np.float32)
        if not np.isfinite(f).all():raise ValueError('Original missingness requires a separate aggregate-availability protocol')
        m,c=aggregate_sum(f,2),aggregate_sum(f,4)
        # Exact for integer counts; allow only float summation precision otherwise.
        np.testing.assert_allclose(aggregate_sum(m,2),c,rtol=1e-6,atol=1e-5)
        temp=dest.with_suffix('.tmp')
        with temp.open('wb') as out:np.savez_compressed(out,x_f_gt=f,x_m_gt=m,x_c_gt=c)
        temp.replace(dest)
        outputs[split]={'path':str(dest.resolve()),'sha256':sha(dest),'shapes':{'F':list(f.shape),'M':list(m.shape),'C':list(c.shape)},
                        'finite':True,'min':float(f.min()),'max':float(f.max()),'nested_sum_checked':True}
        print(f'Prepared {split}: {outputs[split]["shapes"]}',flush=True)
    manifest.write_text(json.dumps({'provenance':provenance,'outputs':outputs},indent=2)+'\n')
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'data/TaxiBJ/v24_clean_abc_20260917')
    p.add_argument('--destination',type=Path,default=ROOT/'data/TaxiBJ/v25_observed_multiscale')
    a=p.parse_args();print(prepare(a.source,a.destination))
