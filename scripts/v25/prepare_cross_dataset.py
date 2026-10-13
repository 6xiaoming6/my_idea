#!/usr/bin/env python3
"""Immutable physical aggregates with cross-split exact-frame exclusion."""
import hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25')]
from prepare_data import sha
from stmoe_imputer.data.observed_multiscale import aggregate_sum

SPECS={
 'bikenyc':{'directory':'BikeNYC','prefix':'bikenyc','name':'BikeNYC','pooling':'sum'},
 'chap':{'directory':'CHAP/beijing','prefix':'chap_beijing','name':'CHAP_Beijing','pooling':'mean'},
}

def frame_ids(x):
    return [[hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()
             for f in row.transpose(1,0,2,3)] for row in x]

def prepare(dataset,project_root=ROOT):
    spec=SPECS[dataset];source=Path(project_root)/'data'/spec['directory']
    files={s:source/f'{spec["prefix"]}_{s}.npz' for s in ('train','val','test')}
    provenance={'schema':'v25_cross_dataset_v1','dataset':dataset,'pooling':spec['pooling'],
        'sources':{s:{'path':str(p.resolve()),'sha256':sha(p)} for s,p in files.items()},
        'split_policy':'Keep original order; exclude later-split windows containing any exact frame from earlier retained splits; no timeline reconstruction',
        'preprocessor_sha256':sha(Path(__file__))}
    if (source/'meta.txt').exists():provenance['original_meta']=(source/'meta.txt').read_text()
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:16]
    dest=source/'v25_observed_multiscale'/fingerprint;manifest=dest/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['provenance']!=provenance:raise ValueError('Provenance mismatch')
        for r in saved['outputs'].values():
            if sha(r['path'])!=r['sha256']:raise ValueError('Prepared data changed')
        return dest
    arrays={s:np.load(p,allow_pickle=False)['x_f_gt'] for s,p in files.items()}
    ids={s:frame_ids(x) for s,x in arrays.items()}
    original={s:set(itertools.chain.from_iterable(rows)) for s,rows in ids.items()}
    overlaps={f'{a}/{b}':len(original[a]&original[b]) for a,b in itertools.combinations(files,2)}
    dest.mkdir(parents=True,exist_ok=True);seen=set();outputs={}
    for s,x in arrays.items():
        if x.dtype!=np.float32 or x.ndim!=5 or not np.isfinite(x).all():raise ValueError('Invalid original data')
        retained=[i for i,row in enumerate(ids[s]) if not seen.intersection(row)]
        rejected=[i for i in range(len(x)) if i not in set(retained)]
        f=x[retained];current=set(itertools.chain.from_iterable(ids[s][i] for i in retained))
        assert not current.intersection(seen)
        seen.update(current)
        if not len(f):raise ValueError('Empty split after purging')
        m,c=aggregate_sum(f,2),aggregate_sum(f,4)
        if spec['pooling']=='mean':m=m/4;c=c/16
        nested=aggregate_sum(m,2)/(4 if spec['pooling']=='mean' else 1)
        np.testing.assert_allclose(nested,c,rtol=1e-6,atol=1e-5)
        path=dest/f'{spec["prefix"]}_{s}.npz'
        if path.exists():raise FileExistsError(path)
        with path.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,x_f_gt=f,x_m_gt=m,x_c_gt=c)
        path.with_suffix('.tmp').replace(path)
        outputs[s]={'path':str(path.resolve()),'sha256':sha(path),'shapes':{'F':list(f.shape),'M':list(m.shape),'C':list(c.shape)},
            'original_windows':len(x),'retained_indices':retained,'rejected_indices':rejected,'unique_frames':len(current),'nested_checked':True}
        print(dataset,s,outputs[s]['shapes'],'removed',rejected,flush=True)
    manifest.write_text(json.dumps({'provenance':provenance,'outputs':outputs,'original_frame_overlap':overlaps,
        'retained_cross_split_frame_overlap':0,
        'limitations':'Exact-content audit does not independently establish raw timestamp continuity or original NaN interpolation provenance.'},indent=2)+'\n')
    return dest

if __name__=='__main__':
    for ds in SPECS:print(prepare(ds))
