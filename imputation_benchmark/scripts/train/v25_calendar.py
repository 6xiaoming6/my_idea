"""Recover calendar covariates by audited frame IDs; never concatenate windows."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
    return h.hexdigest()

def prepare(root,output,sources):
    base=Path(root)/'data/TaxiBJ'
    clean=base/'v24_clean_abc_20260917/manifest.json'
    multi=base/'v25_observed_multiscale/73fe84df49ca05de/manifest.json'
    ext_path=base/'TAXIBJ.ext'
    m=json.loads(clean.read_text());v=json.loads(multi.read_text())
    assert sha(clean)==v['provenance']['source_split_manifest_sha256']
    ext=pd.read_csv(ext_path,usecols=['time'])
    stamps=pd.DatetimeIndex(pd.to_datetime(ext.time,utc=True))
    raw=pd.DatetimeIndex(pd.to_datetime(m['raw_timestamps'],utc=True))
    arrays={};audit=dict(source_sha256={str(p):sha(p) for p in [clean,multi,ext_path]},
        timezone='UTC as explicitly stored in original Z timestamps; no offset guessed',
        ext_rows=len(stamps),duplicate_timestamp_rows=int(stamps.duplicated().sum()),
        features=['day_of_month_minus_one','weekday_Monday_zero','hour','minute_div_15'],
        unused_fields=['Temperature','WindSpeed','Weather0..16'],splits={})
    for split,record in m['splits'].items():
        assert sha(sources[split])==v['outputs'][split]['sha256']
        source=v['provenance']['sources'][split]['path']
        assert sha(source)==record['npz_sha256']==v['provenance']['sources'][split]['sha256']
        ids=np.asarray(record['frame_ids']);wanted=raw[ids.ravel()]
        assert np.all(stamps.unique().get_indexer(wanted)>=0)
        assert np.all(np.diff(wanted.asi8.reshape(ids.shape),axis=1)==1800*10**9)
        with np.load(source) as a,np.load(sources[split]) as b:
            assert np.array_equal(a['x_f_gt'],b['x_f_gt'])
            assert b['x_f_gt'].shape[0]==ids.shape[0] and b['x_f_gt'].shape[2]==ids.shape[1]
        arrays[split]=np.stack([wanted.day-1,wanted.weekday,wanted.hour,wanted.minute//15],-1).reshape(*ids.shape,4).astype(np.int64)
        arrays[split+'_timestamp_ns']=wanted.asi8.reshape(ids.shape)
        audit['splits'][split]=dict(shape=list(arrays[split].shape),unmatched=0,F_identical=True,
            used_duplicate_times=len(wanted.intersection(stamps[stamps.duplicated(False)])),
            start=str(wanted.min()),end=str(wanted.max()))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'calendar.npz',**arrays)
    audit['calendar_sha256']=sha(output/'calendar.npz')
    (output/'calendar_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    return audit

class CalendarDataset(torch.utils.data.Dataset):
    def __init__(self,base,path,split):
        self.base=base
        with np.load(path) as z:self.calendar=z[split].copy()
        assert len(self.calendar)==len(base)
        self.diverse_masks=base.diverse_masks
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        row=self.base[i];row['calendar']=torch.from_numpy(self.calendar[i].copy());return row
    def set_epoch(self,epoch):self.base.set_epoch(epoch)
