import json,hashlib,collections,gc
from pathlib import Path
import numpy as np
import pandas as pd
root=Path('data/TaxiBJ')
def digest(x): return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float32).tobytes()).hexdigest()
report={'splits':{},'raw_years':{},'masks':{}}
full=np.load(root/'taxibj_windows.npz')['x_f_gt']
full_hash=[digest(w) for w in full]
full_ids={h:i for i,h in enumerate(full_hash)}
frame_hashes=[[digest(f) for f in w.transpose(1,0,2,3)] for w in full]
frame_sets={}
for split in ['train','val','test']:
 with np.load(root/f'taxibj_{split}.npz') as z:
  x=z['x_f_gt'];keys=z.files
 ids=[full_ids.get(digest(w),-1) for w in x]
 frame_sets[split]={h for i in ids if i>=0 for h in frame_hashes[i]}
 report['splits'][split]={'keys':keys,'shape':list(x.shape),'nonfinite':int((~np.isfinite(x)).sum()),'negative':int((x<0).sum()),'zero_fraction':float((x==0).mean()),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max()),'channel_mean':x.mean(axis=(0,2,3,4)).tolist(),'channel_std':x.std(axis=(0,2,3,4)).tolist(),'window_mean_quantiles':np.quantile(x.mean(axis=(1,2,3,4)),[0,.1,.5,.9,1]).tolist(),'unique_windows':len(set(digest(w) for w in x)),'full_indices_first_last':[ids[0],ids[-1]],'ordered_contiguous':ids==list(range(ids[0],ids[0]+len(ids))),'full_indices_missing':ids.count(-1)}
 del x
report['cross_split_identical_frames']={f'{a}-{b}':len(frame_sets[a]&frame_sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
raw_lookup=collections.defaultdict(list);raw_index=0;raw_times=[]
for year in [2013,2014,2015,2016]:
 path=root/f'TAXIBJ{year}.grid'
 df=pd.read_csv(path,usecols=['time','row_id','column_id','inflow','outflow'],dtype={'row_id':'int16','column_id':'int16','inflow':'float32','outflow':'float32'})
 times=sorted(df.time.unique());mapping={t:i for i,t in enumerate(times)}
 ti=df.time.map(mapping).to_numpy();rr=df.row_id.to_numpy();cc=df.column_id.to_numpy()
 dense=np.zeros((len(times),2,32,32),dtype=np.float32)
 counts=np.zeros((len(times),32,32),dtype=np.int16)
 np.add.at(counts,(ti,rr,cc),1)
 dense[ti,0,rr,cc]=df.inflow.to_numpy();dense[ti,1,rr,cc]=df.outflow.to_numpy()
 dt=pd.to_datetime(times,utc=True);gaps=np.flatnonzero(np.diff(dt.asi8)!=1800*10**9)
 report['raw_years'][str(year)]={'frames':len(times),'start':times[0],'end':times[-1],'missing_grid_cells':int((counts==0).sum()),'duplicate_grid_cells':int((counts>1).sum()),'non_halfhour_gaps':len(gaps),'gap_examples':[[times[int(i)],times[int(i+1)]] for i in gaps[:5]]}
 for i,frame in enumerate(dense): raw_lookup[digest(frame)].append(raw_index+i)
 raw_index+=len(times);raw_times.extend(times)
 print('year',year,report['raw_years'][str(year)],flush=True)
 del df,dense,counts;gc.collect()
all_ids=[];unmapped=[];ambiguous=[];bad_windows=[]
for wi,hashes in enumerate(frame_hashes):
 ids=[]
 for h in hashes:
  matches=raw_lookup.get(h,[])
  if len(matches)!=1:
   (unmapped if not matches else ambiguous).append([wi,len(ids),len(matches)])
  ids.append(matches[0] if matches else -1)
 all_ids.append(ids)
 if -1 not in ids:
  dt=pd.to_datetime([raw_times[i] for i in ids],utc=True).asi8
  if not np.all(np.diff(dt)==1800*10**9): bad_windows.append({'window':wi,'raw_ids':ids,'times':[raw_times[i] for i in ids]})
starts=np.array([ids[0] for ids in all_ids])
report['raw_mapping']={'total_raw_frames':raw_index,'full_windows':len(full),'unmapped_frames':len(unmapped),'ambiguous_frames':len(ambiguous),'start_stride_counts':{str(k):v for k,v in collections.Counter(np.diff(starts).tolist()).items()},'nonconsecutive_frame_windows':sum(not np.all(np.diff(ids)==1) for ids in all_ids),'discontinuous_time_windows':len(bad_windows),'discontinuous_examples':bad_windows,'split_time_ranges':{s:[raw_times[all_ids[meta['full_indices_first_last'][0]][0]],raw_times[all_ids[meta['full_indices_first_last'][1]][-1]]] for s,meta in report['splits'].items()}}
maskroot=Path('outputs/v24-COE/section13/taxibj_masks_seed2026')
for pattern in ['random_point','node_contiguous','spatial_region','spatiotemporal_block','mixed']:
 out={}
 for split in ['train','val','test']:
  path=maskroot/pattern/'0.4'/f'{split}.csv'
  m=np.loadtxt(path,delimiter=',',dtype=np.uint8,ndmin=2)
  unique=np.unique(m);rates=1-m.mean(axis=1)
  out[split]={'shape':list(m.shape),'binary':bool(np.all(np.isin(unique,[0,1]))),'unique_windows':len(set(hashlib.sha256(row.tobytes()).digest() for row in m)),'missing_rate_min_mean_max':[float(rates.min()),float(rates.mean()),float(rates.max())]}
  del m
 report['masks'][pattern]=out
Path('/tmp/v24_taxibj_data_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
