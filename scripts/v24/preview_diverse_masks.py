#!/usr/bin/env python3
"""CPU-only mask audit and preview; does not instantiate or train a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from stmoe_imputer.data.diverse_masks import FAMILIES,DiverseMaskSchedule,make_diverse_mask
from generate_structured_masks import inspect_npz


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-npz',required=True)
    p.add_argument('--variant',default=str(ROOT/'configs/v24/experiments/abc_d.json'))
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.variant).read_text())['data']['train_mask_diversity']
    header=inspect_npz(a.train_npz);n,_,t,h,w=header['shape_ncthw'];shape=(t,h,w)
    schedule=DiverseMaskSchedule(n,config);records=[];previous=None;start=time.perf_counter()
    for epoch in [1,2]:
        schedule.set_epoch(epoch);hashes=[];rates=[];families=[]
        for i in range(n):
            mask,family=schedule.sample(i,shape)
            hashes.append(hashlib.sha256(mask.tobytes()).hexdigest());rates.append(float(1-mask.mean()));families.append(family)
        records.append({'epoch':epoch,'family_counts':{name:families.count(i) for i,name in enumerate(FAMILIES)},
                        'unique_masks':len(set(hashes)),'mean_missing_rate':float(np.mean(rates)),
                        'min_missing_rate':min(rates),'max_missing_rate':max(rates),
                        'changed_vs_previous':None if previous is None else sum(x!=y for x,y in zip(hashes,previous))})
        previous=hashes
    # Saved example mask arrays allow inspection without relying on a picture.
    examples=np.stack([make_diverse_mask(shape,.4,f,np.random.default_rng(config['seed']+i)) for i,f in enumerate(FAMILIES)])
    np.savez_compressed(out/'examples.npz',observed_mask=examples,families=np.array(FAMILIES))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frames=sorted(set([0,t//3,2*t//3,t-1]));fig,axes=plt.subplots(len(FAMILIES),len(frames)+1,figsize=(12,18),squeeze=False)
    for i,name in enumerate(FAMILIES):
        for j,frame in enumerate(frames):
            axes[i,j].imshow(1-examples[i,frame],cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
            if i==0:axes[i,j].set_title(f't = {frame}')
        axes[i,-1].plot(np.arange(t),(1-examples[i]).mean((1,2)));axes[i,-1].set_ylim(0,1)
        if i==0:axes[i,-1].set_title('Missing fraction / time')
        axes[i,0].set_ylabel(name)
        for ax in axes[i,:-1]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('D training masks: black = missing; each window ~40% missing',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'mask_preview.png',dpi=130);plt.close(fig)
    result={'source_header':header,'policy':config,'epochs':records,'audit_seconds':time.perf_counter()-start,
            'note':'No training performed; masks depend only on seed, epoch, window index and geometry.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
