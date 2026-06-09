import argparse,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from v2.baselines.st_impute.data.dataset import build_datasets_from_config
from v2.baselines.st_impute.utils.metrics import compute_metrics

def to_dev(b,d): return {k:v.to(d) if torch.is_tensor(v) else v for k,v in b.items()}
def mean_fill(b,mean):
    mean=torch.as_tensor(mean,device=b['x_f_obs'].device,dtype=b['x_f_obs'].dtype).unsqueeze(0); return b['x_f_obs']+(1-b['m_f'])*mean
def fit_hist(ds):
    s=c=None
    for i in tqdm(range(len(ds)),desc='fit historical',leave=False):
        x=ds[i]['x_f_gt'].numpy(); s=np.zeros_like(x,dtype='float64') if s is None else s; c=np.zeros_like(x,dtype='float64') if c is None else c; s+=x; c+=1
    return (s/np.maximum(c,1)).astype('float32')
def historical_fill(b,hist):
    hist=torch.as_tensor(hist,device=b['x_f_obs'].device,dtype=b['x_f_obs'].dtype).unsqueeze(0); return b['x_f_obs']+(1-b['m_f'])*hist
def interp1d(y,m):
    T=len(y); idx=np.arange(T); obs=m>0.5
    if obs.all(): return y
    if not obs.any(): return np.zeros_like(y)
    return np.interp(idx,idx[obs],y[obs]).astype('float32')
def linear_fill(b):
    x=b['x_f_obs'].detach().cpu().numpy(); m=b['m_f'].detach().cpu().numpy(); B,C,T,H,W=x.shape; out=np.empty_like(x)
    for bb in range(B):
      for c in range(C):
        for h in range(H):
          for w in range(W): out[bb,c,:,h,w]=interp1d(x[bb,c,:,h,w],m[bb,0,:,h,w])
    return torch.from_numpy(out).to(b['x_f_obs'].device,dtype=b['x_f_obs'].dtype)
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--method',required=True,choices=['mean','historical','linear']); ap.add_argument('--missing_type'); ap.add_argument('--missing_rate',type=float); args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.missing_type: cfg['missing_type']=args.missing_type
    if args.missing_rate is not None: cfg['missing_rate']=args.missing_rate
    dev=cfg.get('device','cuda'); dev='cpu' if dev=='cuda' and not torch.cuda.is_available() else dev; device=torch.device(dev)
    tr,va,te,meta=build_datasets_from_config(cfg); loader=DataLoader(te,batch_size=int(cfg.get('batch_size',16)),shuffle=False,num_workers=int(cfg.get('num_workers',0)))
    hist=fit_hist(tr) if args.method=='historical' else None; total={'mae':0,'rmse':0,'mape':0}; n=0
    for b in tqdm(loader,desc='eval'):
        b=to_dev(b,device); pred=mean_fill(b,meta['mean']) if args.method=='mean' else historical_fill(b,hist) if args.method=='historical' else linear_fill(b); met=compute_metrics(pred,b['x_f_gt'],b['m_f'])
        for k in total: total[k]+=met[k]
        n+=1
    res={'method':args.method,'missing_type':cfg.get('missing_type'),'missing_rate':cfg.get('missing_rate'),'test':{k:v/max(1,n) for k,v in total.items()}}
    print(json.dumps(res,indent=2))
if __name__=='__main__': main()
