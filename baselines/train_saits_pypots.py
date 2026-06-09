import argparse,json
from pathlib import Path
import numpy as np
from v2.baselines.st_impute.data.dataset import build_datasets_from_config

def ds_to_arr(ds):
    xs=[]; gts=[]; ms=[]
    for i in range(len(ds)):
        it=ds[i]; x=it['x_f_obs'].numpy(); gt=it['x_f_gt'].numpy(); m=it['m_f'].numpy(); C,T,H,W=gt.shape
        xsq=np.transpose(x,(1,0,2,3)).reshape(T,C*H*W); gtq=np.transpose(gt,(1,0,2,3)).reshape(T,C*H*W); msq=np.broadcast_to(np.transpose(m,(1,0,2,3)),(T,C,H,W)).reshape(T,C*H*W)
        xsq=xsq.astype('float32'); xsq[msq<.5]=np.nan; xs.append(xsq); gts.append(gtq.astype('float32')); ms.append(msq.astype('float32'))
    return np.stack(xs),np.stack(gts),np.stack(ms)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    try: from pypots.imputation import SAITS
    except ImportError as e: raise ImportError('Install PyPOTS first: pip install pypots') from e
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8')); tr,va,te,meta=build_datasets_from_config(cfg); Xtr,_,_=ds_to_arr(tr); Xv,_,_=ds_to_arr(va); Xte,Gte,Mte=ds_to_arr(te)
    model=SAITS(n_steps=Xtr.shape[1],n_features=Xtr.shape[2],n_layers=2,d_model=256,d_ffn=128,n_heads=4,d_k=64,d_v=64,dropout=0.1,epochs=int(cfg.get('epochs',50)),batch_size=int(cfg.get('batch_size',16)),saving_path=str(Path(cfg.get('save_dir','./runs'))/'saits'))
    model.fit({'X':Xtr},val_set={'X':Xv}); imp=model.impute({'X':Xte}); miss=1-Mte; mae=np.abs((imp-Gte)*miss).sum()/(miss.sum()+1e-6); rmse=np.sqrt((((imp-Gte)**2)*miss).sum()/(miss.sum()+1e-6)); mape=(np.abs((imp-Gte)/(np.abs(Gte)+1e-3))*miss).sum()/(miss.sum()+1e-6); print({'method':'SAITS/PyPOTS','mae':float(mae),'rmse':float(rmse),'mape':float(mape)})
if __name__=='__main__': main()
