import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from v2.baselines.st_impute.data.dataset import build_datasets_from_config
from v2.baselines.st_impute.utils.seed import set_seed
from v2.baselines.st_impute.utils.metrics import masked_smooth_l1_loss, compute_metrics
from v2.baselines.st_impute.models.fine_only import FineOnlyConv3D
from v2.baselines.st_impute.models.conv3d_unet import Conv3DUNet
from v2.baselines.st_impute.models.ms_concat import MultiScaleConcatFusion
from v2.baselines.st_impute.models.fixed_scale_experts import FixedScaleExperts
from v2.baselines.st_impute.models.shared_experts_no_router import SharedExpertsNoRouter
from v2.baselines.st_impute.models.simple_transformer import SimpleTransformerImputer

def to_dev(b,d): return {k:v.to(d) if torch.is_tensor(v) else v for k,v in b.items()}
def build_model(name,C,H,W,T,hidden):
    if name=='fine_only': return FineOnlyConv3D(C,hidden)
    if name=='conv3d_unet': return Conv3DUNet(C,hidden)
    if name=='ms_concat': return MultiScaleConcatFusion(C,hidden)
    if name=='fixed_experts': return FixedScaleExperts(C,hidden)
    if name=='no_router': return SharedExpertsNoRouter(C,hidden,4)
    if name=='transformer': return SimpleTransformerImputer(C,H,W,T,max(128,hidden*2),4,2)
    raise ValueError(name)
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); total={'mae':0,'rmse':0,'mape':0}; n=0
    for b in loader:
        b=to_dev(b,device); out=model(b); met=compute_metrics(out['x_hat_final'],b['x_f_gt'],b['m_f'])
        for k in total: total[k]+=met[k]
        n+=1
    return {k:v/max(1,n) for k,v in total.items()}
def train_epoch(model,loader,opt,device):
    model.train(); s=0; n=0
    for b in tqdm(loader,desc='train',leave=False):
        b=to_dev(b,device); opt.zero_grad(); out=model(b); loss=masked_smooth_l1_loss(out['x_hat_final'],b['x_f_gt'],b['m_f']); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); s+=float(loss.detach().cpu()); n+=1
    return s/max(1,n)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--model',required=True,choices=['fine_only','conv3d_unet','transformer','ms_concat','fixed_experts','no_router']); ap.add_argument('--missing_type'); ap.add_argument('--missing_rate',type=float); args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.missing_type: cfg['missing_type']=args.missing_type
    if args.missing_rate is not None: cfg['missing_rate']=args.missing_rate
    set_seed(int(cfg.get('seed',42))); dev=cfg.get('device','cuda'); dev='cpu' if dev=='cuda' and not torch.cuda.is_available() else dev; device=torch.device(dev)
    tr,va,te,meta=build_datasets_from_config(cfg); sample=tr[0]; C,T,H,W=sample['x_f_gt'].shape
    mk=lambda ds,sh: DataLoader(ds,batch_size=int(cfg.get('batch_size',16)),shuffle=sh,num_workers=int(cfg.get('num_workers',0)),drop_last=sh)
    tl,vl,el=mk(tr,True),mk(va,False),mk(te,False)
    model=build_model(args.model,C,H,W,T,int(cfg.get('hidden_dim',64))).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(cfg.get('lr',1e-3)),weight_decay=float(cfg.get('weight_decay',1e-4)))
    save_dir=Path(cfg.get('save_dir','./runs'))/args.model; save_dir.mkdir(parents=True,exist_ok=True); best=1e18; best_path=save_dir/'best.pt'
    for ep in range(1,int(cfg.get('epochs',50))+1):
        loss=train_epoch(model,tl,opt,device); val=evaluate(model,vl,device); print(f'[{args.model}] epoch={ep:03d} train={loss:.6f} val_mae={val["mae"]:.6f} val_rmse={val["rmse"]:.6f}')
        if val['mae']<best: best=val['mae']; torch.save({'model':model.state_dict(),'config':cfg,'model_name':args.model,'meta':meta},best_path)
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['model']); test=evaluate(model,el,device); res={'model':args.model,'missing_type':cfg.get('missing_type'),'missing_rate':cfg.get('missing_rate'),'test':test}; (save_dir/'result.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
