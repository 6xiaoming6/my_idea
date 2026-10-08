"""Opt-in deterministic backend for the W experiments only."""
import hashlib
import os
from functools import lru_cache
import numpy as np
import torch


def configure(cfg):
    if not cfg.get('train',{}).get('strict_replay',False):return
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def pool(x,factor):
    b,c,t,h,w=x.shape
    if h%factor or w%factor:raise ValueError('Pool requires divisible spatial dimensions')
    return x.reshape(b,c,t,h//factor,factor,w//factor,factor).mean((4,6))


@lru_cache(maxsize=64)
def _weights(n,m):
    weights=torch.zeros(m,n,dtype=torch.float32)
    for i in range(m):
        x=(i+.5)*n/m-.5;lo=int(np.floor(x));fraction=x-lo
        weights[i,min(n-1,max(0,lo))]+=1-fraction
        weights[i,min(n-1,max(0,lo+1))]+=fraction
    return weights


def resize(x,size):
    if x.shape[-2:]==tuple(size):return x
    # FP32 accumulation, then restore input dtype, irrespective of outer autocast.
    with torch.autocast(device_type=x.device.type,enabled=False):
        wy=_weights(x.shape[-2],size[0]).to(x.device)
        wx=_weights(x.shape[-1],size[1]).to(x.device)
        y=torch.matmul(torch.matmul(wy,x.float()),wx.T)
    return y.to(x.dtype)


def state_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for a in x:visit(a)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def common_initialization(model, job):
    """Freeze W01's complete common state, then copy that exact template to every W job."""
    from pathlib import Path
    path = job.get('common_initialization')
    if not path:return
    path = Path(path)
    common = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()
              if not any(n.startswith('main_branch.'+k) for k in ('adapter', 'feedback', 'coverage'))}
    fingerprint = state_hash(common)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['sha256'] != state_hash(saved['model']) or saved['sha256'] != fingerprint:
            raise RuntimeError('Common initialization changed; refusing an unfair run')
        missing, unexpected = model.load_state_dict(saved['model'], strict=False)
        if unexpected or any(not any(n.startswith('main_branch.'+k) for k in ('adapter', 'feedback', 'coverage')) for n in missing):
            raise RuntimeError('Common template schema mismatch')
    else:
        if job['variant'] != 'W01':raise RuntimeError('W01 must freeze the common initialization first')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        torch.save({'model':common, 'sha256':fingerprint, 'seed':job['config']['seed']}, tmp)
        tmp.replace(path)
