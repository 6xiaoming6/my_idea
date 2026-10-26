"""Opt-in I37 execution optimizations; no parameters, buffers or RNG draws."""
import copy
from functools import lru_cache
import torch

FEATURES=('skip_redundant_init','reuse_fine_support','no_grad_intermediate','compact_outputs','cache_constants','pack_scalar_logs','skip_last_history')

def configure(model,cfg):
    spec=copy.deepcopy(cfg.get('train',{}).get('performance',{}))
    if set(spec)-{'enabled','diagnostic_interval',*FEATURES}:raise ValueError('Unknown performance option')
    if type(spec.get('enabled',False)) is not bool:raise ValueError('performance.enabled must be boolean')
    if not spec.get('enabled',False):model.performance={'enabled':False};return
    if (cfg.get('loss',{}).get('lambda_coe_mid',0)!=0 or model.teacher!='none'
            or model.feedback!='none' or model.w!={'enabled':True}
            or cfg['model'].get('aux',{}).get('enabled',False)):
        raise ValueError('Performance path requires the I37 no-probe/no-feedback/no-intermediate-loss contract')
    interval=spec.get('diagnostic_interval',20)
    if type(interval) is not int or interval<1:raise ValueError('diagnostic_interval must be positive')
    for key in FEATURES:
        if type(spec.get(key,True)) is not bool:raise ValueError(key+' must be boolean')
    model.performance=dict(enabled=True,diagnostic_interval=interval,**{k:spec.get(k,True) for k in FEATURES})

def enabled(model,key=None):
    spec=getattr(model,'performance',{})
    return bool(spec.get('enabled',False) and (key is None or spec.get(key,False)))

def diagnostics(model):
    return not enabled(model) or bool(getattr(model,'_collect_diagnostics',True))

def sample_batch(model,index,total):
    return not enabled(model) or index==0 or index==total-1 or index%model.performance['diagnostic_interval']==0

@lru_cache(maxsize=16)
def support_constants(device,dtype,t,h,w,temporal_kernel,spatial_kernel):
    from torch.nn import functional as F
    ones=torch.ones((1,1,t,h,w),device=device,dtype=dtype)
    time_filter=torch.ones((1,1,temporal_kernel,1,1),device=device,dtype=dtype)
    space_filter=torch.ones((1,1,1,spatial_kernel,spatial_kernel),device=device,dtype=dtype)
    space_filter[...,spatial_kernel//2,spatial_kernel//2]=0
    time_denominator=F.conv3d(ones,time_filter,padding=(temporal_kernel//2,0,0))
    space_denominator=F.conv3d(ones,space_filter,padding=(0,spatial_kernel//2,spatial_kernel//2))
    return time_filter,space_filter,time_denominator,space_denominator,torch.arange(t,device=device).reshape(1,1,t,1,1)

@lru_cache(maxsize=16)
def coordinates(device,dtype,t,h,w):
    axes=[torch.linspace(-1,1,n,device=device,dtype=dtype) if n>1 else torch.zeros(1,device=device,dtype=dtype) for n in (t,h,w)]
    return torch.stack(torch.meshgrid(*axes,indexing='ij'),dim=0)

def append_scalars(logs,values):
    """Transfer detached scalars together; keep their existing arithmetic/order."""
    items=[(k,v) for k,v in values.items() if torch.is_tensor(v)]
    if items:
        # Widen only AFTER each original FP32 scalar reduction. Widening exactly
        # represents FP32 values and Python still accumulates in original order.
        packed=torch.stack([v.detach().to(torch.float64) for _,v in items]).cpu().tolist()
        for (key,_),value in zip(items,packed):logs[key].append(float(value))
    for key,value in values.items():
        if not torch.is_tensor(value):logs[key].append(float(value))
