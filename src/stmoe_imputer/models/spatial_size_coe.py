"""lh/lw study with matched fixed/learned spatial Linear mapping controls."""
from functools import lru_cache
import torch
from torch import nn
from .latent_space_coe import LatentSpaceCoE
from .observed_consistency_coe import FreeScaleHistoryConsistencyCoE
from ..utils.deterministic import resize, _weights

@lru_cache(maxsize=64)
def area_weights(source,target):
    if not 0<target<source:raise ValueError('Area weights require downsampling')
    # Pixel-overlap areas, normalized per output cell. No learned parameters.
    edge=torch.arange(target+1,dtype=torch.float64)*(source/target)
    left=torch.arange(source,dtype=torch.float64)
    overlap=(torch.minimum(edge[1:,None],left[None]+1)-torch.maximum(edge[:-1,None],left[None])).clamp_min(0)
    return (overlap/(source/target)).float()


def spatial_map(x,size):
    th,tw=tuple(size);h,w=x.shape[-2:]
    if (h,w)==(th,tw):return x
    if th>=h and tw>=w:return resize(x,(th,tw))
    if th<=h and tw<=w and h%th==0 and w%tw==0:
        # Exact original I37 pooling order, including rectangular integer ratios.
        b,c,t,_,_=x.shape
        return x.reshape(b,c,t,th,h//th,tw,w//tw).mean((4,6))
    with torch.autocast(device_type=x.device.type,enabled=False):
        y=x.float()
        if th<h:y=torch.matmul(area_weights(h,th).to(x.device),y)
        if tw<w:y=torch.matmul(y,area_weights(w,tw).to(x.device).T)
    y=y.to(x.dtype)
    return resize(y,(th,tw)) # remaining enlargements; same bilinear rule as I37


class SpatialBridge(nn.Module):
    def __init__(self,source,target):
        super().__init__();self.source=tuple(source);self.target=tuple(target)
    def forward(self,x):
        if tuple(x.shape[-2:])!=self.source:raise ValueError('Unexpected bridge input size')
        return spatial_map(x,self.target)


class LinearSpatialBridge(nn.Module):
    """Separable learned spatial Linear layers, shared over channels/time/rounds.

    Height and width weights start from the fixed area/bilinear resampler.
    This is an unconstrained learned hidden-feature mapping, not an observed
    mass-conserving projection. Final observation consistency is unchanged.
    """
    def __init__(self,source,target):
        super().__init__();self.source=tuple(source);self.target=tuple(target)
        self.height=nn.Linear(source[0],target[0],bias=False)
        self.width=nn.Linear(source[1],target[1],bias=False)
        def weights(n,m):
            if n==m:return torch.eye(n)
            return area_weights(n,m) if m<n else _weights(n,m)
        with torch.no_grad():
            self.height.weight.copy_(weights(source[0],target[0]))
            self.width.weight.copy_(weights(source[1],target[1]))
    def forward(self,x):
        if tuple(x.shape[-2:])!=self.source:raise ValueError('Unexpected linear bridge input size')
        # FP32 mapping matches fixed-resampling accumulation precision; AMP
        # remains enabled for the unchanged expert/encoder/decoder operations.
        with torch.autocast(device_type=x.device.type,enabled=False):
            y=self.width(x.float())
            y=self.height(y.transpose(-1,-2)).transpose(-1,-2)
        return y.to(x.dtype)


class SpatialSizeCoE(LatentSpaceCoE):
    @classmethod
    def from_config(cls,cfg):
        spec=cfg['model']['coe']['latent_size']
        if set(spec)!={'lh','lw','projection'} or any(type(spec[k]) is not int or not 4<=spec[k]<=64 for k in ('lh','lw')):
            raise ValueError('Specify lh/lw in [4,64] and projection')
        if spec['projection'] not in ('fixed','linear_in','linear_out','linear_both'):raise ValueError('Unknown projection')
        m=FreeScaleHistoryConsistencyCoE.from_config.__func__(cls,cfg)
        m.latent_spec=dict(mapping='linear',return_mode='replace',delta=True,history=True,path='free_last_f')
        m.fine_size=(cfg['model']['main']['h'],cfg['model']['main']['w'])
        m.latent_size=(spec['lh'],spec['lw'])
        native=[tuple(v//f for v in m.fine_size) for f in (1,2,4)]
        enter=LinearSpatialBridge if spec['projection'] in ('linear_in','linear_both') else SpatialBridge
        leave=LinearSpatialBridge if spec['projection'] in ('linear_out','linear_both') else SpatialBridge
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg['seed'])+115001)
            m.to_latent=nn.ModuleList([enter(s,m.latent_size) for s in native])
            m.to_native=nn.ModuleList([leave(m.latent_size,s) for s in native])
        return m
    @staticmethod
    def _map_spatial(x,size):return spatial_map(x,size)
