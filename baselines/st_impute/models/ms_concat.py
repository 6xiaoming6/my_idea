import torch, torch.nn as nn
from .common import ResBlock3D, upsample_to
class ScaleEncoder(nn.Module):
    def __init__(self,c_in,h):
        super().__init__(); self.net=nn.Sequential(nn.Conv3d(c_in+1,h,3,padding=1),nn.GroupNorm(min(8,h),h),nn.GELU(),ResBlock3D(h))
    def forward(self,x,m): return self.net(torch.cat([x,m],dim=1))
class MultiScaleConcatFusion(nn.Module):
    def __init__(self,c_in,hidden_dim=64):
        super().__init__(); h=hidden_dim; self.ef=ScaleEncoder(c_in,h); self.em=ScaleEncoder(c_in,h); self.ec=ScaleEncoder(c_in,h); self.fuse=nn.Sequential(nn.Conv3d(h*3,h,1),ResBlock3D(h),ResBlock3D(h)); self.head=nn.Sequential(nn.Conv3d(h,h//2,3,padding=1),nn.GELU(),nn.Conv3d(h//2,c_in,1))
    def forward(self,batch):
        f=self.ef(batch['x_f_obs'],batch['m_f']); m=upsample_to(self.em(batch['x_m_obs'],batch['m_m']),f); c=upsample_to(self.ec(batch['x_c_obs'],batch['m_c']),f); h=self.fuse(torch.cat([f,m,c],1)); out=self.head(h); return {'x_hat_main':out,'x_hat_final':out,'h_st_aux':h}
