import torch, torch.nn as nn
from .common import ResBlock3D, upsample_to
class Embed(nn.Module):
    def __init__(self,c_in,h):
        super().__init__(); self.net=nn.Sequential(nn.Conv3d(c_in+1,h,3,padding=1),nn.GroupNorm(min(8,h),h),nn.GELU())
    def forward(self,x,m): return self.net(torch.cat([x,m],1))
class Expert(nn.Module):
    def __init__(self,h): super().__init__(); self.net=nn.Sequential(ResBlock3D(h),ResBlock3D(h))
    def forward(self,x): return self.net(x)
class SharedExpertsNoRouter(nn.Module):
    def __init__(self,c_in,hidden_dim=64,num_experts=4):
        super().__init__(); h=hidden_dim; self.ef=Embed(c_in,h); self.em=Embed(c_in,h); self.ec=Embed(c_in,h); self.experts=nn.ModuleList([Expert(h) for _ in range(num_experts)]); self.cross=nn.Sequential(nn.Conv3d(h*3,h,1),ResBlock3D(h)); self.fuse=nn.Sequential(nn.Conv3d(h*4,h,1),ResBlock3D(h)); self.head=nn.Conv3d(h,c_in,1)
    def apply_exp(self,x): return torch.stack([e(x) for e in self.experts],0).mean(0)
    def forward(self,batch):
        f=self.apply_exp(self.ef(batch['x_f_obs'],batch['m_f'])); m=upsample_to(self.apply_exp(self.em(batch['x_m_obs'],batch['m_m'])),f); c=upsample_to(self.apply_exp(self.ec(batch['x_c_obs'],batch['m_c'])),f); cross=self.cross(torch.cat([f,m,c],1)); h=self.fuse(torch.cat([f,m,c,cross],1)); out=self.head(h); return {'x_hat_main':out,'x_hat_final':out,'h_st_aux':h}
