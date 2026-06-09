import torch, torch.nn as nn
from .common import ResBlock3D, upsample_to
class FixedExpert(nn.Module):
    def __init__(self,c_in,h):
        super().__init__(); self.net=nn.Sequential(nn.Conv3d(c_in+1,h,3,padding=1),nn.GroupNorm(min(8,h),h),nn.GELU(),ResBlock3D(h),ResBlock3D(h))
    def forward(self,x,m): return self.net(torch.cat([x,m],1))
class FixedScaleExperts(nn.Module):
    def __init__(self,c_in,hidden_dim=64):
        super().__init__(); h=hidden_dim; self.f=FixedExpert(c_in,h); self.m=FixedExpert(c_in,h); self.c=FixedExpert(c_in,h); self.gate=nn.Sequential(nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(h*3,3),nn.Softmax(-1)); self.fuse=nn.Sequential(nn.Conv3d(h,h,1),ResBlock3D(h)); self.head=nn.Conv3d(h,c_in,1)
    def forward(self,batch):
        f=self.f(batch['x_f_obs'],batch['m_f']); m=upsample_to(self.m(batch['x_m_obs'],batch['m_m']),f); c=upsample_to(self.c(batch['x_c_obs'],batch['m_c']),f); g=self.gate(torch.cat([f,m,c],1)); h=g[:,0].view(-1,1,1,1,1)*f+g[:,1].view(-1,1,1,1,1)*m+g[:,2].view(-1,1,1,1,1)*c; h=self.fuse(h); out=self.head(h); return {'x_hat_main':out,'x_hat_final':out,'h_st_aux':h,'scale_gate':g}
