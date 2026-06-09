import torch, torch.nn as nn, torch.nn.functional as F
from .common import ConvBlock3D
class Conv3DUNet(nn.Module):
    def __init__(self,c_in,hidden_dim=64):
        super().__init__(); h=hidden_dim
        self.e1=ConvBlock3D(c_in+1,h); self.d1=nn.Conv3d(h,h*2,(1,4,4),stride=(1,2,2),padding=(0,1,1)); self.e2=ConvBlock3D(h*2,h*2); self.d2=nn.Conv3d(h*2,h*4,(1,4,4),stride=(1,2,2),padding=(0,1,1)); self.b=ConvBlock3D(h*4,h*4)
        self.u2=nn.ConvTranspose3d(h*4,h*2,(1,4,4),stride=(1,2,2),padding=(0,1,1)); self.dec2=ConvBlock3D(h*4,h*2); self.u1=nn.ConvTranspose3d(h*2,h,(1,4,4),stride=(1,2,2),padding=(0,1,1)); self.dec1=ConvBlock3D(h*2,h); self.head=nn.Conv3d(h,c_in,1)
    def forward(self,batch):
        x=torch.cat([batch['x_f_obs'],batch['m_f']],dim=1); e1=self.e1(x); e2=self.e2(F.gelu(self.d1(e1))); b=self.b(F.gelu(self.d2(e2)))
        u2=self.u2(b); u2=F.interpolate(u2,size=e2.shape[-3:],mode='trilinear',align_corners=False) if u2.shape[-3:]!=e2.shape[-3:] else u2; d2=self.dec2(torch.cat([u2,e2],1))
        u1=self.u1(d2); u1=F.interpolate(u1,size=e1.shape[-3:],mode='trilinear',align_corners=False) if u1.shape[-3:]!=e1.shape[-3:] else u1; d1=self.dec1(torch.cat([u1,e1],1))
        out=self.head(d1); return {'x_hat_main':out,'x_hat_final':out,'h_st_aux':d1}
