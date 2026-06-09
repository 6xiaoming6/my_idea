import torch.nn as nn, torch.nn.functional as F
class ResBlock3D(nn.Module):
    def __init__(self,dim,groups=8):
        super().__init__(); groups=min(groups,dim); self.c1=nn.Conv3d(dim,dim,3,padding=1); self.n1=nn.GroupNorm(groups,dim); self.c2=nn.Conv3d(dim,dim,3,padding=1); self.n2=nn.GroupNorm(groups,dim)
    def forward(self,x):
        y=F.gelu(self.n1(self.c1(x))); y=self.n2(self.c2(y)); return F.gelu(x+y)
class ConvBlock3D(nn.Module):
    def __init__(self,cin,cout,groups=8):
        super().__init__(); groups=min(groups,cout); self.net=nn.Sequential(nn.Conv3d(cin,cout,3,padding=1),nn.GroupNorm(groups,cout),nn.GELU(),nn.Conv3d(cout,cout,3,padding=1),nn.GroupNorm(groups,cout),nn.GELU())
    def forward(self,x): return self.net(x)
def upsample_to(x,ref):
    return F.interpolate(x,size=ref.shape[-3:],mode='trilinear',align_corners=False)
