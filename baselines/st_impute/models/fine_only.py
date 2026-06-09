import torch, torch.nn as nn
from .common import ResBlock3D
class FineOnlyConv3D(nn.Module):
    def __init__(self,c_in,hidden_dim=64,num_blocks=4):
        super().__init__(); self.proj=nn.Sequential(nn.Conv3d(c_in+1,hidden_dim,3,padding=1),nn.GroupNorm(min(8,hidden_dim),hidden_dim),nn.GELU()); self.blocks=nn.Sequential(*[ResBlock3D(hidden_dim) for _ in range(num_blocks)]); self.head=nn.Sequential(nn.Conv3d(hidden_dim,hidden_dim//2,3,padding=1),nn.GELU(),nn.Conv3d(hidden_dim//2,c_in,1))
    def forward(self,batch):
        h=self.blocks(self.proj(torch.cat([batch['x_f_obs'],batch['m_f']],dim=1))); out=self.head(h); return {'x_hat_main':out,'x_hat_final':out,'h_st_aux':h}
