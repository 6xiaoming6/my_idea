import torch, torch.nn as nn
class SimpleTransformerImputer(nn.Module):
    def __init__(self,c_in,h,w,window_len,hidden_dim=256,nhead=4,num_layers=2):
        super().__init__(); self.c_in=c_in; self.h=h; self.w=w; self.window_len=window_len; F=c_in*h*w; self.inp=nn.Linear(F*2,hidden_dim); layer=nn.TransformerEncoderLayer(hidden_dim,nhead,hidden_dim*4,batch_first=True,dropout=0.1,activation='gelu'); self.enc=nn.TransformerEncoder(layer,num_layers); self.pos=nn.Parameter(torch.randn(1,window_len,hidden_dim)*0.02); self.out=nn.Linear(hidden_dim,F)
    def forward(self,batch):
        x=batch['x_f_obs']; m=batch['m_f']; B,C,T,H,W=x.shape
        xs=x.permute(0,2,1,3,4).reshape(B,T,C*H*W); ms=m.expand(B,C,T,H,W).permute(0,2,1,3,4).reshape(B,T,C*H*W)
        h=self.enc(self.inp(torch.cat([xs,ms],-1))+self.pos[:,:T]); out=self.out(h).reshape(B,T,C,H,W).permute(0,2,1,3,4); return {'x_hat_main':out,'x_hat_final':out,'h_st_aux':h}
