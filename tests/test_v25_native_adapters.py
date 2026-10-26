import sys
import os
from pathlib import Path
import numpy as np
import unittest
import torch
ROOT=Path(os.environ.get('V25_BASELINE_ROOT',Path(__file__).resolve().parents[1]))
SOURCES=Path(os.environ.get('V25_BASELINE_SOURCES',ROOT/'imputation_benchmark/baselines'))
script_dir=Path(__file__).resolve().parent
sys.path.insert(0,str(script_dir if (script_dir/'v25_native_adapters.py').exists() else ROOT/'imputation_benchmark/scripts/train'))
import v25_native_adapters as a

def test_native_adapter(name):
    torch.set_num_threads(2);torch.manual_seed(7)
    shape=(2,2,12,4,4);gt=torch.rand(shape)*20
    mask=(torch.rand(shape)>.4).float()
    b=dict(x_f_obs=gt*mask,m_f=mask,x_f_gt=gt,target_mask=1-mask)
    b['calendar']=torch.zeros((2,12,4),dtype=torch.long)
    b['calendar'][:,:,2]=torch.arange(12)//2
    b['calendar'][:,:,3]=(torch.arange(12)%2)*2
    stats={'node':np.zeros((2,4,4),np.float32),'std':np.ones((2,4,4),np.float32)}
    net=a.build(name,SOURCES,torch.device('cpu'),shape[1:])
    net.eval()
    with torch.no_grad():
        rng=torch.get_rng_state()
        p,_=a.forward(name,net,b,stats,'cpu')
        polluted={k:v.clone() for k,v in b.items()}
        polluted['x_f_gt'].fill_(float('nan'))
        polluted['x_f_obs'][mask==0]=float('nan')
        for k in ['x_m_gt','x_c_gt','x_m_obs','x_c_obs']:polluted[k]=torch.tensor(float('nan'))
        torch.set_rng_state(rng)
        q,_=a.forward(name,net,polluted,stats,'cpu')
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        assert torch.isfinite(p).all() and p.shape==gt.shape
        torch.testing.assert_close(p[mask.bool()],gt[mask.bool()])
    net.train();_,loss=a.forward(name,net,b,stats,'cpu',True)
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum()>0 for g in grads)

def test_layout_and_adjacency():
    x=torch.arange(2*2*12*4*4).reshape(2,2,12,4,4)
    assert torch.equal(a.unflat(a.flat(x),x.shape),x)
    assert torch.equal(a.ungraph(a.graph(x),x.shape),x)
    g=a.adjacency(4,4)
    assert np.allclose(g,g.T) and g[0].sum()==2 and g[5].sum()==4
    assert g[3,4]==0

class AdapterTests(unittest.TestCase):
    def test_models(self):
        for name in a.METHODS:
            with self.subTest(model=name):test_native_adapter(name)
    def test_layout(self):test_layout_and_adjacency()
    def test_grin_auxiliary_sum(self):
        class Fixed(torch.nn.Module):
            def forward(self,x,mask):
                return torch.ones_like(x),torch.stack([torch.full_like(x,i) for i in [2,3,4,5]])
        x=torch.zeros(1,2,12,4,4);mask=torch.ones_like(x)
        batch=dict(x_f_obs=x,m_f=mask,x_f_gt=x)
        stats={'node':np.zeros((2,4,4),np.float32),'std':np.ones((2,4,4),np.float32)}
        _,loss=a.forward('GRIN',Fixed(),batch,stats,'cpu',True)
        self.assertEqual(loss.item(),15.)

if __name__ == '__main__':unittest.main()
