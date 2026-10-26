"""Semantic tests for the external baseline data adapter (not model rewrites)."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'imputation_benchmark/scripts/train/run_v25_simple_baselines.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name('run_v25_simple_baselines.py')
spec = importlib.util.spec_from_file_location('simple_baselines', SCRIPT)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
if (ROOT/'src').exists():
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'imputation_benchmark/baselines/SAITS')]
else:
    sys.path[:0] = [str(SCRIPT.parent/'src'), str(SCRIPT.parent/'upstream')]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.stats = dict(node=np.full((2,8,8),3.,np.float32),std=np.ones((2,8,8),np.float32),channel=np.array([3.,3.],np.float32))
        gen=torch.Generator().manual_seed(1)
        x=torch.rand((2,2,12,8,8),generator=gen)*10
        mf=(torch.rand(x.shape,generator=gen)>.4).float()
        self.batch=dict(x_f_gt=x,m_f=mf,x_f_obs=torch.where(mf.bool(),x,0.),target_mask=1-mf,mask_family=torch.tensor([0,1]))
        for factor,n in [(2,'m'),(4,'c')]:
            gt=x.reshape(2,2,12,8//factor,factor,8//factor,factor).sum((4,6))
            mask=(torch.rand(gt.shape,generator=gen)>.4).float()
            self.batch['m_'+n]=mask;self.batch['x_'+n+'_obs']=torch.where(mask.bool(),gt,0.)

    def test_patch_roundtrip(self):
        x=torch.arange(2*2*12*32*32).reshape(2,2,12,32,32)
        torch.testing.assert_close(b.unpatch(b.patches(x),x.shape),x,rtol=0,atol=0)

    def test_temporal_interpolation_and_fallback(self):
        x=np.array([2,0,0,8,0],dtype=np.float32).reshape(1,1,5,1,1)
        m=np.array([1,0,0,1,0],dtype=bool).reshape(x.shape)
        np.testing.assert_allclose(b.temporal(x,m,np.array([[[9]]])),np.array([2,4,6,8,8]).reshape(x.shape))
        np.testing.assert_allclose(b.temporal(x,m*False,np.array([[[9]]])),9.)

    def test_traditional_hidden_poison_and_observation_preservation(self):
        for name,opt in [('ChannelMean',None),('NodeMean',None),('TemporalLinear',None),('SpatialMean',3),('SoftImpute',.1)]:
            x=b.traditional(name,self.batch,self.stats,opt)
            poison={k:v.clone() for k,v in self.batch.items()}
            poison['x_f_gt'].fill_(987654)
            poison['x_f_obs'][~poison['m_f'].bool()]=float('nan')
            y=b.traditional(name,poison,self.stats,opt)
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            m=self.batch['m_f'].bool()
            torch.testing.assert_close(x[m],self.batch['x_f_obs'][m],rtol=0,atol=0)

    def test_native_loss_and_gradients_need_no_targets_or_coarse_inputs(self):
        # The adapter must preserve the upstream objective, including gradients.
        visible={k:self.batch[k] for k in ['x_f_obs','m_f']}
        for name in b.NEURAL:
            torch.manual_seed(7)
            model=b.new_model(name,torch.device('cpu'));model.train()
            inputs,_,_=b.neural_inputs(visible,self.stats,'cpu')
            native=model(inputs,'train') if name.startswith('BRITS') else model(inputs)
            expected=native['reconstruction_loss']+native['consistency_loss']
            params=tuple(model.parameters())
            expected_grad=torch.autograd.grad(expected,params)
            pred,actual=b.neural_forward(model,name,visible,self.stats,'cpu',True)
            actual_grad=torch.autograd.grad(actual,params)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for x,y in zip(expected_grad,actual_grad):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            poison={k:v.clone() for k,v in self.batch.items()}
            for k in ['x_f_gt','target_mask','x_m_obs','x_c_obs','m_m','m_c']:
                poison[k].fill_(float('nan'))
            poisoned_pred,poisoned_loss=b.neural_forward(model,name,poison,self.stats,'cpu',True)
            torch.testing.assert_close(pred,poisoned_pred,rtol=0,atol=0)
            torch.testing.assert_close(actual,poisoned_loss,rtol=0,atol=0)

    def test_legacy_policy_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Legacy suite'):
            b.validate_policy({'schema':1})
        b.validate_policy({'baseline_policy':b.BASELINE_POLICY})

    def test_rnn_no_targets_and_backward(self):
        for name in b.NEURAL:
            torch.manual_seed(7);model=b.new_model(name,torch.device('cpu'));model.eval()
            x,_=b.neural_forward(model,name,self.batch,self.stats,'cpu')
            poison={k:v.clone() for k,v in self.batch.items()}
            poison['x_f_gt'].fill_(float('nan'));poison['x_f_obs'][~poison['m_f'].bool()]=float('nan')
            y,_=b.neural_forward(model,name,poison,self.stats,'cpu')
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            _,loss=b.neural_forward(model,name,self.batch,self.stats,'cpu',True);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_metric_counts_and_masking(self):
        scores=b.Scores();pred=self.batch['x_f_gt']+2
        pred[self.batch['m_f'].bool()]=999999
        scores.update(pred,self.batch);r=scores.result()
        self.assertAlmostEqual(r['mae'],2,places=6);self.assertAlmostEqual(r['rmse'],2,places=6)
        self.assertEqual(r['count'],int(self.batch['target_mask'].sum()))


if __name__=='__main__':unittest.main()
