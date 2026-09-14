import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/v14-exploration'))
import run_aggregation_confirmation as c


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def options(self):
        return {'dim': 8, 'strides': [2, 4], 'wide_dilations': [2, 4]}

    def test_fixed_masks_across_model_seeds_split_distinct(self):
        shape = (3, 2, 2, 8, 8)
        for pattern in ('scattered','block'):
            a = c.fixed_mask_function(1729, 42)(shape, pattern, .4, 42, [1,3,5])
            b = c.fixed_mask_function(1729, 3407)(shape, pattern, .4, 3407, [1,3,5])
            torch.testing.assert_close(a,b)
            val = c.fixed_mask_function(1729, 42)(shape, pattern, .4, 100042, [1,3,5])
            self.assertFalse(torch.equal(a,val))

    def test_geometry_can_distinguish_missing_locations(self):
        router = c.GeometryRouter(torch.nn.Conv3d(9,3,1)).eval()
        with torch.no_grad():
            router.head.weight.zero_(); router.head.bias.zero_()
            router.head.weight[0,0,0,0,0] = 2.
        x = torch.zeros(1,9,2,8,12)
        x[:,-1,:, :4,:] = 1.
        gate = router(x).softmax(1).permute(0,2,3,4,1)[x[:,-1]==0]
        self.assertGreater(float(gate[:,0].max()-gate[:,0].min()), .01)
        static = c.StaticRouter()(x).softmax(1)
        self.assertEqual(float(static.flatten(2).std(-1).max()), 0.)

    def test_hidden_values_and_all_modes_backward_checkpoint(self):
        x = torch.randn(2,2,2,8,12)
        mask = (torch.rand(2,1,2,8,12)>.4).float()
        hidden = x.masked_fill(~mask.expand_as(x).bool(), float('nan'))
        for name in c.VARIANTS:
            with self.subTest(name=name):
                model = c.ConfirmationProbe(2,name,self.options()).eval()
                a = model(x,mask)[0]
                torch.testing.assert_close(a,model(hidden,mask)[0],atol=0,rtol=0)
                a.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
                for m in (torch.zeros_like(mask),torch.ones_like(mask)):
                    self.assertTrue(torch.isfinite(model(x,m)[0]).all())
                clone = c.ConfirmationProbe(2,name,self.options()).eval()
                clone.load_state_dict(model.state_dict())
                torch.testing.assert_close(a,clone(x,mask)[0])

    def test_single_wide_is_normalized_and_shared_weights(self):
        torch.manual_seed(42); wide = c.ConfirmationProbe(2,'single_wide',self.options())
        torch.manual_seed(42); uni = c.ConfirmationProbe(2,'uniform',self.options())
        f=torch.randn(1,8,2,8,12);mask=torch.ones(1,1,2,8,12)
        a,idx,h,w=wide.assignments(f,mask,2,0)
        b,j,_,_=uni.assignments(f,mask,2,0)
        torch.testing.assert_close(a,b[...,-25:]*3)
        torch.testing.assert_close(a.sum(-1),torch.ones(2,96))
        for key,value in wide.state_dict().items():
            torch.testing.assert_close(value,uni.state_dict()[key])

    def test_protocol_identity_isolates_training_budgets_and_masks(self):
        policy=json.loads((ROOT/'configs/v14-exploration/aggregation_confirmation.json').read_text())
        first,_=c.identity(policy,'BikeNYC','uniform','block',42,'cpu')
        full=copy.deepcopy(policy)
        full['samples'].update(full['profiles']['full']['samples'])
        second,_=c.identity(full,'BikeNYC','uniform','block',42,'cpu')
        self.assertNotEqual(first['fingerprint'],second['fingerprint'])
        changed=copy.deepcopy(policy);changed['mask_seed']+=1
        third,_=c.identity(changed,'BikeNYC','uniform','block',42,'cpu')
        self.assertNotEqual(first['fingerprint'],third['fingerprint'])

    def test_summary_keeps_missing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy={'output_dir':tmp}
            rows=[dict(dataset='BikeNYC',pattern='block',variant='uniform',seed=42,result=None)]
            c.save_summary(rows,policy,None)
            self.assertIn('MISSING',next(Path(tmp).glob('*.log')).read_text())
            result=json.loads(next(Path(tmp).glob('*.json')).read_text())
            self.assertIsNone(result['rows'][0]['result'])

    def test_full_training_protocol_and_diagnostics(self):
        policy=json.loads((ROOT/'configs/v14-exploration/aggregation_confirmation.json').read_text())
        policy['model']=self.options();policy['train'].update(epochs=2,val_epoch=1,batch_size=2)
        data={s:torch.randn(3,2,2,8,12) for s in ('train','val','test')}
        indices={s:[0,3,6] for s in data}
        before=(c.base.Probe,c.base.epoch_pass,c.base.masks)
        try:
            c.base.Probe,c.base.epoch_pass=c.ConfirmationProbe,c.epoch_pass
            for variant in ('single_wide','uniform','static_moe','geometry_moe','context_moe'):
                r,_=c.identity(policy,'BikeNYC',variant,'block',42,'cpu')
                c.base.masks=c.fixed_mask_function(policy['mask_seed'],42)
                with tempfile.TemporaryDirectory() as tmp:
                    result=c.base.train_job(r,Path(tmp),data,indices,torch.device('cpu'))
                    self.assertIsNotNone(c.base.complete(Path(tmp),r))
                    if variant.endswith('moe'):
                        self.assertIn('2',result['val']['routing_missing'])
                    self.assertEqual(len(list(Path(result['run']).glob('*.pt'))),1)
        finally:
            c.base.Probe,c.base.epoch_pass,c.base.masks=before


if __name__=='__main__':
    unittest.main()
