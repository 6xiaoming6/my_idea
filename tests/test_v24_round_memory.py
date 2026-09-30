import copy
import unittest
import torch
from test_v24_n_exploration import config,batch
from stmoe_imputer.models import DualBranchSTImputer


class RoundMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_zero_initialized_memory_matches_n3_train_and_eval(self):
        for training in (True,False):
            torch.manual_seed(7);base=DualBranchSTImputer.from_config(config('N3')).train(training)
            torch.manual_seed(7);memory=DualBranchSTImputer.from_config(config('N7')).train(training)
            for key,value in base.state_dict().items():torch.testing.assert_close(value,memory.state_dict()[key],rtol=0,atol=0)
            data=batch();a=base(data);b=memory(data)
            torch.testing.assert_close(a['x_hat_main'],b['x_hat_main'],rtol=0,atol=0)
            torch.testing.assert_close(a['coe']['route_weights'],b['coe']['route_weights'],rtol=0,atol=0)
            for x,y in zip(a['coe']['predictions'],b['coe']['predictions']):torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertEqual(b['coe']['round_memory_gates'].abs().sum().item(),0)

    def test_history_is_causal_and_batch_local_without_target_leakage(self):
        model=DualBranchSTImputer.from_config(config('N7')).eval()
        model.main_branch.memory_gate[-1].bias.data.fill_(.2)
        data=batch()
        with torch.no_grad():
            first=model(data)
            model(batch())
            altered={**data,'x_f_gt':data['x_f_gt']+10000.,
                     'x_f_obs':torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))}
            second=model(altered)
        torch.testing.assert_close(first['x_hat_main'],second['x_hat_main'],rtol=0,atol=0)
        weights=first['coe']['round_memory_weights']
        self.assertTrue((weights[:,0]==0).all())
        for step in range(1,4):
            torch.testing.assert_close(weights[:,step,:step].sum(-1),torch.ones(2))
            self.assertEqual(weights[:,step,step:].abs().sum().item(),0)
        self.assertEqual(model.main_branch._memory_states,[])
        self.assertEqual(model.main_branch._memory_summaries,[])

    def test_main_loss_trains_gate_without_extra_expert_execution(self):
        torch.manual_seed(7);model=DualBranchSTImputer.from_config(config('N7')).train()
        backbone=model.main_branch
        for router in backbone.routers:
            torch.nn.init.zeros_(router[-1].weight)
            router[-1].bias.data.copy_(torch.arange(8,0,-1).float())
        calls=[];handles=[]
        for eid,expert in enumerate(backbone.routed_experts()):
            def hook(module,args,out,eid=eid):calls.append((eid,args[0].shape[0]))
            handles.append(expert.register_forward_hook(hook))
        data=batch();out=model(data)
        loss=((out['x_hat_main']-data['x_f_gt']).square()*(1-data['m_f'])).mean();loss.backward()
        for handle in handles:handle.remove()
        self.assertEqual(set(eid for eid,_ in calls),{0,1})
        self.assertEqual(sum(n for _,n in calls),2*4*2)
        grad=backbone.memory_gate[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)
        for expert in backbone.routed_experts()[2:]:self.assertTrue(all(p.grad is None for p in expert.parameters()))
        # After opening the gate, retrieval keys/queries must also get task gradients.
        model.zero_grad(set_to_none=True);backbone.memory_gate[-1].bias.data.fill_(.2)
        out=model(data);out['x_hat_main'].square().mean().backward()
        for module in (backbone.memory_query,backbone.memory_key):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)

    def test_memory_path_keeps_gradients_to_old_states(self):
        model=DualBranchSTImputer.from_config(config('N7')).main_branch
        model.memory_gate[-1].bias.data.fill_(.2)
        model._memory_states=[];model._memory_summaries=[]
        model._memory_weight_history=[];model._memory_gate_history=[];model._memory_delta_history=[]
        old=torch.randn(2,8,3,8,8,requires_grad=True)
        current=torch.randn_like(old,requires_grad=True)
        values=torch.randn(2,2,3,8,8);mask=(torch.rand_like(values)>.4).float()
        support=model._observation_support(mask);position=model._position(values)
        model._expert_input(old,values,mask,support,position,0)
        result=model._expert_input(current,values,mask,support,position,1)
        (result*torch.randn_like(result)).sum().backward()
        self.assertIsNotNone(old.grad);self.assertGreater(old.grad.abs().sum().item(),0)

    def test_cannot_combine_n7_with_scale_or_local_routing(self):
        for name in ('local_routing','spatial_scale'):
            cfg=config('N7');cfg['model']['coe'][name]={'enabled':True}
            with self.assertRaises((ValueError,KeyError)):
                DualBranchSTImputer.from_config(cfg)


if __name__=='__main__':unittest.main()
