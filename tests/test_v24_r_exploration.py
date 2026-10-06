import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/v24'))
from run_r_exploration import jobs
from run_n_exploration import jobs as old_jobs
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data.diverse_masks import make_diverse_mask,make_triple_mask
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


def config(name):
    c=jobs(variants=(name,))[0]['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
    c['train']['amp']=False
    return c


def model(name):
    torch.manual_seed(7)
    return DualBranchSTImputer.from_config(config(name))


def batch(n=2):
    x=torch.randn(n,2,3,8,8);mask=(torch.rand_like(x)>.4).float()
    return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x}


class RExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_repeat_configs_only_change_seeds_and_metadata(self):
        for r,n in (('R1','N3'),('R2','N5'),('R3','N6')):
            a=jobs(variants=(r,))[0]['config'];b=old_jobs(variants=(n,))[0]['config']
            self.assertEqual(a['seed'],17);self.assertEqual(a['data']['loader_seed'],17)
            a['seed']=b['seed'];a['data']['loader_seed']=b['data']['loader_seed']
            a.pop('experiment_plan');b.pop('experiment_plan');self.assertEqual(a,b)
        self.assertEqual([j['variant'] for j in jobs()],[f'R{i}' for i in range(1,11)])
        for j in jobs('bikenyc'):self.assertEqual(j['config']['data']['dataset_name'],'BikeNYC')

    def test_fixed_and_free_initial_match_fixed_and_train_head(self):
        fixed=model('R2').eval();data=batch()
        with torch.no_grad():baseline=fixed(data)
        for name in ('R4','R7'):
            m=model(name).eval()
            with torch.no_grad():out=m(data)
            torch.testing.assert_close(out['x_hat_final'],baseline['x_hat_final'],rtol=0,atol=0)
            torch.testing.assert_close(out['coe']['selected_scales'],torch.tensor([[1,1,0,0]]*2))
            m.train();out=m(data);(out['x_hat_final']-data['x_f_gt']).square().mean().backward()
            grad=m.main_branch.scale_routers[0][-1].bias.grad
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum(),0)

    def test_free_all_paths_sparse_execution(self):
        data=batch();m=model('R7').train();back=m.main_branch
        for router in back.routers:
            torch.nn.init.zeros_(router[-1].weight);router[-1].bias.data.copy_(torch.arange(8,0,-1))
        for code in range(16):
            path=[int(bool(code&(1<<step))) for step in range(4)]
            for step,head in enumerate(back.scale_routers):head[-1].bias.data.copy_(torch.tensor([-3.,3.] if path[step] else [3.,-3.]))
            calls=[];handles=[]
            for eid,e in enumerate(back.routed_experts()):
                handles.append(e.register_forward_pre_hook(lambda m,args,eid=eid:calls.append((eid,args[0].shape[0],args[0].shape[-1]))))
            out=m(data)
            for h in handles:h.remove()
            self.assertEqual(out['coe']['selected_scales'].tolist(),[path]*2)
            self.assertEqual(sum(b for _,b,_ in calls),16)
            self.assertEqual(sum(b for _,b,w in calls if w==4),sum(path)*4)
            self.assertEqual(set(e for e,_,_ in calls),{0,1})
            self.assertAlmostEqual(out['diagnostics']['coe']['expert_grid_equivalents'].item(),8-1.5*sum(path))
        m.zero_grad();out=m(data);out['x_hat_final'].square().mean().backward()
        for e in back.routed_experts()[2:]:self.assertTrue(all(p.grad is None for p in e.parameters()))

    def test_perturb_checkpoint_rng_and_eval(self):
        m=model('R5').train();data=batch(12)
        global_rng=torch.random.get_rng_state().clone()
        out=m(data);torch.testing.assert_close(torch.random.get_rng_state(),global_rng)
        paths=out['coe']['selected_scales'].tolist()
        self.assertTrue(all(p in ([1,1,0,0],[0,1,1,0]) for p in paths))
        self.assertEqual(len({tuple(p) for p in paths}),2)
        state=copy.deepcopy(m.state_dict());a=m(data)['coe']['selected_scales']
        restored=model('R5').train();restored.load_state_dict(state)
        torch.testing.assert_close(a,restored(data)['coe']['selected_scales'])
        rng=restored.main_branch._perturb_rng_state.clone();restored.eval()
        self.assertEqual(restored(data)['coe']['selected_scales'].tolist(),[[1,1,0,0]]*12)
        torch.testing.assert_close(rng,restored.main_branch._perturb_rng_state)
        restored.train();restored.main_branch.set_routing_epoch(9)
        self.assertEqual(restored(data)['coe']['selected_scales'].tolist(),[[1,1,0,0]]*12)
        torch.testing.assert_close(rng,restored.main_branch._perturb_rng_state)
        self.assertTrue(all(not p.requires_grad for p in restored.main_branch.scale_routers.parameters()))

    def test_independent_experts_used_and_disjoint(self):
        m=model('R6');data=batch();out=m(data)
        self.assertEqual(out['coe']['selected_scales'].tolist(),[[1,1,0,0]]*2)
        a={id(p) for e in m.main_branch.routed_experts(0) for p in e.parameters()}
        b={id(p) for e in m.main_branch.routed_experts(1) for p in e.parameters()}
        self.assertFalse(a&b)
        out['x_hat_final'].square().mean().backward()
        for step in range(4):self.assertTrue(any(p.grad is not None for e in m.main_branch.routed_experts(step) for p in e.parameters()))

    def test_memory_histories_causality_initial_and_gradient(self):
        data=batch();base=model('R1').eval()
        with torch.no_grad():reference=base(data)['x_hat_final']
        for name in ('R8','R9'):
            m=model(name).eval()
            with torch.no_grad():out=m(data)
            torch.testing.assert_close(out['x_hat_final'],reference,rtol=0,atol=0)
            weights=out['coe']['round_memory_weights']
            for step in range(1,4):
                expected=torch.zeros(4)
                if name=='R8':expected[:step]=1/step
                else:expected[step-1]=1
                torch.testing.assert_close(weights[0,step],expected)
            self.assertFalse(m.main_branch._memory_states)
            m.train();out=m(data);out['x_hat_final'].square().mean().backward()
            grad=m.main_branch.memory_gate[-1].weight.grad
            self.assertGreater(grad.abs().sum(),0);self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue(all(not p.requires_grad for p in m.main_branch.memory_query.parameters()))
            m.eval()
            with torch.no_grad():a=m(data)['x_hat_final'];m(batch(3));b=m(data)['x_hat_final']
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_direction_pairs_tie_fusion_sparse_and_candidate(self):
        m=model('R10');back=m.main_branch
        self.assertEqual(back.direction_allowed_pairs.sum().item(),19)
        for logits in (torch.zeros(3,8),torch.randn(3,8),torch.tensor([[10.,1.,9.,2.,8.,0.,7.,3.]])):
            chosen=back._select_native_topk(logits).sort(-1).values
            for pair in chosen.tolist():
                i=(back.pair_indices==torch.tensor(pair)).all(-1).nonzero().item()
                self.assertTrue(back.direction_allowed_pairs[i])
            native=logits.topk(2,-1).indices.sort(-1).values
            for i,pair in enumerate(native):
                ix=(back.pair_indices==pair).all(-1).nonzero().item()
                if back.direction_allowed_pairs[ix]:torch.testing.assert_close(chosen[i],pair)
        for router in back.routers:
            torch.nn.init.zeros_(router[-1].weight)
            router[-1].bias.data.copy_(torch.tensor([10.,1.,9.,2.,8.,0.,7.,3.]))
        calls=[];handles=[e.register_forward_pre_hook(lambda m,args,eid=i:calls.append((eid,args[0].shape[0]))) for i,e in enumerate(back.routed_experts())]
        out=m(batch())
        for h in handles:h.remove()
        weights=out['coe']['route_weights'];self.assertTrue(((weights>0).sum(-1)==2).all())
        torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights.sum(-1)))
        self.assertEqual(sum(n for _,n in calls),16);self.assertEqual(set(e for e,_ in calls),{0,7})
        expected=torch.tensor([10.,3.]).softmax(0)
        torch.testing.assert_close(weights[0,0,[0,7]],expected)
        logits=out['coe']['route_logits'];pairs=back.pair_indices
        probs=(logits[:,:,pairs[:,0]]+logits[:,:,pairs[:,1]]).float().softmax(-1)
        torch.testing.assert_close(out['coe']['pair_probs'],probs)
        out['x_hat_final'].square().mean().backward()
        for i,e in enumerate(back.routed_experts()):
            if i not in (0,7):self.assertTrue(all(p.grad is None for p in e.parameters()))

    def test_triple_exact_distinct_components(self):
        for shape in ((12,32,32),(12,16,8),(3,8,8)):
            count=round(np.prod(shape)*.4)
            a,c=make_triple_mask(shape,count,np.random.default_rng(7),True)
            b=make_diverse_mask(shape,.4,'node_plus_time_plus_space',np.random.default_rng(7))
            np.testing.assert_array_equal(a,b)
            self.assertEqual(int((a==0).sum()),count)
            union=sum(x.astype(int) for x in c.values())
            np.testing.assert_array_equal(union,1-a)
            self.assertTrue(all(abs(x.sum()-count/3)<1 for x in c.values()))

    def test_scale_metrics_exact_sample_weight_and_family(self):
        m=model('R7').eval();acc=CoERoutingMetricAccumulator()
        for n,coarse in ((3,True),(1,False)):
            for h in m.main_branch.scale_routers:h[-1].bias.data.copy_(torch.tensor([-2.,2.] if coarse else [2.,-2.]))
            out=m(batch(n));out['coe']['mask_family']=torch.zeros(n,dtype=torch.long)
            acc.update(out['coe'])
        r=acc.compute();self.assertEqual(r['coe_scale_path_cccc_fraction'],.75)
        self.assertEqual(r['coe_condition_family_random_point_scale_path_ffff_fraction'],.25)
        self.assertEqual(r['coe_coarse_rounds_mean'],3.)
        self.assertEqual(r['coe_expert_grid_equivalents'],3.5)

    def test_no_target_or_missing_payload_leak(self):
        data=batch();altered={**data,'x_f_gt':torch.full_like(data['x_f_gt'],1e8),'x_f_obs':torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))}
        for name in ('R4','R5','R6','R7','R8','R9','R10'):
            m=model(name).eval()
            with torch.no_grad():a=m(data);b=m(altered)
            torch.testing.assert_close(a['x_hat_final'],b['x_hat_final'],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
