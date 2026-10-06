import copy
import itertools
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/v24'))
from run_triscale_exploration import jobs
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.spatial_scale_coe import spatial_pool,spatial_resize
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


def config(name='T4',memory=False,independent=False):
    c=jobs(variants=(name,))[0]['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
    c['model']['coe']['triscale']['recent_memory']=memory
    if independent:c['model']['coe']['expert_sharing']='per_step'
    return c


def model(name='T4',**kwargs):
    torch.manual_seed(7);return DualBranchSTImputer.from_config(config(name,**kwargs))


def batch(n=2):
    x=torch.randn(n,2,3,8,8);mask=(torch.rand_like(x)>.4).float()
    return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x}


def force_path(m,path):
    for sid,head in zip(path,m.main_branch.scale_routers):head[-1].bias.data.fill_(-5);head[-1].bias.data[sid]=5


class TriScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_initial_fixed_free_equal_and_gradients(self):
        data=batch()
        for fixed,free in [('T1','T2'),('T3','T4')]:
            a=model(fixed).eval();b=model(free).eval()
            with torch.no_grad():oa=a(data);ob=b(data)
            torch.testing.assert_close(oa['x_hat_final'],ob['x_hat_final'],rtol=0,atol=0)
            self.assertEqual(ob['coe']['triscale_choices'].tolist(),[[2,1,0,0]]*2)
            b.train();out=b(data);(out['x_hat_final']-data['x_f_gt']).square().mean().backward()
            for h in b.main_branch.scale_routers:
                g=h[-1].bias.grad;self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum(),0)

    def test_81_paths_true_sparse_and_area(self):
        m=model('T4').train();data=batch();back=m.main_branch
        for router in back.routers:router[-1].weight.data.zero_();router[-1].bias.data.copy_(torch.arange(8,0,-1))
        for path in itertools.product(range(3),repeat=4):
            force_path(m,path);calls=[]
            hs=[e.register_forward_pre_hook(lambda m,a,eid=i:calls.append((eid,a[0].shape[0],a[0].shape[-1]))) for i,e in enumerate(back.routed_experts())]
            out=m(data)
            for h in hs:h.remove()
            self.assertEqual(out['coe']['triscale_choices'].tolist(),[list(path)]*2)
            self.assertEqual(sum(n for _,n,_ in calls),16);self.assertEqual(set(i for i,_,_ in calls),{0,1})
            for sid,w in enumerate((8,4,2)):self.assertEqual(sum(n for _,n,width in calls if width==w),path.count(sid)*4)
            acc=CoERoutingMetricAccumulator();acc.update(out['coe']);r=acc.compute()
            self.assertAlmostEqual(r['coe_expert_grid_equivalents'],2*sum(1/4**i for i in path))
        m.zero_grad();out=m(data);out['x_hat_final'].square().mean().backward()
        for e in back.routed_experts()[2:]:self.assertTrue(all(p.grad is None for p in e.parameters()))

    def test_restore_formula_and_fine_direct_exact(self):
        b=model('T4').main_branch;original=torch.randn(2,8,3,8,8,requires_grad=True)
        for f in (1,2,4):
            update=torch.randn(2,8,3,8//f,8//f,requires_grad=True)
            got=b._restore_resolution(update,original,f)
            expected=update if f==1 else original-spatial_resize(spatial_pool(original,f),(8,8))+spatial_resize(update,(8,8))
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
            if f==1:self.assertIs(got,update)
            got.square().mean().backward();self.assertTrue(torch.isfinite(update.grad).all())
        data=batch();a=model('T2').eval();b=model('T4').eval();force_path(a,[0]*4);force_path(b,[0]*4)
        torch.testing.assert_close(a(data)['x_hat_final'],b(data)['x_hat_final'],rtol=0,atol=0)

    def test_exploration_rng_resume_eval_and_epoch9(self):
        m=model('T5').train();data=batch(60);state=torch.random.get_rng_state().clone()
        a=m(data);torch.testing.assert_close(state,torch.random.get_rng_state())
        paths=a['coe']['triscale_choices'];self.assertEqual(len(set(map(tuple,paths.tolist()))),12)
        self.assertTrue(all(sorted(p)==[0,0,1,2] for p in paths.tolist()))
        saved=copy.deepcopy(m.state_dict());n=model('T5').train();n.load_state_dict(saved)
        torch.testing.assert_close(m(data)['coe']['triscale_choices'],n(data)['coe']['triscale_choices'])
        rng=m.main_branch._explore_rng_state.clone();m.eval();force_path(m,[2]*4)
        self.assertEqual(m(data)['coe']['triscale_choices'].tolist(),[[2]*4]*60)
        m.train();m.main_branch.set_routing_epoch(9)
        self.assertEqual(m(data)['coe']['triscale_choices'].tolist(),[[2]*4]*60)
        torch.testing.assert_close(rng,m.main_branch._explore_rng_state)

    def test_recent_memory_original_detail_and_isolation(self):
        data=batch();a=model('T4').eval();b=model('T4',memory=True).eval()
        torch.testing.assert_close(a(data)['x_hat_final'],b(data)['x_hat_final'],rtol=0,atol=0)
        back=b.main_branch;back.memory_gate[-1].bias.data.fill_(.2);captured=[]
        original=back._restore_resolution
        original_input=back._expert_input
        expected=[]
        def capture_input(hidden,*args,**kwargs):
            expected[:]=[hidden]
            return original_input(hidden,*args,**kwargs)
        back._expert_input=capture_input
        def check(update,hidden,factor):
            # Dispatch must supply the uncorrected hidden, not the memory input.
            torch.testing.assert_close(hidden,expected[0],rtol=0,atol=0)
            captured.append(hidden)
            return original(update,hidden,factor)
        back._restore_resolution=check
        out=b(data);hist=out['coe']['round_memory_weights']
        for step in range(1,4):self.assertEqual(hist[0,step].argmax().item(),step-1)
        self.assertIsNone(back._previous_hidden);self.assertIsNone(back._original_hidden)
        out['x_hat_final'].square().mean().backward();self.assertGreater(back.memory_gate[-1].weight.grad.abs().sum(),0)
        before=b(data)['x_hat_final'];b(batch(3));after=b(data)['x_hat_final'];torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertTrue(captured)

    def test_independent_and_no_target_leak(self):
        m=model('T5',independent=True).eval();data=batch()
        pools=[{id(p) for e in m.main_branch.routed_experts(s) for p in e.parameters()} for s in range(4)]
        self.assertFalse(pools[0]&pools[1]);a=m(data)
        changed={**data,'x_f_gt':torch.full_like(data['x_f_gt'],1e8),'x_f_obs':torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))}
        torch.testing.assert_close(a['x_hat_final'],m(changed)['x_hat_final'],rtol=0,atol=0)
        a['x_hat_final'].square().mean().backward()
        for step in range(4):self.assertTrue(any(p.grad is not None for e in m.main_branch.routed_experts(step) for p in e.parameters()))

    def test_exact_metrics_conditions_and_merge(self):
        m=model('T2').eval();a=CoERoutingMetricAccumulator();parts=[]
        for n,path in [(3,[2]*4),(1,[0]*4)]:
            force_path(m,path);out=m(batch(n));out['coe']['mask_family']=torch.zeros(n,dtype=torch.long)
            a.update(out['coe']);part=CoERoutingMetricAccumulator();part.update(out['coe']);parts.append(part)
        merged=CoERoutingMetricAccumulator()
        for p in parts:merged.merge(p)
        r=a.compute();self.assertEqual(r,merged.compute());self.assertEqual(r['coe_scale_path_cccc_fraction'],.75)
        self.assertEqual(r['coe_condition_family_random_point_scale_path_ffff_fraction'],.25)
        self.assertEqual(r['coe_expert_grid_equivalents'],2.375);self.assertEqual(r['coe_expert_execution_count'],8)
        self.assertEqual(len([k for k in r if k.startswith('coe_scale_path_')]),81)

if __name__=='__main__':unittest.main()
