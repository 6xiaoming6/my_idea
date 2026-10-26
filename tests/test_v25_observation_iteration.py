import copy
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'tests')]
import test_v25_combination_aux as fixtures
from observation_iteration_config import make_job, CANDIDATES, select_candidate
from stmoe_imputer.models.observation_regions import ObservationRegions
from stmoe_imputer.models.observed_consistency_coe import _blocks, _expand_cells
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from preflight_taxibj_input_communication import check_consistency


class IterationTests(unittest.TestCase):
    def test_zero_half_delta_stays_finite(self):
        from stmoe_imputer.models.observation_iteration_coe import ObservationIterationCoE
        h=torch.ones(2,8,3,8,8,dtype=torch.float16)
        d=torch.zeros_like(h,requires_grad=True)
        out=ObservationIterationCoE.rms_match(d,h)
        self.assertEqual(float(out.abs().sum()),0.)
        # Real gate is bounded/zero at initialization; forward must never NaN.
        self.assertTrue(bool(torch.isfinite(out).all()))
    def setUp(self):
        self.f=fixtures.CombinationAuxTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def job(self,method):
        j=self.f.job('I37');spec=make_job(method)['config']['model']['coe'].get('observation_iteration')
        if spec:j['config']['model']['coe']['observation_iteration']=spec
        return j

    def ledger(self,batch=None):
        b=self.f.batch if batch is None else batch
        return ObservationRegions([(b['x_'+s+'_obs']/f**2,b['m_'+s]) for s,f in zip('fmc',(1,2,4))])

    def test_region_algebra_and_hidden_poison(self):
        b=self.f.batch;l=self.ledger();p=torch.randn_like(b['x_f_gt']);d=torch.randn_like(p)
        self.assertTrue(bool((sum(m.int() for m in l.owners)<=1).all()))
        torch.testing.assert_close(l.project(l.project(d)),l.project(d),atol=1e-6,rtol=1e-6)
        for m,f in zip(l.owners,(1,2,4)):
            torch.testing.assert_close(_blocks(l.detail(d)*m,f).sum(-1),torch.zeros_like(_blocks(d,f).sum(-1)),atol=1e-6,rtol=0)
        for e in l.residuals(b['x_f_gt']):torch.testing.assert_close(e,torch.zeros_like(e),atol=1e-4,rtol=0)
        dirty=copy.deepcopy(b)
        for s in 'fmc':dirty['x_'+s+'_obs'][~dirty['m_'+s].bool()]=float('nan')
        for x,y in zip(l.residuals(p),self.ledger(dirty).residuals(p)):torch.testing.assert_close(x,y,rtol=0,atol=0)
        for v in (0,1):
            edge=copy.deepcopy(b)
            for s in 'fmc':edge['m_'+s].fill_(v)
            ll=self.ledger(edge)
            self.assertTrue(bool(torch.isfinite(ll.project(d)).all()))
            if not v:self.assertEqual(float(ll.project(d).abs().sum()),0)
        previous=torch.randn_like(p);current=torch.randn_like(p);read=torch.randn_like(p)
        innovation=l.history_innovation(read,current,previous);recent=l.detail(current-previous)
        torch.testing.assert_close((innovation*recent).sum((2,3,4)),torch.zeros_like(p[:,:,0,0,0]),atol=1e-4,rtol=0)
        torch.testing.assert_close(l.project(innovation),torch.zeros_like(p),atol=1e-6,rtol=0)
        self.assertEqual(float(l.history_innovation(current,current,previous).abs().sum()),0)
        for sid,f in ((1,2),(2,4)):
            old=torch.randn(len(p),8,*p.shape[2:]);v=torch.randn(len(p),8,p.shape[2],p.shape[3]//f,p.shape[4]//f)
            new=l.update(old,v,sid);owner=l.owners[sid].all(1,keepdim=True);n=_blocks(owner.float(),f).sum(-1)
            reduce=lambda h:_blocks(h*owner,f).sum(-1)/n.clamp_min(1)
            null=lambda h:h-_expand_cells(reduce(h),f)*owner
            torch.testing.assert_close(null(new),null(old),atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(reduce(new)*(n>0),v*(n>0),atol=1e-6,rtol=1e-6)

    def test_old_off_parity(self):
        j=self.f.job('I37');a=self.f.helper.model(j).eval()
        j['config']['model']['coe']['observation_iteration']={'enabled':False}
        b=self.f.helper.model(j).eval()
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        oa,ob=a(self.f.batch),b(self.f.batch)
        torch.testing.assert_close(oa['x_hat_main'],ob['x_hat_main'],rtol=0,atol=0)
        torch.testing.assert_close(compute_coe_loss(oa,self.f.batch,j['config'])[0],compute_coe_loss(ob,self.f.batch,j['config'])[0],rtol=0,atol=0)

    def test_sum_mean_and_immutable_observations(self):
        j=self.job('I51');a=self.f.helper.model(j).eval();b=copy.deepcopy(self.f.batch)
        original={k:v.clone() for k,v in b.items() if torch.is_tensor(v)}
        out=a(b)
        for k,v in original.items():torch.testing.assert_close(v,b[k],rtol=0,atol=0)
        meanjob=copy.deepcopy(j);meanjob['config']['data']['scales']['pooling_mode']='mean'
        meanmodel=self.f.helper.model(meanjob).eval();meanmodel.load_state_dict(a.state_dict())
        for s,f in (('m',2),('c',4)):b['x_'+s+'_obs']/=f*f
        torch.testing.assert_close(meanmodel(b)['x_hat_main'],out['x_hat_main'],atol=1e-5,rtol=1e-5)

    def test_refresh_relay_zero_innovation_and_gradients(self):
        j=self.job('I51');m=self.f.helper.model(j).train();branch=m.main_branch
        native=branch._native;seen=[]
        def capture(c,h,p,sid,ids):
            seen.append((c['exploration_step'],p.detach().clone(),c['last_prediction'].detach().clone(),c['completion'].detach().clone()))
            return native(c,h,p,sid,ids)
        branch._native=capture
        with torch.no_grad():
            branch.x_history.inject[-1].bias.fill_(.37);branch.x_history.inject[-1].weight.normal_(0,.01)
        output=m(self.f.batch)
        self.assertEqual(float(output['coe']['diagnostics']['x_step2_pool_delta_abs']),0.)
        self.assertTrue(any(not torch.equal(p,current) for _,p,current,_ in seen))
        for step,p,current,completion in seen:
            torch.testing.assert_close(completion,torch.where(self.f.batch['m_f'].bool(),self.f.batch['x_f_obs'],current),rtol=0,atol=0)
        compute_coe_loss(output,self.f.batch,j['config'])[0].backward()
        for name in ('x_observation_delta','x_observation_scale','x_observation_expert'):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in getattr(branch,name).parameters()),name)

    def test_selection_is_validation_only_and_fallback(self):
        from observation_iteration_config import protocols
        _,dev,_=protocols();self.assertEqual([x['seed'] for x in dev['evaluations'].values()],list(range(2026101701,2026101707)))
        def record(id,ood,split='val'):
            return dict(split=split,sets={k:dict(metrics=dict(mae=id if k=='in_distribution' else ood,rmse=ood*2)) for k in dev['evaluations']})
        ref=record(10,10);params={'I42':1,'I43':2}
        selected=select_candidate(ref,{'I42':record(10.4,1),'I43':record(10.2,9)},params)
        self.assertEqual(selected['selected'],'I43');self.assertTrue(selected['qualified_for_replication'])
        bad=select_candidate(ref,{'I42':record(10.4,1),'I43':record(10.5,.5)},params)
        self.assertEqual(bad['selected'],'I42');self.assertFalse(bad['qualified_for_replication'])
        with self.assertRaises(ValueError):select_candidate(ref,{'I42':record(10,1,'test')},{'I42':1})

    def test_variants_forward_gradients_and_causality(self):
        for name in CANDIDATES+['I52']:
            with self.subTest(method=name):
                j=self.job(name);m=self.f.helper.model(j).train();b=self.f.batch;calls=[]
                hooks=[e.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0]))) for e in m.main_branch.routed_experts()]
                try:out=m(b)
                finally:
                    for hook in hooks:hook.remove()
                self.assertEqual(sum(calls),8*len(b['m_f']))
                check_consistency(out,b,j['config']);choices=out['coe']['triscale_choices']
                self.assertTrue(bool((choices[:,-1]==0).all()))
                if name=='I42':self.assertTrue(bool((choices==choices.new_tensor([1,1,0,0])).all()))
                loss,_=compute_coe_loss(out,b,j['config']);loss.backward()
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
                if name in ('I47','I50','I51'):
                    self.assertEqual(float(out['coe']['diagnostics']['observation_step2_history_innovation_abs']),0)
                m.eval();clean=m(b)['x_hat_main'];dirty=copy.deepcopy(b)
                for s in 'fmc':dirty['x_'+s+'_obs'][~dirty['m_'+s].bool()]=float('nan')
                dirty['x_f_gt'].fill_(123456.)
                torch.testing.assert_close(m(dirty)['x_hat_main'],clean,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
