"""W-series: fixed U02 backbone with deterministic scales and opt-in extensions."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .id_priority_coe import IDPriorityCoE
from .temporal_spatial_coe import TemporalSpatialCoE
from .four_direction_coe import head
from ..utils.deterministic import pool,resize


def box_sum(x):
    return F.conv3d(x,torch.ones(x.shape[1],1,3,3,3,device=x.device,dtype=x.dtype),padding=1,groups=x.shape[1])


def propagate(error,mask,local=True):
    """The input error is already masked. No data from hidden positions enters."""
    if not local:return error,mask,mask.sum(1,keepdim=True)>0
    count=box_sum(mask.float());capacity=box_sum(torch.ones_like(mask).float())
    value=torch.where(count>0,box_sum(error.float())/count.clamp_min(1),0.)
    return value,count/capacity.clamp_min(1),count.sum(1,keepdim=True)>0


class BackboneExplorationCoE(IDPriorityCoE):
    @classmethod
    def from_config(cls,cfg):
        m=super().from_config(cfg);m.w=copy.deepcopy(cfg['model']['coe']['backbone_exploration'])
        if m.communication!='delta' or m.bound!=.1 or m.scale_policy!='fixed':raise ValueError('W requires fixed CMFF U02')
        m.adapter=m.w.get('adapter','none');m.feedback=m.w.get('feedback','none')
        if m.adapter not in ('none','scale_id','round_id','scale_affine','round_affine','shared_rank','scale_rank','round_rank','coverage_rank'):raise ValueError('Unknown adapter')
        if m.feedback not in ('none','point','local','missing','missing_grad','router','both'):raise ValueError('Unknown feedback')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg.get('seed',7)+83001)
            count=4 if m.adapter.startswith('round') else 1 if m.adapter=='shared_rank' else 3
            if m.adapter.endswith('_id'):m.adapter_identity=nn.Parameter(torch.zeros(count,m.dim))
            if m.adapter.endswith('_affine'):
                m.adapter_gamma=nn.Parameter(torch.zeros(count,m.dim));m.adapter_beta=nn.Parameter(torch.zeros(count,m.dim))
            if m.adapter.endswith('_rank'):
                m.adapters=nn.ModuleList([nn.Sequential(nn.Conv3d(m.dim,8,1),nn.GELU(),nn.Conv3d(8,m.dim,1)) for _ in range(count)])
                for a in m.adapters:nn.init.zeros_(a[-1].weight);nn.init.zeros_(a[-1].bias)
                if m.adapter=='coverage_rank':m.coverage_affine=nn.Parameter(torch.zeros(3,2))
            torch.random.default_generator.manual_seed(cfg.get('seed',7)+83002)
            if m.feedback not in ('none','router'):
                m.feedback_input=nn.Sequential(nn.Conv3d(2*m.c_in,32,1),nn.GELU(),nn.Conv3d(32,m.dim,1))
                nn.init.zeros_(m.feedback_input[-1].weight);nn.init.zeros_(m.feedback_input[-1].bias)
            if m.feedback in ('router','both'):m.feedback_router=head(m.feature_dim+4*m.c_in,m.num_experts)
        return m

    def _feedback_field(self,c):
        mask=c['mask'].float();x=c['x'].float();pred=c['last_prediction'].float()
        if self.feedback!='missing_grad':pred=pred.detach()
        scale=(torch.where(mask.bool(),x.square(),0.).sum((2,3,4),keepdim=True)/mask.sum((2,3,4),keepdim=True).clamp_min(1)).sqrt().clamp_min(1).detach()
        error=torch.where(mask.bool(),(x-pred)/scale,0.)
        return propagate(error,mask,self.feedback!='point')

    def _memory(self,c):
        corrected,route,gate=super()._memory(c)
        stats={k:c['h'].new_zeros(()) for k in ('feedback_abs','feedback_route_abs','visible_residual_abs','feedback_coverage')}
        if c['has_prev'] and self.feedback!='none':
            field,coverage,valid=self._feedback_field(c);h=c['h']
            stats['visible_residual_abs']=(c['x']-c['last_prediction']).detach().float().abs()[c['mask'].bool()].mean() if c['mask'].bool().any() else h.new_zeros(())
            stats['feedback_coverage']=valid.float().mean()
            if hasattr(self,'feedback_input'):
                rms=h.detach().float().square().mean((2,3,4),keepdim=True).sqrt()
                delta=.05*rms*self.feedback_input(torch.cat((field,coverage),1)).float().tanh()*valid
                if self.feedback in ('missing','missing_grad','both'):delta=delta*(1-c['mask']).mean(1,keepdim=True)
                corrected=corrected+delta.to(corrected.dtype);stats['feedback_abs']=delta.detach().abs().mean()
            if hasattr(self,'feedback_router'):
                features=self._router_features(h,c['completion'],c['ss'],torch.zeros_like(c['initial_prediction']),1-c['mask'],None)
                descriptors=torch.cat((features,self.summary(field,c['mask']),self.summary(coverage,c['mask'])),1)
                route=.1*self.feedback_router(descriptors).tanh();stats['feedback_route_abs']=route.detach().abs().mean()
        c['w_stats']=stats
        return corrected,route,gate

    def adapt(self,z,coverage,sid,step):
        index=step if self.adapter.startswith('round') else 0 if self.adapter=='shared_rank' else sid
        if self.adapter=='none':return z
        if self.adapter.endswith('_id'):return z+self.adapter_identity[index][None,:,None,None,None].to(z.dtype)
        if self.adapter.endswith('_affine'):
            return z+self.adapter_gamma[index][None,:,None,None,None].to(z.dtype)*z+self.adapter_beta[index][None,:,None,None,None].to(z.dtype)
        delta=.1*self.adapters[index](z)
        if self.adapter=='coverage_rank':
            a,b=self.coverage_affine[index];delta=delta*(2*torch.sigmoid(a*coverage.mean(1,keepdim=True)+b)).to(delta.dtype)
        return z+delta

    def _execute(self,c,corrected,weights,scale_weights,step):
        sid=(2,1,0,0)[step];factor=(1,2,4)[sid]
        if not bool((scale_weights[:,sid]==1).all()):raise ValueError('W paths must stay CMFF')
        h=corrected;mask=c['mask'];completion=c['completion'];support=c['support'];pos=c['pos']
        if factor>1:
            coverage=pool(mask.float(),factor)
            values=pool(torch.where(mask.bool(),completion,0.).float(),factor)/coverage.clamp_min(1e-8)
            completion=torch.where(coverage>0,values,pool(completion,factor));mask=coverage
            h=pool(h,factor);support=pool(support,factor);pos=pool(pos,factor)
        z=self.state_norm(self.state_projection(torch.cat((h,completion,mask,support,pos),1)))
        adapted=self.adapt(z,mask,sid,step)
        c['w_stats']['adapter_abs']=(adapted-z).detach().float().abs().mean()
        update=TemporalSpatialCoE._dispatch_weighted(self,adapted,weights,step)
        return resize(update,c['h'].shape[-2:]) if factor>1 else update,None,[]

    def _round(self,c,step,*args,**kwargs):
        new,row=super()._round(c,step,*args,**kwargs)
        self._w_stats.append(c['w_stats'])
        return new,row

    def forward(self,*args,**kwargs):
        self._w_stats=[]
        try:
            out=super().forward(*args,**kwargs);out['w_spec']=self.w
            for i,stats in enumerate(self._w_stats,1):
                for k,v in stats.items():out['coe']['diagnostics'][f'w_step{i}_{k}']=v.detach()
            return out
        finally:del self._w_stats
