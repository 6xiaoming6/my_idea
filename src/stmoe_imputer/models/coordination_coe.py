"""V1-V4: CMFF bounded communication plus output-space region coordination."""
import copy
import torch
from torch import nn
from .id_priority_coe import IDPriorityCoE
from ..coordination import block_mean, coordinate


class CoordinationCoE(IDPriorityCoE):
    @classmethod
    def from_config(cls,cfg):
        m=super().from_config(cfg)
        spec=cfg['model']['coe']['coordination']
        m.coordination_mode=spec['mode'];m.coordination_strength=float(spec.get('strength',.1))
        m.coordination_aux_weight=float(spec.get('aux_weight',.05))
        if m.coordination_mode not in ('aux','uniform','learned') or m.scale_policy!='fixed' or m.communication!='delta' or m.bound!=.1:
            raise ValueError('Coordination requires fixed CMFF U02 with mode aux/uniform/learned')
        if not 0<=m.coordination_strength<=1 or not 0<=m.coordination_aux_weight<=1:
            raise ValueError('Invalid coordination coefficients')
        # Independent decoder copies start identically and consume no global RNG.
        m.region_heads=nn.ModuleDict({k:copy.deepcopy(m.decoder) for k in ('c','m')})
        if m.coordination_mode=='learned':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(cfg.get('seed',7))+82001)
                m.allocation_head=nn.Sequential(nn.Conv3d(m.dim+3*m.c_in+1,16,1),nn.GELU(),nn.Conv3d(16,m.c_in,1))
                nn.init.zeros_(m.allocation_head[-1].weight);nn.init.zeros_(m.allocation_head[-1].bias)
        return m

    def _record_native_scale(self,c,update,sid,step,selected):
        if step>=2:return
        label=('c','m')[step]
        if sid!=(2,1)[step] or len(selected)!=len(c['h']):
            raise RuntimeError('Region heads require full-batch fixed CMFF')
        c.setdefault('coordination_means',{})[label]=self.region_heads[label](update)

    def _round(self,c,step,*args,**kwargs):
        new,row=super()._round(c,step,*args,**kwargs)
        if step==3:
            raw=row['prediction'];pred=raw;stages={'raw':raw};allocations={}
            for label,factor in (('c',4),('m',2)):
                if self.coordination_mode!='aux':
                    mean=new['coordination_means'][label];scores=None
                    if self.coordination_mode=='learned':
                        current,_=block_mean(pred,~c['mask'].bool(),factor)
                        residual=(mean.float()-current).repeat_interleave(factor,-2).repeat_interleave(factor,-1)
                        feature=torch.cat((new['h'].float(),pred.float(),c['mask'].float(),residual,
                                           torch.full_like(pred[:,:1].float(),factor/4)),1)
                        scores=self.allocation_head(feature)
                    pred,record=coordinate(pred,c['mask'],mean,factor,self.coordination_strength,scores)
                    allocations[label]=record
                stages['after_'+label]=pred
            new['coordination']={'means':new['coordination_means'],'stages':stages,'allocations':allocations,'aux_weight':self.coordination_aux_weight}
            self._coordination_output['value']=new['coordination']
            # Keep round diagnostics as actual four expert rounds; output repair is separate.
        return new,row

    def forward(self,x_f,m_f,**kwargs):
        # Capture the final context locally; no tensors or histories survive a batch.
        captured={}
        # Avoid installing hooks/monkey patches: the forward below uses a transient
        # output collector cleared in finally, with the regular superclass loop.
        self._coordination_output=captured
        try:
            out=super().forward(x_f,m_f,**kwargs)
            info=captured['value']
            out['coordination']=info
            out['x_hat_main']=info['stages']['after_m']
            return out
        finally:
            del self._coordination_output
