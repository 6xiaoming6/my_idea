"""Fixed-grid latent CoE. Original I37 registry/defaults remain unchanged.

All H0..H4 use one spatial grid; native observations stay immutable. History
retains causal, detached F predictions, pooled to the latent grid at read time.
No observed value or prediction is reclassified as an observation.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from .observed_consistency_coe import FreeScaleHistoryConsistencyCoE
from .four_direction_coe import FourDirectionCoE
from .id_priority_coe import IDPriorityCoE
from .temporal_spatial_coe import TemporalSpatialCoE
from ..utils.deterministic import pool, resize
from ..performance import enabled as perf, diagnostics as collect_diagnostics


def grid_map(x, size):
    size = tuple(size)
    if x.shape[-2:] == size:
        return x
    h, w = x.shape[-2:]
    if h >= size[0] and w >= size[1] and h % size[0] == 0 and w % size[1] == 0:
        factor = h // size[0]
        if factor == w // size[1]:
            return pool(x, factor)
    return resize(x, size)


def pack(x, f):
    b,c,t,h,w = x.shape
    return x.reshape(b,c,t,h//f,f,w//f,f).permute(0,1,4,6,2,3,5).reshape(b,c*f*f,t,h//f,w//f)


def unpack(x, f):
    b,c,t,h,w = x.shape
    return x.reshape(b,c//(f*f),f,f,t,h,w).permute(0,1,4,5,2,6,3).reshape(b,c//(f*f),t,h*f,w*f)


class GridBridge(nn.Module):
    """Pool/interpolate, identity-initialized channel map, or trainable subcells."""
    def __init__(self, dim, source, target, mode):
        super().__init__()
        self.source, self.target, self.mode = tuple(source), tuple(target), mode
        self.factor = max(source[0],target[0]) // min(source[0],target[0])
        self.down = source[0] > target[0]
        self.transform = nn.Identity()
        if mode == 'learned':
            self.transform = nn.Conv3d(dim,dim,1,bias=False)
            nn.init.zeros_(self.transform.weight)
        elif mode == 'subcell' and source != target:
            f = self.factor
            self.transform = nn.Conv3d(dim*f*f if self.down else dim,
                                       dim if self.down else dim*f*f,1,bias=False)
            nn.init.zeros_(self.transform.weight)
            with torch.no_grad():
                for c in range(dim):
                    if self.down:
                        self.transform.weight[c,c*f*f:(c+1)*f*f,0,0,0] = 1/(f*f)
                    else:
                        self.transform.weight[c*f*f:(c+1)*f*f,c,0,0,0] = 1

    def forward(self,x):
        if tuple(x.shape[-2:]) != self.source:
            raise ValueError(f'Bridge expected {self.source}, got {x.shape[-2:]}')
        if self.mode == 'subcell' and self.source != self.target:
            return self.transform(pack(x,self.factor)) if self.down else unpack(self.transform(x),self.factor)
        y = grid_map(x,self.target)
        return y + self.transform(y) if self.mode == 'learned' else y


class LatentSpaceCoE(FreeScaleHistoryConsistencyCoE):
    @classmethod
    def from_config(cls,cfg):
        spec = copy.deepcopy(cfg['model']['coe']['latent_space'])
        allowed = {'factor','mapping','return_mode','delta','history','path'}
        if set(spec) != allowed or spec['factor'] not in (1,2,4):
            raise ValueError('Explicit complete latent specification required')
        if spec['mapping'] not in ('linear','learned','subcell') or spec['return_mode'] not in ('replace','residual'):
            raise ValueError('Unknown latent map/update')
        if spec['path'] not in ('free_last_f','CMFF') or any(type(spec[k]) is not bool for k in ('delta','history')):
            raise ValueError('Invalid latent routing/communication')
        m = super().from_config(cfg)
        m.latent_spec = spec
        main = cfg['model']['main']
        m.fine_size = (main['h'],main['w'])
        m.latent_size = tuple(v//spec['factor'] for v in m.fine_size)
        if any(v % 4 for v in m.fine_size):
            raise ValueError('F grid must be divisible by 4')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg['seed'])+114001)
            native = [tuple(v//f for v in m.fine_size) for f in (1,2,4)]
            m.to_latent = nn.ModuleList([GridBridge(m.dim,s,m.latent_size,spec['mapping']) for s in native])
            m.to_native = nn.ModuleList([GridBridge(m.dim,m.latent_size,s,spec['mapping']) for s in native])
        if not spec['delta']:
            m.communication = 'none'
            m.delta_steps = ()
            m.history_gate.requires_grad_(False)
        if not spec['history']:
            m.x_history.requires_grad_(False)
        if spec['path'] == 'CMFF':
            m.scale_policy = 'fixed'
            for head in m.scale_heads:
                head.requires_grad_(False)
        return m

    def output_hidden(self,h):
        return self.to_native[0](h)

    def decode_state(self,h):
        return self.decoder(self.output_hidden(h))

    def summary(self,h,mask):
        return FourDirectionCoE.summary(self,h,grid_map(mask,h.shape[-2:]))

    def _router_features(self,hidden,values,support_summary,change,missing,pattern_summary=None):
        # Only the hidden-state missing-weighted summary needs a new grid.
        # F value summaries and global support statistics retain I37 semantics.
        point_missing = grid_map(missing.mean(1,keepdim=True),hidden.shape[-2:])
        fraction = missing.mean((1,2,3,4)).unsqueeze(1)
        return torch.cat((hidden.mean((2,3,4)), self._missing_pool(hidden,point_missing),
            values.mean((2,3,4)),self._missing_pool(values,missing),support_summary,
            change.mean((2,3,4)),self._missing_pool(change,missing),fraction,
            (fraction==0).to(hidden.dtype)),1)

    def _initialize(self,x,mask):
        # Reuse the canonical context but map each native encoding BEFORE fusion.
        c = FourDirectionCoE._initialize(self,x,mask)
        encodings = [self.to_latent[0](c['h'])]
        pyramid = []
        for sid,((values,native_mask),factor) in enumerate(zip(self._observations,(1,2,4))):
            pos = pool(c['pos'],factor) if factor>1 else c['pos']
            support = c['support'] if sid==0 and perf(self,'reuse_fine_support') else self._observation_support(native_mask)
            pyramid.append((values,native_mask,support,pos))
            if sid:
                encoded = self.encoder(torch.cat((values,native_mask,support,pos),1))
                encoded = encoded*native_mask.mean(1,keepdim=True)
                encodings.append(self.to_latent[sid](encoded))
        h = self.observation_fusion(torch.cat(encodings,1))
        pred = self.decode_state(h)
        unset = torch.full((len(x),),-1,dtype=torch.long,device=x.device)
        observed = torch.where(c['mask'].bool(),c['x'].float().square(),0.)
        scale = (observed.sum((2,3,4),keepdim=True)/c['mask'].sum((2,3,4),keepdim=True).clamp_min(1)).sqrt().clamp_min(1).detach()
        c.update(h=h,h0=h,prev=h,initial_prediction=pred,last_prediction=pred,
            completion=torch.where(c['mask'].bool(),c['x'],pred),observed_pyramid=pyramid,
            output_scale=unset,previous_output_scale=unset,x_pool=[],x_scale=scale)
        return c

    def _memory(self,c):
        enabled = c.get('exploration_step',-1)+1 in self.delta_steps
        corrected,route,gate = IDPriorityCoE._memory(self,c if enabled else dict(c,has_prev=False))
        c['w_stats'] = {k:c['h'].new_zeros(()) for k in
            ('feedback_abs','feedback_route_abs','visible_residual_abs','feedback_coverage')}
        stats = {'pool_size':c['h'].new_tensor(len(c['x_pool'])), 'pool_delta_abs':c['h'].new_zeros(())}
        if c['x_pool'] and self.latent_spec['history']:
            history = [grid_map(p,self.latent_size) for p in c['x_pool']]
            delta,weights = self.x_history(c['h'],history,grid_map(c['mask'],self.latent_size),c['x_scale'])
            delta = delta*self.history_gain if self.history_gain != 1 else delta
            corrected = corrected+delta
            if collect_diagnostics(self):
                stats['pool_delta_abs'] = delta.detach().float().abs().mean()
                for i in range(len(history)):
                    stats[f'pool_weight_P{i+1}'] = weights[:,i].detach().mean()
        c['x_stats'] = stats
        return corrected,route,gate

    def _execute(self,c,corrected,weights,scale_weights,step):
        out = torch.zeros_like(c['h'])
        for sid,factor in enumerate((1,2,4)):
            ids = (scale_weights[:,sid].detach()!=0).nonzero().flatten()
            if not ids.numel():
                continue
            values,mask,support,pos = c['observed_pyramid'][sid]
            current = corrected.index_select(0,ids)
            native = self.to_native[sid](current)
            fallback = c['completion'].index_select(0,ids)
            if factor>1:
                fallback = pool(fallback,factor)
            nm = mask.index_select(0,ids)
            completion = torch.where(nm.bool(),values.index_select(0,ids),fallback)
            z = self.state_norm(self.state_projection(torch.cat((native,completion,nm,
                support.index_select(0,ids),pos.index_select(0,ids)),1)))
            update = TemporalSpatialCoE._dispatch_weighted(self,z,weights.index_select(0,ids),step)
            if self.latent_spec['return_mode']=='residual':
                update = current+self.to_latent[sid](update-native)
            else:
                update = self.to_latent[sid](update)
            update = update*scale_weights[ids,sid,None,None,None,None].to(update.dtype)
            out = out.index_add(0,ids,update.to(out.dtype))
        c['w_stats']['adapter_abs'] = out.new_zeros(())
        return out,None,[]

    def _round(self,c,step,*args,**kwargs):
        if tuple(c['h'].shape[-2:]) != self.latent_size:
            raise RuntimeError('Round input escaped the latent grid')
        new,row = super()._round(c,step,*args,**kwargs)
        if tuple(new['h'].shape[-2:]) != self.latent_size:
            raise RuntimeError('Round output escaped the latent grid')
        if not self.latent_spec['history']:
            new['x_pool'] = []
        return new,row

    def forward(self,*args,**kwargs):
        out = super().forward(*args,**kwargs)
        d = out['coe']['diagnostics']
        for key,value in (('height',self.latent_size[0]),('width',self.latent_size[1]),
                          ('channels',self.dim),('history_enabled',self.latent_spec['history']),
                          ('delta_enabled',self.latent_spec['delta'])):
            d['latent_'+key] = out['x_hat_main'].new_tensor(float(value))
        return out
