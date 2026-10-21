"""Isolated M/S/G/P experiments. All per-window memory lives in a local context.

The inherited encoder/experts/native routers are constructed before extensions,
so common parameters exactly retain the B3 initialization. Probe rollouts resume
from detached contexts and never call the public forward or advance RNG buffers.
"""
from __future__ import annotations
import copy
import math
import time
from itertools import permutations
import torch
from torch import nn
from torch.nn import functional as F
from .temporal_spatial_coe import TemporalSpatialCoE, SUPPORT_FEATURE_NAMES
from .spatial_scale_coe import spatial_pool, spatial_resize, observed_pool


def head(inputs, outputs, hidden=64, bias=0.):
    net=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,outputs))
    nn.init.zeros_(net[-1].weight);nn.init.constant_(net[-1].bias,bias)
    return net


def detach_context(value, ids=None):
    if torch.is_tensor(value):
        return (value.index_select(0,ids) if ids is not None and value.ndim else value).detach()
    if isinstance(value,dict):return {k:detach_context(v,ids) for k,v in value.items()}
    if isinstance(value,list):return [detach_context(v,ids) for v in value]
    return value


class FourDirectionCoE(TemporalSpatialCoE):
    @classmethod
    def from_config(cls,cfg):
        m=super().from_config(cfg);m.spec=copy.deepcopy(cfg['model']['coe']['four_direction'])
        if (m.num_steps!=4 or m.top_k!=2 or m.pair_mode!='native' or m.routing_mode!='hard'
            or m.state_update_mode!='direct' or m.completion_feedback or m.use_shared
            or not m.use_routed or m.router_features!='legacy' or m.fusion_mode!='original'):
            raise ValueError('Four-direction suite requires the B3 native four-round direct protocol')
        m.memory=m.spec.get('memory','none');m.scale=m.spec.get('scale','fine');m.pair=m.spec.get('pair','native')
        m.teacher=m.spec.get('teacher','none');m.probe_request=None
        valid_memory=('none','residual','anchor','signed','positive','delta','delta_detach','router','both','message','gru')
        valid_scale=('fine','fixed','free','sample','teacher','history','bank','top2','fixed2','soft3','equal3','permutation')
        if m.memory not in valid_memory or m.scale not in valid_scale:raise ValueError('Invalid memory/scale mode')
        if m.pair not in ('native','equal','joint','joint_history','partner','bounded','joint_teacher_only'):raise ValueError('Invalid pair mode')
        d=m.dim;f=m.routers[0][0].normalized_shape[0];m.feature_dim=f
        if m.memory=='residual':m.keep_logits=nn.Parameter(torch.full((4,),math.log(.1/.9)))
        if m.memory in ('anchor','signed','positive','delta','delta_detach','both'):
            bias=math.atanh(.05) if m.memory=='signed' else math.log(.05/.95) if m.memory=='positive' else 0.
            m.history_gate=head(4*d,d,bias=bias)
        if m.memory in ('router','both'):m.history_router=head(6*d,m.num_experts)
        if m.memory=='message':
            m.message_net=nn.Sequential(nn.LayerNorm(4*d+2*m.num_experts+2),nn.Linear(4*d+2*m.num_experts+2,64),nn.GELU())
            m.message_route=nn.Linear(64,m.num_experts);m.message_film=nn.Linear(64,2*d)
            for h in (m.message_route,m.message_film):nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
        if m.memory=='gru':
            m.memory_gru=nn.GRUCell(4*d,64);m.gru_route=nn.Linear(64,m.num_experts);m.gru_film=nn.Linear(64,2*d)
            for h in (m.gru_route,m.gru_film):nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
        if m.scale not in ('fine',):
            # Identical common scale heads for all choices, including fixed controls.
            m.scale_heads=nn.ModuleList([head(f+4,3) for _ in range(4)])
            if m.scale in ('fixed','fixed2','equal3','permutation'):m.scale_heads.requires_grad_(False)
        if m.scale=='history':m.scale_history=nn.ModuleList([nn.Linear(6,3,bias=False) for _ in range(4)])
        if m.scale=='bank':
            m.bank_route=nn.ModuleList([head(6*d+3,3) for _ in range(4)])
            m.bank_gate=nn.ModuleList([head(4*d+1,d) for _ in range(3)])
        if m.scale=='history':
            for h in m.scale_history:nn.init.zeros_(h.weight)
        if m.pair in ('joint','joint_history','joint_teacher_only','partner'):
            extra=(6*d+2*m.num_experts+2) if m.pair=='joint_history' else m.num_experts if m.pair=='partner' else 0
            m.pair_heads=nn.ModuleList([head(f+extra,m.num_experts if m.pair=='partner' else len(m.pair_indices)) for _ in range(4)])
        if m.pair=='bounded':m.response_head=head(4*d+2,1)
        gen=torch.Generator().manual_seed(m.spec.get('route_seed',20261031))
        m.register_buffer('route_rng',gen.get_state());gen.manual_seed(m.spec.get('probe_seed',20261032))
        m.register_buffer('probe_rng',gen.get_state());m.register_buffer('batch_clock',torch.zeros((),dtype=torch.long))
        m.register_buffer('permutations',torch.tensor(sorted(set(permutations((2,1,0,0)))),dtype=torch.long))
        return m

    def summary(self,h,mask):
        return torch.cat((h.mean((2,3,4)),self._missing_pool(h,(1-mask).mean(1,keepdim=True))),1)

    def prepare_training_batch(self,batch):
        self.probe_request=None
        clock=int(self.batch_clock.item());self.batch_clock.add_(1)
        if self.teacher!='none' and clock%20==0:
            from ..losses import supervision_mask
            valid=supervision_mask(batch['x_f_gt'],batch['m_f'],batch.get('target_mask')).flatten(1).any(1)
            ids=valid.nonzero().flatten()[:4]
            if ids.numel():self.probe_request=(clock//20%4,ids)

    def _random(self,probs):
        g=torch.Generator();g.set_state(self.route_rng.cpu())
        ids=torch.multinomial(probs.detach().float().cpu(),1,generator=g).squeeze(1)
        self.route_rng.copy_(g.get_state().to(self.route_rng.device))
        return ids.to(probs.device)

    def _initialize(self,x,mask):
        if x.ndim!=5 or mask.shape[0]!=x.shape[0] or x.shape[-2]%4 or x.shape[-1]%4:
            raise ValueError('Expected matching [B,C,T,H,W] with spatial dimensions divisible by four')
        if not bool(((mask==0)|(mask==1)).all()):raise ValueError('Binary mask required')
        obs=mask.bool().expand_as(x)
        if not bool((torch.isfinite(x)|~obs).all()):raise ValueError('Nonfinite observed input')
        mask=obs.to(x.dtype);x=torch.where(obs,x,torch.zeros_like(x));support=self._observation_support(mask)
        miss=1-mask;sm=miss.repeat(1,len(SUPPORT_FEATURE_NAMES),1,1,1)
        ss=torch.cat((support.mean((2,3,4)),support.std((2,3,4),unbiased=False),support.amax((2,3,4)),self._missing_pool(support,sm)),1)
        pos=self._position(x);h=self.encoder(torch.cat((x,mask,support,pos),1));pred=self.decoder(h)
        ctx={'h':h,'h0':h,'prev':h,'has_prev':False,'x':x,'mask':mask,'support':support,'ss':ss,'pos':pos,
             'initial_prediction':pred,'completion':torch.where(obs,x,pred),'last_prediction':pred,
             'counts':x.new_zeros((len(x),3)),'last_scale':x.new_zeros((len(x),3)),
             'last_ids':x.new_zeros((len(x),2*self.num_experts)), 'last_within':x.new_zeros((len(x),2)),
             'message':x.new_zeros((len(x),64)),'gru':x.new_zeros((len(x),64)),
             'ages':x.new_zeros((len(x),3))}
        if self.scale=='bank':ctx['bank']=[spatial_pool(h,fac) for fac in (1,2,4)]
        if self.scale=='permutation' and self.training:
            start,end=self.spec.get('explore_range',[1,8])
            if start<=self.routing_epoch<=end:
                paths=self.spec.get('paths')
                options=self.permutations if paths is None else torch.tensor(paths,device=x.device)
                ids=self._random(x.new_ones((len(x),len(options))))
                ctx['random_path']=options.to(x.device).index_select(0,ids)
        return ctx

    def _memory(self,c):
        h=c['h'];past=c['prev'];mask=c['mask'];zero=h.new_zeros((len(h),self.dim))
        current=self.summary(h,mask);old=self.summary(past,mask);delta=h-past
        route=h.new_zeros((len(h),self.num_experts));corrected=h;gate=zero
        if self.memory=='anchor':
            gate=self.history_gate(torch.cat((current,self.summary(c['h0'],mask)),1)).tanh()
            corrected=h+gate[:,:,None,None,None]*(c['h0']-h)
        elif c['has_prev'] and self.memory in ('signed','positive','delta','delta_detach','both'):
            gate=self.history_gate(torch.cat((current,old),1))
            gate=gate.sigmoid() if self.memory=='positive' else gate.tanh()
            if self.memory in ('signed','positive'):change=-delta
            else:
                ratio=(h.detach().float().square().mean((2,3,4),keepdim=True).sqrt()/delta.detach().float().square().mean((2,3,4),keepdim=True).sqrt().clamp_min(1e-6))
                change=delta*ratio.to(delta.dtype)
                if self.memory=='delta_detach':change=change.detach()
            corrected=h+gate[:,:,None,None,None]*change
        if self.memory in ('router','both') and c['has_prev']:
            route=self.history_router(torch.cat((current,old,self.summary(delta,mask)),1))
        elif self.memory in ('message','gru') and c['has_prev']:
            msg=c['message'] if self.memory=='message' else c['gru']
            route=(self.message_route if self.memory=='message' else self.gru_route)(msg)
            scale,bias=(self.message_film if self.memory=='message' else self.gru_film)(msg).chunk(2,1)
            corrected=h*(1+scale[:,:,None,None,None])+bias[:,:,None,None,None]
        return corrected,route,gate

    def _scales(self,c,features,step,force=None,probe=False):
        b=len(features);prior_path=self.spec.get('fixed_path',[2,1,0,0]);mode=self.scale
        base=torch.cat((features,c['counts'].to(features.dtype)/4,features.new_full((b,1),(4-step)/4)),1)
        extra=None
        if mode=='fine':logits=features.new_zeros((b,3));choice=features.new_zeros((b,),dtype=torch.long)
        elif mode in ('fixed','permutation'):
            logits=features.new_zeros((b,3));choice=features.new_full((b,),prior_path[step],dtype=torch.long)
            if not probe and 'random_path' in c:choice=c['random_path'][:,step]
        else:
            logits=self.scale_heads[step](base).float()
            if mode in ('free','sample','teacher','history','bank'):
                prior=logits.new_zeros(3);prior[prior_path[step]]=1e-3;logits=logits+prior
            elif mode=='top2':
                prior=logits.new_zeros(3);prior[list(((2,1),(1,0),(2,0),(1,0))[step])]=1e-3;logits=logits+prior
            if mode=='history':
                delta=(c['h']-c['prev']).float();missing=(1-c['mask']).mean(1,keepdim=True)
                global_rms=delta.square().mean((1,2,3,4)).clamp_min(1e-12).sqrt()[:,None]
                local_rms=self._missing_pool(delta.square(),missing).mean(1).clamp_min(1e-12).sqrt()[:,None]
                w=c['last_within'].float();ent=-(w*w.clamp_min(1e-8).log()).sum(1,keepdim=True)
                extra=torch.cat((c['last_scale'],global_rms,local_rms,ent),1).to(base.dtype)
                logits=logits+self.scale_history[step](extra).float()
            elif mode=='bank':
                extra=torch.cat([self.summary(v,spatial_pool(c['mask'],fac)) for v,fac in zip(c['bank'],(1,2,4))]+[c['ages']/4],1)
                logits=logits+self.bank_route[step](extra).float()
            choice=logits.argmax(1)
        probs=logits.float().softmax(1)
        if mode in ('fine','fixed','permutation'):probs=F.one_hot(choice,3).float()
        if mode=='sample' and self.training and not probe and self.routing_epoch<=20:
            choice=self._random(.8*probs+.2/3)
        if mode in ('top2','fixed2'):
            ids=(logits.topk(2,1).indices if mode=='top2' else torch.tensor(((2,1),(1,0),(2,0),(1,0))[step],device=logits.device).expand(b,2))
            selected=F.one_hot(ids,3).any(1);weights=probs*selected if mode=='top2' else selected.float()
            weights=weights/weights.sum(1,keepdim=True);probs=weights if mode=='fixed2' else probs
        elif mode in ('soft3','equal3'):weights=probs if mode=='soft3' else torch.ones_like(probs)/3;probs=weights
        else:
            if force is not None:choice=force
            selected=F.one_hot(choice,3).float()
            weights=selected+(probs-probs.detach()) if self.training and not probe and mode not in ('fine','fixed','permutation') else selected
        active=weights.detach()!=0
        return weights,probs,active,{'base':base,'extra':extra,'logits':logits}

    def _pair_features(self,c,features,logits):
        if self.pair=='joint_history':
            return torch.cat((features,self.summary(c['h'],c['mask']),self.summary(c['prev'],c['mask']),
                self.summary(c['h']-c['prev'],c['mask']),c['last_ids'],c['last_within']),1)
        if self.pair=='partner':return torch.cat((features,F.one_hot(logits.topk(2,1).indices[:,0],self.num_experts).to(features.dtype)),1)
        return features

    def _pairs(self,c,features,logits,step,force=None,probe=False):
        p=logits.float().softmax(1);native=logits.topk(2,1).indices
        native_mask=F.one_hot(native,self.num_experts).any(1)
        native_weights=p*native_mask;native_weights=native_weights/native_weights.sum(1,keepdim=True).clamp_min(1e-8)
        left,right=self.pair_indices.unbind(1);base=logits.float()[:,left]+logits.float()[:,right]
        scores=base;pf=self._pair_features(c,features,logits)
        ids=native;coefficient=logits.new_ones((len(logits),))
        if self.pair in ('joint','joint_history','joint_teacher_only','partner'):
            correction=self.pair_heads[step](pf).float()
            if self.pair=='partner':
                scores=logits.float()+correction;primary=native[:,0];masked=scores.scatter(1,primary[:,None],-1e9)
                second=torch.where(masked.gather(1,native[:,1:]).squeeze(1)==masked.max(1).values,native[:,1],masked.argmax(1))
                ids=torch.stack((primary,second),1)
                prob=masked.softmax(1);selected_prob=prob.gather(1,second[:,None]).squeeze(1)
            else:
                scores=base+correction;best=scores.argmax(1)
                native_sorted=native.sort(1).values
                a,z=native_sorted.unbind(1);native_pid=a*(2*self.num_experts-a-1)//2+z-a-1
                best=torch.where(scores.gather(1,native_pid[:,None]).squeeze(1)==scores.max(1).values,native_pid,best)
                ids=self.pair_indices[best];prob=scores.softmax(1);selected_prob=prob.gather(1,best[:,None]).squeeze(1)
            if self.training and not probe and self.pair!='joint_teacher_only':coefficient=1+(selected_prob-selected_prob.detach())
        if force is not None:ids=force;coefficient=coefficient*0+1
        # Use exactly the native normalization at initialization, including ties.
        sel=F.one_hot(ids,self.num_experts).any(1);weights=p*sel;weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        if self.pair=='equal':weights=sel.to(p.dtype)/2
        ordered=weights.topk(2,1).indices.sort(1).values
        # Balance importance ALWAYS uses native candidate probabilities, never corrected pair scores.
        importance=self._pair_importance(logits,base.softmax(1),1.)
        return weights,ordered,importance,coefficient,{'features':pf,'scores':scores,'base_logits':logits,'native':native,'chosen':ids}

    def _execute(self,c,corrected,weights,scale_weights,step):
        out=torch.zeros_like(c['h']);response=None;bank=list(c.get('bank',[]));d=self.dim
        for sid,fac in enumerate((1,2,4)):
            selected=(scale_weights[:,sid].detach()!=0).nonzero().flatten()
            if not selected.numel():continue
            h=corrected.index_select(0,selected);mask=c['mask'].index_select(0,selected)
            completion=c['completion'].index_select(0,selected);support=c['support'].index_select(0,selected);pos=c['pos'].index_select(0,selected)
            if fac>1:
                values,coverage=observed_pool(completion,mask,fac)
                completion=torch.where(coverage>0,values,spatial_pool(completion,fac));mask=coverage
                h=spatial_pool(h,fac);support=spatial_pool(support,fac);pos=spatial_pool(pos,fac)
            if self.scale=='bank':
                cached=bank[sid].index_select(0,selected)
                gate=self.bank_gate[sid](torch.cat((self.summary(h,mask),self.summary(cached,mask),c['ages'].index_select(0,selected)[:,sid:sid+1]/4),1)).tanh()
                h=h+gate[:,:,None,None,None]*(cached-h)
            unified=self.state_norm(self.state_projection(torch.cat((h,completion,mask,support,pos),1)))
            w=weights.index_select(0,selected)
            if self.memory=='message' or self.pair=='bounded':
                ids=w.topk(2,1).indices
                a=self._dispatch(unified,ids[:,0],step);b=self._dispatch(unified,ids[:,1],step)
                within=w.gather(1,ids)
                desc=torch.cat((self.summary(a,mask),self.summary(b,mask),within),1)
                if self.pair=='bounded':
                    delta=.5*self.response_head(desc.detach()).tanh()
                    difference=within[:,1:2].clamp_min(1e-8).log()-within[:,:1].clamp_min(1e-8).log()
                    shift=torch.sigmoid(difference+delta)-torch.sigmoid(difference)
                    within=within+torch.cat((-shift,shift),1)
                update=a*within[:,0,None,None,None,None].to(a.dtype)+b*within[:,1,None,None,None,None].to(b.dtype)
                if self.memory=='message':
                    msg=torch.cat((desc[:,:4*d],F.one_hot(ids,self.num_experts).flatten(1).to(desc.dtype),within),1)
                    embedded=self.message_net(msg)
                    if response is None:response=embedded.new_zeros((len(out),64))
                    response=response.index_add(0,selected,embedded*scale_weights[selected,sid,None].to(embedded.dtype))
            else:update=super()._dispatch_weighted(unified,w,step)
            if self.scale=='bank':bank[sid]=bank[sid].index_copy(0,selected,update.to(bank[sid].dtype))
            restored=spatial_resize(update,c['h'].shape[-2:]) if fac>1 else update
            out=out.index_add(0,selected,(restored*scale_weights[selected,sid,None,None,None,None].to(restored.dtype)).to(out.dtype))
        return out,response,bank

    def _round(self,c,step,force_scale=None,force_pair=None,probe=False):
        h=c['h'];miss=1-c['mask']
        features=self._router_features(h,c['completion'],c['ss'],torch.zeros_like(c['initial_prediction']),miss,None)
        corrected,route,gate=self._memory(c);logits=self.routers[step](features)+route
        ew,ids,importance,coefficient,pair_record=self._pairs(c,features,logits,step,force_pair,probe)
        sw,sp,active,scale_record=self._scales(c,features,step,force_scale,probe)
        update,msg,bank=self._execute(c,corrected,ew,sw,step)
        next_h=update*coefficient[:,None,None,None,None].to(update.dtype)
        if self.memory=='residual':
            keep=self.keep_logits[step].sigmoid();next_h=keep*h+(1-keep)*next_h
        pred=self.decoder(next_h);change=torch.where(c['mask'].bool(),torch.zeros_like(pred),(pred-c['last_prediction']).abs())
        new=dict(c);new.update(h=next_h,prev=h,has_prev=True,last_prediction=pred,
            counts=c['counts']+active.to(c['counts'].dtype),last_scale=sw.detach(),
            last_ids=F.one_hot(ids,self.num_experts).flatten(1).to(h.dtype),last_within=ew.gather(1,ids),
            ages=torch.where(active,torch.zeros_like(c['ages']),c['ages']+1))
        if msg is not None:new['message']=msg
        if bank:new['bank']=bank
        if self.memory=='gru':new['gru']=self.memory_gru(torch.cat((self.summary(next_h,c['mask']),self.summary(next_h-h,c['mask'])),1),c['gru'].to(next_h.dtype))
        row={'prediction':pred,'change':change,'logits':logits,'probs':logits.float().softmax(1),'weights':ew,
             'ids':ids,'importance':importance,'scale_weights':sw,'scale_probs':sp,'active':active,
             'gate':gate,'input_delta':(corrected-h).detach().float().abs().mean(),
             'route_delta':route.detach().float().abs().mean(),'memory_norm':(c['message'] if self.memory=='message' else c['gru']).detach().float().norm(dim=1).mean(),
             'cache_age':c['ages'].detach().float().mean(),'delta':(next_h-h).detach().float().abs().mean(),'scale_record':scale_record,'pair_record':pair_record}
        return new,row

    def forward(self,x_f,m_f,**kwargs):
        ctx=self._initialize(x_f,m_f);initial=ctx['initial_prediction'];rows=[];capture=None
        request=self.probe_request if self.training else None
        forced_scales=kwargs.get('forced_scales')
        for step in range(4):
            if request is not None and step==request[0]:capture=detach_context(ctx,request[1])
            force_pair=kwargs.get('forced_pair_indices') if kwargs.get('forced_pair_step')==step else None
            ctx,row=self._round(ctx,step,forced_scales[:,step] if forced_scales is not None else None,force_pair)
            rows.append(row)
        stack=lambda key:torch.stack([r[key] for r in rows],1)
        predictions=[r['prediction'] for r in rows];weights=stack('weights');ids=stack('ids');a,b=ids.unbind(2)
        diagnostics={'hard_fraction':ctx['h'].new_tensor(1.),'missing_fraction':(1-ctx['mask']).mean()}
        for i,r in enumerate(rows,1):
            diagnostics[f'step{i}_state_change_abs']=r['delta']
            for key in ('input_delta','route_delta','memory_norm','cache_age'):diagnostics[f'step{i}_{key}']=r[key]
            diagnostics[f'step{i}_memory_gate_abs_mean']=r['gate'].detach().float().abs().mean()
            diagnostics[f'step{i}_memory_gate_signed_mean']=r['gate'].detach().float().mean()
            diagnostics[f'step{i}_pair_vs_top2_disagreement_rate']=(r['ids']!=r['logits'].topk(2,1).indices.sort(1).values).any(1).float().mean()
        coe={'predictions':predictions,'initial_prediction':initial,'initial_completion':ctx['completion'],
             'candidate_predictions':predictions,'candidate_changes':[r['change'] for r in rows],
             'changes':[r['change'] for r in rows], 'acceptance_weights':torch.ones_like(torch.stack(predictions,1)),
             'completions':[torch.where(ctx['mask'].bool(),ctx['x'],p) for p in predictions],
             'candidate_completions':[torch.where(ctx['mask'].bool(),ctx['x'],p) for p in predictions],
             'selected_experts':ids,'pair_ids':a*(2*self.num_experts-a-1)//2+b-a-1,
             'route_logits':stack('logits'),'route_probs':stack('probs'),'route_weights':weights,'route_importance':stack('importance'),
             'paths':weights.argmax(2),'paths_are_discrete':True,'routing_mode':'hard','configured_routing_mode':'hard',
             'expert_names':self.expert_names,'num_experts':self.num_experts,'num_steps':4,'top_k':2,
             'use_shared':False,'use_routed':True,'support':ctx['support'],'support_feature_names':SUPPORT_FEATURE_NAMES,
             'observation_mask':ctx['mask'],'diagnostics':diagnostics,
             'triscale_choices':stack('scale_weights').detach().argmax(2),'triscale_probabilities':stack('scale_probs'),
             'triscale_executed':stack('active'),'triscale_weights':stack('scale_weights')}
        output={'x_hat_main':predictions[-1],'h_st_aux':ctx['h'],'coe':coe,'diagnostics':{'coe':diagnostics}}
        if capture is not None:output['four_probe']={'context':capture,'step':request[0],'ids':request[1], 'record':rows[request[0]]}
        return output

    def candidate_loss(self,batch,outputs):
        info=outputs.get('four_probe')
        if info is None:return outputs['x_hat_main'].sum()*0,{}
        from ..losses import supervision_mask
        started=time.perf_counter();step=info['step'];ids=info['ids'];ctx=info['context'];row=info['record']
        target=batch['x_f_gt'].index_select(0,ids);mask=supervision_mask(batch['x_f_gt'],batch['m_f'],batch.get('target_mask')).index_select(0,ids)
        if self.teacher=='scale':
            candidates=torch.arange(3,device=ids.device)[None].expand(len(ids),3)
            features=row['scale_record']['base'].index_select(0,ids).detach()
            logits=self.scale_heads[step](features).float()
            prior=logits.new_zeros(3);prior[self.spec.get('fixed_path',[2,1,0,0])[step]]=1e-3
            logits=logits+prior
        else:
            rec=row['pair_record'];native=rec['native'].index_select(0,ids);chosen=rec['chosen'].index_select(0,ids)
            base_logits=rec['base_logits'].index_select(0,ids).detach().float()
            top=base_logits.topk(4,1).indices;lists=[]
            g=torch.Generator();g.set_state(self.probe_rng.cpu())
            for j in range(len(ids)):
                options=[]
                def add(pair):
                    pair=tuple(sorted(int(v) for v in pair))
                    if pair not in options:options.append(pair)
                add(native[j]);add(chosen[j])
                if self.pair=='partner':
                    primary=int(native[j,0]);legal=[(primary,k) for k in range(self.num_experts) if k!=primary]
                    for e in top[j].tolist():
                        if e!=primary:add((primary,e))
                else:
                    legal=self.pair_indices.tolist()
                    for p in self.pair_indices.tolist():
                        if p[0] in top[j] and p[1] in top[j]:add(p)
                options=options[:5]
                for k in torch.randperm(len(legal),generator=g).tolist():
                    if len(options)>=5:break
                    add(legal[k])
                lists.append(options)
            self.probe_rng.copy_(g.get_state().to(self.probe_rng.device))
            candidates=torch.tensor(lists,device=ids.device,dtype=torch.long)
            pf=rec['features'].index_select(0,ids).detach();corr=self.pair_heads[step](pf).float()
            if self.pair=='partner':
                primary=native[:,0];partners=torch.where(candidates[:,:,0]==primary[:,None],candidates[:,:,1],candidates[:,:,0])
                logits=(base_logits+corr).gather(1,partners)
            else:
                left,right=candidates.unbind(2);pids=left*(2*self.num_experts-left-1)//2+right-left-1
                scores=base_logits[:,self.pair_indices[:,0]]+base_logits[:,self.pair_indices[:,1]]+corr
                logits=scores.gather(1,pids)
        errors=[]
        with torch.no_grad():
            for k in range(candidates.shape[1]):
                c=dict(ctx)
                end=step+1 if self.teacher=='current' else 4
                for r in range(step,end):
                    scale=candidates[:,k] if self.teacher=='scale' and r==step else None
                    pair=candidates[:,k] if self.teacher!='scale' and r==step else None
                    c,trial=self._round(c,r,scale,pair,probe=True)
                diff=torch.where(mask,(trial['prediction']-target).abs(),torch.zeros_like(target)).float()
                errors.append(diff.flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1))
            errors=torch.stack(errors,1)
            temp=(.1*errors.mean(1,keepdim=True)).clamp_min(1e-6)
            q=(-errors/temp).softmax(1)
        loss=-(q*logits.log_softmax(1)).sum(1).mean()
        params=list(self.scale_heads[step].parameters() if self.teacher=='scale' else self.pair_heads[step].parameters())
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        norm=sum(g.detach().float().square().sum() for g in grads if g is not None).sqrt()
        return loss,{'four_probe_candidates':float(candidates.numel() if self.teacher=='scale' else candidates.shape[0]*candidates.shape[1]),
            'four_probe_seconds':time.perf_counter()-started,'four_probe_error_range':float((errors.max(1).values-errors.min(1).values).mean()),
            'four_probe_head_grad_norm':float(norm),'four_probe_round':float(step+1)}
