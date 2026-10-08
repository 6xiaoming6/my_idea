"""Isolated U-series: bounded communication, staged scales and head-only teachers."""
from __future__ import annotations
import copy
import time
import torch
from torch import nn
from torch.nn import functional as F
from .temporal_spatial_coe import TemporalSpatialCoE
from .four_direction_coe import FourDirectionCoE, head, detach_context
from .spatial_scale_coe import spatial_pool, spatial_resize, observed_pool


class IDPriorityCoE(FourDirectionCoE):
    @classmethod
    def from_config(cls, cfg):
        # The exact baseline constructor runs before any extension draws randomness.
        m = TemporalSpatialCoE.from_config.__func__(cls, cfg)
        m.u = copy.deepcopy(cfg['model']['coe']['id_priority'])
        if (m.num_steps != 4 or m.top_k != 2 or m.pair_mode != 'native'
                or m.routing_mode != 'hard' or m.state_update_mode != 'direct'
                or m.completion_feedback or m.use_shared or not m.use_routed
                or m.expert_sharing != 'shared' or m.router_features != 'legacy'
                or m.fusion_mode != 'original'):
            raise ValueError('U series requires the shared four-round native Top2 direct backbone')
        m.communication = m.u.get('communication', 'none')
        m.scale_policy = m.u.get('scale_policy', 'fixed')
        m.constraint = m.u.get('constraint', 'none')
        m.warmup_epochs = int(m.u.get('fixed_epochs', 0))
        m.bound = float(m.u.get('bound', .1))
        if m.communication not in ('none', 'delta', 'raw_delta', 'innovation', 'innovation_detach', 'innovation_router'):
            raise ValueError('Invalid U communication')
        if m.scale_policy not in ('fixed', 'st', 'teacher', 'top2', 'fixed2'):
            raise ValueError('Invalid U scale policy')
        if m.constraint not in ('none', 'last_two_fine', 'monotone') or m.warmup_epochs < 0 or not 0 < m.bound <= 1:
            raise ValueError('Invalid constraint, fixed epochs or message bound')
        if m.scale_policy in ('top2', 'fixed2') and m.constraint != 'none':
            raise ValueError('Scale-pair groups do not use Top1 constraints')
        # Reuse the proven native-pair, forward and diagnostic contracts.
        m.spec = {'fixed_path': [2, 1, 0, 0]}
        m.memory = 'none'; m.scale = 'fixed'; m.pair = 'native'
        m.teacher = 'scale' if m.scale_policy == 'teacher' else 'none'
        m.probe_request = None
        m.feature_dim = m.routers[0][0].normalized_shape[0]
        seed = int(cfg.get('seed', 7))
        # Fork CPU RNG; constructors create CPU tensors. Global/data RNG resumes exactly.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 71001)
            m.scale_heads = nn.ModuleList([head(m.feature_dim + 4, 3) for _ in range(4)])
            for step, h in enumerate(m.scale_heads):
                if m.scale_policy in ('fixed', 'fixed2') or (m.constraint == 'last_two_fine' and step >= 2) or (m.constraint == 'monotone' and step == 3):
                    h.requires_grad_(False)
            torch.random.default_generator.manual_seed(seed + 71002)
            if m.communication not in ('none', 'innovation_router'):
                m.history_gate = head(4 * m.dim, m.dim)
            if m.communication == 'innovation_router':
                m.innovation_router = head(4 * m.dim, m.num_experts)
            torch.random.default_generator.manual_seed(seed + 71003)
            if m.u.get('response_fusion', False):
                m.response_head = head(4 * m.dim + 2, 1)
            if m.u.get('scale_identity', False):
                m.scale_identity = nn.Parameter(torch.zeros(3, m.dim))
        gen = torch.Generator().manual_seed(seed + 71004)
        m.register_buffer('route_rng', gen.get_state())
        m.register_buffer('probe_rng', gen.get_state().clone())
        m.register_buffer('batch_clock', torch.zeros((), dtype=torch.long))
        m.register_buffer('teacher_clock', torch.zeros((), dtype=torch.long))
        m.register_buffer('stage_epoch', torch.ones((), dtype=torch.long))
        return m

    def set_routing_epoch(self, epoch):
        super().set_routing_epoch(epoch)
        self.stage_epoch.fill_(int(epoch))

    def scales_open(self):
        return int(self.stage_epoch.item()) > self.warmup_epochs

    def learnable_steps(self):
        if self.constraint == 'last_two_fine':
            return (0, 1)
        if self.constraint == 'monotone':
            return (0, 1, 2)
        return (0, 1, 2, 3)

    def prepare_training_batch(self, batch):
        self.probe_request = None
        clock = int(self.batch_clock.item()); self.batch_clock.add_(1)
        if self.teacher == 'none' or not self.scales_open() or clock % 20:
            return
        from ..losses import supervision_mask
        valid = supervision_mask(batch['x_f_gt'], batch['m_f'], batch.get('target_mask')).flatten(1).any(1)
        ids = valid.nonzero().flatten()[:4]
        steps = self.learnable_steps()
        step = steps[int(self.teacher_clock.item()) % len(steps)]
        self.teacher_clock.add_(1)
        if ids.numel():
            self.probe_request = (step, ids)

    @staticmethod
    def rms_match(message, h):
        ratio = (h.detach().float().square().mean((2, 3, 4), keepdim=True).sqrt()
                 / message.detach().float().square().mean((2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6))
        return message * ratio.to(message.dtype)

    def _memory(self, c):
        h = c['h']; zero = h.new_zeros((len(h), self.dim))
        route = h.new_zeros((len(h), self.num_experts))
        if not c['has_prev'] or self.communication == 'none':
            return h, route, zero
        message = c['message'] if self.communication.startswith('innovation') else h - c['prev']
        if self.communication != 'raw_delta':
            message = self.rms_match(message, h)
        if self.communication == 'innovation_detach':
            message = message.detach()
        if self.communication == 'innovation_router':
            features = torch.cat((self.summary(h, c['mask']), self.summary(message, c['mask'])), 1)
            return h, self.bound * self.innovation_router(features).tanh(), zero
        # Same gate features and placement as M07, including the detach ablation.
        features = torch.cat((self.summary(h, c['mask']), self.summary(c['prev'], c['mask'])), 1)
        gate = self.bound * self.history_gate(features).tanh()
        return h + gate[:, :, None, None, None] * message, route, gate

    def legal_scales(self, c, step):
        legal = torch.ones((len(c['h']), 3), device=c['h'].device, dtype=torch.bool)
        if self.constraint == 'last_two_fine' and step >= 2:
            legal[:, 1:] = False
        elif self.constraint == 'monotone':
            if step:
                previous = c['last_scale'].argmax(1)
                legal &= torch.arange(3, device=legal.device)[None] <= previous[:, None]
            if step == 3:
                legal[:, 1:] = False
        return legal

    def _scales(self, c, features, step, force=None, probe=False):
        b = len(features); device = features.device
        base = torch.cat((features, c['counts'].to(features.dtype) / 4,
                          features.new_full((b, 1), (4-step)/4)), 1)
        legal = self.legal_scales(c, step)
        policy = self.scale_policy if self.scales_open() else 'fixed'
        fixed = self.spec['fixed_path'][step]
        if policy == 'fixed':
            choice = torch.full((b,), fixed, device=device, dtype=torch.long)
            if force is not None and not torch.equal(force, choice):
                raise ValueError('Cannot force another scale in a fixed phase')
            weights = F.one_hot(choice, 3).float(); logits = torch.zeros_like(weights)
            return weights, weights, weights.bool(), {'base': base, 'legal': legal, 'logits': logits}
        logits = self.scale_heads[step](base).float()
        prior = logits.new_zeros(3)
        if policy in ('top2', 'fixed2'):
            pair = ((2, 1), (1, 0), (2, 0), (1, 0))[step]
            prior[list(pair)] = 1e-3
            logits = logits + prior
            ids = logits.topk(2, 1).indices if policy == 'top2' else torch.tensor(pair, device=device).expand(b, 2)
            active = F.one_hot(ids, 3).any(1)
            if policy == 'fixed2':
                weights = active.float() / 2
            else:
                chosen = logits.gather(1, ids).softmax(1).clamp_min(1e-30)
                chosen = chosen / chosen.sum(1, keepdim=True)
                weights = torch.zeros_like(logits).scatter(1, ids, chosen)
            return weights, logits.softmax(1) if policy == 'top2' else weights, active, {'base': base, 'legal': legal, 'logits': logits}
        prior[fixed] = 1e-3
        logits = (logits + prior).masked_fill(~legal, -1e9)
        probs = logits.softmax(1)
        choice = logits.argmax(1) if force is None else force
        if not legal.gather(1, choice[:, None]).all():
            raise ValueError('Forced scale violates the path constraint')
        hard = F.one_hot(choice, 3).float()
        weights = hard + (probs - probs.detach()) if policy == 'st' and self.training and not probe else hard
        return weights, probs, hard.bool(), {'base': base, 'legal': legal, 'logits': logits}

    def _execute(self, c, corrected, weights, scale_weights, step):
        out = torch.zeros_like(c['h'])
        innovation = torch.zeros_like(c['h']) if self.communication.startswith('innovation') else None
        for sid, factor in enumerate((1, 2, 4)):
            selected = (scale_weights[:, sid].detach() != 0).nonzero().flatten()
            if not selected.numel():
                continue
            h, mask, completion, support, pos = [c[k].index_select(0, selected) for k in ('h', 'mask', 'completion', 'support', 'pos')]
            h = corrected.index_select(0, selected)
            if factor > 1:
                values, coverage = observed_pool(completion, mask, factor)
                completion = torch.where(coverage > 0, values, spatial_pool(completion, factor)); mask = coverage
                h = spatial_pool(h, factor); support = spatial_pool(support, factor); pos = spatial_pool(pos, factor)
            z = self.state_norm(self.state_projection(torch.cat((h, completion, mask, support, pos), 1)))
            if hasattr(self, 'scale_identity') and self.scales_open():
                z = z + self.scale_identity[sid][None, :, None, None, None].to(z.dtype)
            w = weights.index_select(0, selected)
            if self.u.get('response_fusion', False):
                ids = w.topk(2, 1).indices
                a = self._dispatch(z, ids[:, 0], step); b = self._dispatch(z, ids[:, 1], step)
                within = w.gather(1, ids)
                desc = torch.cat((self.summary(a, mask), self.summary(b, mask), within), 1)
                correction = .1 * self.response_head(desc.detach()).tanh()
                difference = within[:, 1:2].clamp_min(1e-8).log() - within[:, :1].clamp_min(1e-8).log()
                shift = torch.sigmoid(difference + correction) - torch.sigmoid(difference)
                within = within + torch.cat((-shift, shift), 1)
                update = a * within[:, 0, None, None, None, None].to(a.dtype) + b * within[:, 1, None, None, None, None].to(b.dtype)
            else:
                update = TemporalSpatialCoE._dispatch_weighted(self, z, w, step)
            if hasattr(self, '_record_native_scale'):
                self._record_native_scale(c, update, sid, step, selected)
            if innovation is not None:
                change = update - z
                if factor > 1:
                    change = spatial_resize(change, c['h'].shape[-2:])
                innovation = innovation.index_add(0, selected, (change * scale_weights[selected, sid, None, None, None, None].to(change.dtype)).to(innovation.dtype))
            restored = spatial_resize(update, c['h'].shape[-2:]) if factor > 1 else update
            out = out.index_add(0, selected, (restored * scale_weights[selected, sid, None, None, None, None].to(restored.dtype)).to(out.dtype))
        return out, innovation, []

    def _round(self, c, step, force_scale=None, force_pair=None, probe=False):
        new, row = super()._round(c, step, force_scale, force_pair, probe)
        if c['has_prev'] and self.communication.startswith('innovation'):
            row['memory_norm'] = c['message'].detach().float().square().mean().sqrt()
        return new, row

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        diagnostics = out['coe']['diagnostics']
        diagnostics['scale_phase_open'] = out['x_hat_main'].new_tensor(float(self.scales_open()))
        diagnostics['communication_bound'] = out['x_hat_main'].new_tensor(self.bound)
        for i in range(1, 5):
            diagnostics[f'step{i}_gate_fraction_of_bound'] = diagnostics[f'step{i}_memory_gate_abs_mean'] / self.bound
        return out

    def candidate_loss(self, batch, outputs):
        info = outputs.get('four_probe')
        if info is None:
            return outputs['x_hat_main'].sum() * 0, {}
        from ..losses import supervision_mask
        start = time.perf_counter(); step = info['step']; ids = info['ids']
        legal = info['record']['scale_record']['legal'].index_select(0, ids)
        keep = (legal.sum(1) > 1).nonzero().flatten()
        if not keep.numel():
            return outputs['x_hat_main'].sum() * 0, {'four_probe_candidates': 0., 'four_probe_windows': 0., 'four_probe_round': float(step+1)}
        ids = ids.index_select(0, keep); legal = legal.index_select(0, keep)
        c = detach_context(info['context'], keep)
        features = info['record']['scale_record']['base'].index_select(0, ids).detach()
        logits = self.scale_heads[step](features).float()
        prior = logits.new_zeros(3); prior[self.spec['fixed_path'][step]] = 1e-3
        logits = (logits + prior).masked_fill(~legal, -1e9)
        target = batch['x_f_gt'].index_select(0, ids)
        mask = supervision_mask(batch['x_f_gt'], batch['m_f'], batch.get('target_mask')).index_select(0, ids)
        errors = torch.zeros((len(ids), 3), device=ids.device, dtype=torch.float32)
        buffers = {n: b.detach().clone() for n, b in self.named_buffers()}
        cuda_devices = [ids.device.index if ids.device.index is not None else torch.cuda.current_device()] if ids.is_cuda else []
        # Rollouts consume no global RNG and never use the public forward/batch hooks.
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            for sid in range(3):
                selected = legal[:, sid].nonzero().flatten()
                if not selected.numel():
                    continue
                context = detach_context(c, selected)
                for r in range(step, 4):
                    force = torch.full((len(selected),), sid, device=ids.device, dtype=torch.long) if r == step else None
                    context, trial = self._round(context, r, force_scale=force, probe=True)
                valid = mask.index_select(0, selected)
                diff = torch.where(valid, (trial['prediction'] - target.index_select(0, selected)).abs(), 0.).float()
                errors[selected, sid] = diff.flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
            for n, b in self.named_buffers():
                if not torch.equal(b, buffers[n]):
                    raise RuntimeError('Teacher mutated model buffer: ' + n)
            temp = (.1 * errors.sum(1, keepdim=True) / legal.sum(1, keepdim=True)).clamp_min(1e-6)
            q = (-errors / temp).masked_fill(~legal, -1e9).softmax(1)
        loss = -(q * logits.log_softmax(1)).sum(1).mean()
        grads = torch.autograd.grad(loss, list(self.scale_heads[step].parameters()), retain_graph=True, allow_unused=True)
        norm = sum(g.detach().float().square().sum() for g in grads if g is not None).sqrt()
        spread = errors.masked_fill(~legal, -torch.inf).max(1).values - errors.masked_fill(~legal, torch.inf).min(1).values
        return loss, {'four_probe_candidates': float(legal.sum()), 'four_probe_windows': float(len(ids)),
                      'four_probe_seconds': time.perf_counter()-start, 'four_probe_head_grad_norm': float(norm),
                      'four_probe_error_range': float(spread.mean()), 'four_probe_round': float(step+1)}
