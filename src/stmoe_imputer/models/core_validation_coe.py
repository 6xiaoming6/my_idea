"""Opt-in fixed-path controls for the first two contributions."""
import copy
import torch
from .backbone_exploration_coe import BackboneExplorationCoE
from .temporal_spatial_coe import TemporalSpatialCoE


class CoreValidationCoE(BackboneExplorationCoE):
    @classmethod
    def from_config(cls, cfg):
        shared = copy.deepcopy(cfg)
        shared['model']['coe']['expert_sharing'] = 'shared'
        m = super().from_config(shared)
        m.core_spec = copy.deepcopy(cfg['model']['coe']['core_validation'])
        if m.w != {'enabled': True}:
            raise ValueError('Core validation excludes all W training/adapter extensions')
        path = m.core_spec.get('path', 'CMFF')
        mode = m.core_spec.get('communication', 'conditional')
        if path not in ('CMFF', 'FFFF') or mode not in ('none', 'conditional', 'unconditional', 'raw'):
            raise ValueError('Invalid fixed-path control')
        m.spec['fixed_path'] = [{'F':0, 'M':1, 'C':2}[x] for x in path]
        if mode == 'none':
            m.communication = 'none'
            m.history_gate.requires_grad_(False)
        elif mode == 'raw':
            m.communication = 'raw_delta'
        sharing = cfg['model']['coe']['expert_sharing']
        if sharing == 'per_step':
            m.step_pattern_experts.extend(copy.deepcopy(m.pattern_experts) for _ in range(3))
        elif sharing != 'shared':
            raise ValueError('Unknown expert sharing')
        m.expert_sharing = sharing
        return m

    def _memory(self, c):
        if self.core_spec.get('communication') != 'unconditional':
            return super()._memory(c)
        h = c['h']; gate = h.new_zeros((len(h), self.dim))
        route = h.new_zeros((len(h), self.num_experts))
        c['w_stats'] = {k:h.new_zeros(()) for k in
                       ('feedback_abs','feedback_route_abs','visible_residual_abs','feedback_coverage')}
        if not c['has_prev']:
            return h, route, gate
        # Same head, zero input: isolate state conditioning, retain initialization.
        features = h.new_zeros((len(h), 4*self.dim))
        gate = self.bound * self.history_gate(features).tanh()
        message = self.rms_match(h-c['prev'], h)
        return h + gate[:, :, None, None, None]*message, route, gate

    def _execute(self, c, corrected, weights, scale_weights, step):
        if self.core_spec['path'] == 'CMFF':
            return super()._execute(c, corrected, weights, scale_weights, step)
        if not bool((scale_weights[:, 0] == 1).all()):
            raise ValueError('FFFF control left the fine scale')
        z = self.state_norm(self.state_projection(torch.cat(
            (corrected, c['completion'], c['mask'], c['support'], c['pos']), 1)))
        c['w_stats']['adapter_abs'] = z.new_zeros(())
        return TemporalSpatialCoE._dispatch_weighted(self, z, weights, step), None, []
