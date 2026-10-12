"""Opt-in Z controls: preserve the Y12 backbone, initialization and sparse dispatch."""
import copy
import torch
from .history_followup_coe import HistoryFollowupCoE
from .history_plane_interaction_coe import HistoryReader


class MechanismHistoryReader(HistoryReader):
    def read(self, history, mask, scale):
        result, weights = super().read(history, mask, scale)
        if self.representation == 'innovation':
            # Center the full historical read on the most recent prediction.
            # One-entry pools contain zero innovation; the original mask input remains.
            result = result - history[-1]
        return result, weights


class HistoryMechanismCoE(HistoryFollowupCoE):
    @classmethod
    def from_config(cls, cfg):
        spec = cfg['model']['coe']['history_mechanism']
        if set(spec) != {'enabled','interaction','history','representation'} or spec['enabled'] is not True:
            raise ValueError('Invalid Z configuration')
        if spec['interaction'] not in ('concat','product_difference','product_only','difference_only'):
            raise ValueError('Invalid Z interaction')
        if spec['history'] not in ('adaptive','mean','last') or spec['representation'] not in ('absolute','innovation'):
            raise ValueError('Invalid Z history')
        if cfg['model']['coe']['history_followup'] != dict(enabled=True,global_plane='none',history_pair=True):
            raise ValueError('Z requires the Y12 backbone without planes')
        if cfg['loss'].get('lambda_coe_mid',0) != 0:
            raise ValueError('Z excludes intermediate supervision')
        model = super().from_config(cfg)
        model.z_spec = copy.deepcopy(spec)
        old = model.x_history
        # Identical keys and values to Y12, including the scorer for fixed-read controls.
        with torch.random.fork_rng(devices=[]):
            reader = MechanismHistoryReader(model.c_in,model.dim,'adaptive')
        reader.load_state_dict(old.state_dict())
        reader.mode = spec['history']; reader.representation = spec['representation']
        if spec['history'] != 'adaptive':
            reader.score.requires_grad_(False)
        model.x_history = reader
        return model

    def _interaction_features(self,a,b):
        mode = self.z_spec['interaction']
        if mode == 'concat':
            return torch.cat((a,b),1)
        if mode == 'product_only':
            return torch.cat((a*b,torch.zeros_like(a)),1)
        if mode == 'difference_only':
            return torch.cat((torch.zeros_like(a),a-b),1)
        return torch.cat((a*b,a-b),1)
