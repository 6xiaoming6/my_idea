"""v25: independently masked observations of precomputed physical aggregates."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .diverse_masks import ALL_FAMILIES, DiverseMaskSchedule, make_diverse_mask
from .transforms import prepare_single_scale


def aggregate_sum(x: np.ndarray, factor: int) -> np.ndarray:
    """[N,C,T,H,W] -> physical sums, independently for each variable/time."""
    if x.ndim != 5 or x.shape[-2] % factor or x.shape[-1] % factor:
        raise ValueError('Expected BCTHW divisible by aggregation factor')
    n, c, t, h, w = x.shape
    return x.reshape(n,c,t,h//factor,factor,w//factor,factor).sum((4,6), dtype=np.float32)


class ObservedMultiScaleDataset(Dataset):
    """Only visible values/masks reach the model; full F is a training label.

    M/C ground truth must already be present on disk. No fallback from masked F.
    Fine-only controls mask M/C out but retain the identical model/interface.
    """
    def __init__(self, path, schedule, input_scales='FMC'):
        if input_scales not in ('F', 'FMC'):
            raise ValueError('input_scales must be F or FMC')
        self.input_scales = input_scales
        with np.load(path, allow_pickle=False) as z:
            self.values = [np.array(z[k], dtype=np.float32) for k in ('x_f_gt','x_m_gt','x_c_gt')]
        f = self.values[0]
        if f.ndim != 5 or f.shape[-2] % 4 or f.shape[-1] % 4:
            raise ValueError('Expected preconstructed aligned BCTHW scales')
        for x, factor in zip(self.values, (1,2,4)):
            if x.shape != (*f.shape[:3],f.shape[-2]//factor,f.shape[-1]//factor) or not np.isfinite(x).all():
                raise ValueError('Invalid shape or nonfinite physical aggregate')
        self.diverse_masks = DiverseMaskSchedule(len(f), schedule)

    def __len__(self):
        return len(self.values[0])

    def set_epoch(self, epoch):
        self.diverse_masks.set_epoch(epoch)

    def __getitem__(self, index):
        schedule = self.diverse_masks
        mask, family_id = schedule.sample(index, self.values[0].shape[2:])
        target = torch.from_numpy(self.values[0][index])
        sample = prepare_single_scale({'x_f_gt':target, 'm_f':torch.from_numpy(mask)[None]})
        sample['mask_family'] = torch.tensor(family_id, dtype=torch.long)
        assignment = int(schedule.assignments[index])
        rate = schedule.rates[assignment // len(schedule.families)]
        epoch = schedule.epoch if schedule.resample_each_epoch else 1
        for sid, name in enumerate(('m','c'), 1):
            x = torch.from_numpy(self.values[sid][index])
            # Same family/rate as F, independent scale RNG; never inspect target values.
            rng = np.random.default_rng(np.random.SeedSequence([schedule.seed,epoch,index,100003*sid]))
            mask = make_diverse_mask(x.shape[1:], rate, ALL_FAMILIES[family_id], rng)
            m = torch.from_numpy(mask)[None].expand_as(x).clone()
            if self.input_scales == 'F':
                m.zero_()
            sample['m_'+name] = m
            sample['x_'+name+'_obs'] = torch.where(m.bool(), x, 0.)
        return sample


def build_observed_dataset(cfg, path, split):
    from copy import deepcopy
    if path is None or split not in ('train','val','test'):
        raise ValueError('v25 requires explicit preconstructed data and split')
    data = cfg['data']
    if data.get('scales', {}).get('pooling_mode') != 'sum':
        raise ValueError('v25 first validation uses physical sums')
    schedule = deepcopy(data['train_mask_diversity' if split=='train' else 'eval_mask_diversity'])
    schedule['seed'] += {'train':0, 'val':20000, 'test':30000}[split]
    if split != 'train':
        schedule['resample_each_epoch'] = False
    return ObservedMultiScaleDataset(path, schedule, data['input_scales'])
