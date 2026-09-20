"""Target-independent, reproducible masks with balanced geometric families.

All fields are geometry/randomness only. Selecting the smallest field values
keeps the rounded missing-cell budget exact across shapes; ties are randomized.
"""
from __future__ import annotations

import math
import numpy as np

FAMILIES = ('random_point', 'node_outage', 'temporal_gap', 'spatial_region',
            'spatiotemporal_block', 'stripe', 'moving_region', 'multi_block', 'composite')


def _field(shape, family, rng):
    t, h, w = np.meshgrid(*(np.linspace(0, 1, n) for n in shape), indexing='ij')
    center = rng.uniform(0, 1, 3)
    scale = rng.uniform(.12, .6, 3)
    dt = abs(t - center[0]) / scale[0]
    dh, dw = (h - center[1]) / scale[1], (w - center[2]) / scale[2]
    if family == 'random_point':
        return rng.random(shape)
    if family == 'node_outage':
        return np.broadcast_to(rng.random(shape[1:])[None], shape)
    if family == 'temporal_gap':
        return dt
    if family == 'spatial_region':
        angle = rng.uniform(0, np.pi)
        y, x = h - center[1], w - center[2]
        return ((y*np.cos(angle)+x*np.sin(angle))/scale[1])**2 + ((x*np.cos(angle)-y*np.sin(angle))/scale[2])**2
    if family == 'spatiotemporal_block':
        return np.maximum(np.maximum(dt, abs(dh)), abs(dw))
    if family == 'stripe':
        return abs(dh) if rng.integers(2) else abs(dw)
    if family == 'moving_region':
        # Centers drift linearly, without wrapping at the spatial boundary.
        velocity = rng.uniform(-.8, .8, 2)
        return ((h-center[1]-velocity[0]*(t-.5))/scale[1])**2 + ((w-center[2]-velocity[1]*(t-.5))/scale[2])**2
    if family == 'multi_block':
        return np.minimum.reduce([_field(shape, 'spatiotemporal_block', rng) for _ in range(int(rng.integers(2, 5)))])
    if family == 'composite':
        # A union of 2–3 distinct components. Quantile ranks put geometries on
        # comparable scales; variable quotas prevent one component dominating.
        structured = str(rng.choice(('temporal_gap', 'spatial_region', 'spatiotemporal_block',
                                     'stripe', 'moving_region', 'multi_block')))
        choices = [structured, *rng.choice([f for f in FAMILIES[:-1] if f != structured],
                                           size=int(rng.integers(1, 3)), replace=False)]
        fields = []
        for part in choices:
            raw = _field(shape, part, rng).ravel()
            order = np.lexsort((rng.random(raw.size), raw))
            ranks = np.empty(raw.size); ranks[order] = np.arange(raw.size)/raw.size
            fields.append(ranks.reshape(shape)/rng.uniform(.5, 1.5))
        return np.minimum.reduce(fields)
    raise ValueError(f'Unknown mask family: {family}')


def make_diverse_mask(shape, rate, family, rng):
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('Expected positive integer T/H/W dimensions')
    if family not in FAMILIES or not math.isfinite(rate) or not 0 < rate < 1:
        raise ValueError('Expected a known family and a missing rate strictly between zero and one')
    count = round(math.prod(shape)*rate)
    if not 0 < count < math.prod(shape):
        raise ValueError('Shape/rate must leave both observed and missing cells')
    field = _field(shape, family, rng).ravel()
    order = np.lexsort((rng.random(field.size), field))
    mask = np.ones(field.size, dtype=np.float32); mask[order[:count]] = 0
    return mask.reshape(shape)


class DiverseMaskSchedule:
    """One family/rate per sample; balanced coverage re-shuffled each epoch.

    Family labels are for logging only, never model inputs. The local RNG does
    not depend on worker count, batch order, model initialization or targets.
    """
    def __init__(self, length, config):
        self.length = length
        self.families = tuple(config.get('families', FAMILIES))
        self.rates = tuple(config.get('rates', [.4]))
        self.seed = config.get('seed', 20260917)
        self.resample_each_epoch = bool(config.get('resample_each_epoch', True))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Mask seed must be a nonnegative integer')
        if not self.families or len(set(self.families)) != len(self.families) or any(f not in FAMILIES for f in self.families):
            raise ValueError('Mask families must be unique supported names')
        if not self.rates or any(not math.isfinite(r) or not 0 < r < 1 for r in self.rates):
            raise ValueError('Mask rates must be finite and strictly between zero and one')
        self.set_epoch(1)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Mask epoch must be a positive integer')
        self.epoch = epoch
        schedule_epoch = epoch if self.resample_each_epoch else 1
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, schedule_epoch, 0]))
        combinations = len(self.families)*len(self.rates)
        offset = (schedule_epoch-1)*self.length % combinations
        self.assignments = (np.arange(self.length)+offset) % combinations
        rng.shuffle(self.assignments)

    def sample(self, index, shape):
        assignment = int(self.assignments[index])
        family = self.families[assignment % len(self.families)]
        rate = self.rates[assignment // len(self.families)]
        sample_epoch = self.epoch if self.resample_each_epoch else 1
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, sample_epoch, index, 1]))
        mask = make_diverse_mask(tuple(shape), rate, family, rng)
        return mask, FAMILIES.index(family)
