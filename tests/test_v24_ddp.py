"""Exact sharding and statistic merging used by the two-process v24 trainer."""
from __future__ import annotations

import importlib.util
import pickle
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stmoe_imputer.metrics import MaskedMetricAccumulator
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator
from stmoe_imputer.engine import _CoEQualityMetrics

spec = importlib.util.spec_from_file_location("v24_train", ROOT / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class DDPStatisticsTests(unittest.TestCase):
    def test_exact_samplers_cover_odd_dataset_without_duplicates(self):
        dataset = list(range(511))
        left = train.ExactDistributedTrainSampler(dataset, 0, 2, 7)
        right = train.ExactDistributedTrainSampler(dataset, 1, 2, 7)
        self.assertEqual(len(left), 256)
        self.assertEqual(len(right), 255)
        self.assertEqual(set(left) | set(right), set(dataset))
        self.assertFalse(set(left) & set(right))
        self.assertEqual((len(left) + 7) // 8, (len(right) + 7) // 8)
        before = list(left)
        left.set_epoch(2)
        self.assertNotEqual(before, list(left))
        eval_parts = [list(train.ExactDistributedEvalSampler(dataset, rank, 2)) for rank in range(2)]
        self.assertEqual(sorted(eval_parts[0] + eval_parts[1]), dataset)

    def test_masked_and_routing_totals_merge_exactly(self):
        torch.manual_seed(12)
        target = torch.randn(5, 2, 3, 4, 4)
        pred = target + torch.randn_like(target) * .2
        mask = (torch.rand_like(target) > .4).float()
        full = MaskedMetricAccumulator()
        full.update(pred, target, mask)
        shards = [MaskedMetricAccumulator(), MaskedMetricAccumulator()]
        for rank, shard in enumerate(shards):
            shard.update(pred[rank::2], target[rank::2], mask[rank::2])
        shards[0].merge(shards[1])
        for key, value in full.compute().items():
            self.assertAlmostEqual(value, shards[0].compute()[key], delta=1e-6)

        weights = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]],
                                [[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]],
                                [[1., 0.], [0., 1.]]])
        record = {"route_weights": weights, "route_probs": weights * .8 + .1,
                  "paths": weights.argmax(-1), "selected_experts": weights.argmax(-1).unsqueeze(-1),
                  "routing_mode": "hard", "expert_names": ("T", "S"), "top_k": 1}
        whole = CoERoutingMetricAccumulator(); whole.update(record)
        pieces = []
        for rank in range(2):
            part = CoERoutingMetricAccumulator()
            part.update({key: value[rank::2] if torch.is_tensor(value) else value
                         for key, value in record.items()})
            pieces.append(pickle.loads(pickle.dumps(part)))
        pieces[0].merge(pieces[1])
        self.assertEqual(set(whole.compute()), set(pieces[0].compute()))
        for key, value in whole.compute().items():
            self.assertAlmostEqual(value, pieces[0].compute()[key], delta=1e-12, msg=key)
        self.assertEqual(whole.compute()["coe_routing_sample_count"], 5.0)

    def test_quality_accumulator_can_cross_process_boundary(self):
        quality = _CoEQualityMetrics()
        restored = pickle.loads(pickle.dumps(quality))
        quality.merge(restored)
        self.assertEqual(quality.compute(), {})


if __name__ == "__main__":
    unittest.main()
