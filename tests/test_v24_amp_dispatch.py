"""Mixed-precision sparse dispatch must preserve window order and gradients."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE
from test_v24_coe import compact_config, make_batch

ROOT = Path(__file__).resolve().parents[1]


class AmpDispatchTests(unittest.TestCase):
    def test_low_precision_expert_outputs_scatter_into_float_state_with_gradients(self):
        # CPU autocast does not reproduce every CUDA normalization/convolution
        # dtype boundary. Force the reported boundary explicitly for regression.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(11)
                model = TemporalSpatialCoE.from_config(compact_config()).train()
                handles = [expert.register_forward_hook(
                    lambda module, args, output: output.to(dtype)) for expert in model.routed_experts()]
                try:
                    state = torch.randn(4, 8, 5, 3, 5, requires_grad=True)
                    paths = torch.tensor([1, 0, 1, 0])
                    result = model._dispatch(state, paths)
                    expected = torch.stack([model.routed_experts()[int(path)](
                        state[index:index+1])[0].float() for index, path in enumerate(paths)])
                    self.assertEqual(result.dtype, state.dtype)
                    torch.testing.assert_close(result, expected)
                    result.square().mean().backward()
                    self.assertTrue(torch.isfinite(state.grad).all())
                    self.assertGreater(float(state.grad.abs().sum()), 0)
                    for expert in model.routed_experts():
                        gradients = [p.grad for p in expert.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
                finally:
                    for handle in handles:
                        handle.remove()

    def exercise_autocast(self, device, dtype):
        variants = ('full', 'fixed_tt', 'fixed_ts', 'fixed_st', 'fixed_ss', 'parallel',
                    'soft', 'shared_only', 'routed_only', 'initial_router', 'no_expert_state_update',
                    'chain4_balance')
        batch = {key: value.to(device) for key, value in make_batch().items()}
        observed = batch['m_f'].bool().expand_as(batch['x_f_gt'])
        for variant in variants:
            with self.subTest(device=device, variant=variant):
                cfg = deep_update(compact_config(), json.loads(
                    (ROOT / f'configs/v24/experiments/{variant}.json').read_text()))
                model = DualBranchSTImputer.from_config(cfg).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                with torch.autocast(device_type=device, dtype=dtype):
                    output = model(batch)
                    loss, _ = compute_main_stage_loss(output, batch, cfg)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                optimizer.step()
                model.eval()
                # Evaluation hard routing also reaches sparse dispatch.
                with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
                    output = model(batch)
                self.assertTrue(torch.isfinite(output['x_hat_final']).all())
                torch.testing.assert_close(output['x_comp'][observed],
                                           batch['x_f_gt'][observed], rtol=0, atol=0)

    def test_cpu_autocast_all_core_controls_train_and_evaluate(self):
        self.exercise_autocast('cpu', torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_fp16_all_core_controls_train_and_evaluate(self):
        self.exercise_autocast('cuda', torch.float16)


if __name__ == '__main__':
    unittest.main()
