"""Guard the sparse Top-1 baseline and the B-series comparison protocol."""
import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/v24'))
from run_b3_c3 import jobs
from stmoe_imputer.models import DualBranchSTImputer


class StructureBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_common_protocol_and_pool_sharing(self):
        plan = jobs(variants=('B1', 'B2', 'B3', 'B4', 'B5'))
        reference = None
        for job in plan:
            cfg = copy.deepcopy(job['config'])
            coe = cfg['model']['coe']
            self.assertEqual(coe['num_steps'] * coe['top_k'], 8)
            self.assertEqual(len(coe['fixed_expert_steps']), coe['num_steps'])
            cfg['model']['main']['dim'] = 8
            coe['router_hidden_dim'] = 8
            model = DualBranchSTImputer.from_config(cfg).main_branch
            first = model.routed_experts(0)[0]
            for step in range(1, model.num_steps):
                self.assertEqual(first is model.routed_experts(step)[0], coe['expert_sharing'] == 'shared')
            for key in ('num_steps', 'top_k', 'expert_sharing', 'fixed_expert_steps', 'top1_selection'):
                coe.pop(key, None)
            cfg.pop('experiment_plan')
            if reference is None:
                reference = cfg
            else:
                self.assertEqual(cfg, reference)

    def test_top1_argmax_sparse_forward_and_task_gradient(self):
        for variant in ('B4', 'B5'):
            with self.subTest(variant=variant):
                torch.manual_seed(7)
                cfg = jobs(variants=(variant,))[0]['config']
                cfg['model']['main']['dim'] = 8
                cfg['model']['coe']['router_hidden_dim'] = 8
                model = DualBranchSTImputer.from_config(cfg).train()
                backbone = model.main_branch
                # Make every round choose expert 0, so unselected execution and
                # accidental dense gradients cannot hide behind mixed batches.
                for router in backbone.routers:
                    torch.nn.init.zeros_(router[-1].weight)
                    router[-1].bias.data.copy_(torch.arange(8, 0, -1).float())
                selected, unselected = [], []
                for step in range(8):
                    for eid, expert in enumerate(backbone.routed_experts(step)):
                        target = selected if eid == 0 else unselected
                        if all(expert is not old for old in target):
                            target.append(expert)
                counts = [0, 0]
                hooks = []
                def hook_for(kind):
                    def hook(module, args, output):
                        counts[kind] += args[0].shape[0]
                    return hook
                for expert in selected:
                    hooks.append(expert.register_forward_hook(hook_for(0)))
                for expert in unselected:
                    hooks.append(expert.register_forward_hook(hook_for(1)))
                x = torch.randn(2, 2, 3, 8, 8)
                mask = (torch.rand_like(x) > .4).float()
                batch = {'x_f_obs': x * mask, 'm_f': mask}
                out = model(batch)
                weights = out['coe']['route_weights']
                self.assertTrue(torch.equal(weights.argmax(-1), out['coe']['route_logits'].argmax(-1)))
                self.assertTrue(torch.equal(weights[..., 0], torch.ones(2, 8)))
                self.assertEqual(torch.count_nonzero(weights[..., 1:]).item(), 0)
                self.assertEqual(counts, [16, 0])
                # Main task alone must train each router; no balance loss here.
                ((out['x_hat_main'] - x).square() * (1-mask)).mean().backward()
                for router in backbone.routers:
                    grad = router[-1].bias.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                for expert in unselected:
                    self.assertTrue(all(p.grad is None for p in expert.parameters()))
                for hook in hooks:
                    hook.remove()
                with torch.no_grad():
                    evaluated = model.eval()(batch)
                torch.testing.assert_close(out['x_hat_main'], evaluated['x_hat_main'])
                torch.testing.assert_close(weights, evaluated['coe']['route_weights'])


if __name__ == '__main__':
    unittest.main()
