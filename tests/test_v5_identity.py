"""V5 step0: ΔD=0 ⇒ Y equals frozen Base; no GT in forward."""

import inspect
import unittest

import torch

from model.RetinexRefMainNet import RetinexRefMainNet
from model.V5Model import V5Model, v5_step0_deltas_zero
from model.V5RetinexBridge import bridge_zero_delta, prefix_to_h4, tiled_v5_forward
from v3b_runtime import assert_no_gt_in_forward


class TestV5Identity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = RetinexRefMainNet(n_feat=40, num_blocks=(1, 2, 2), level=2).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.branch = V5Model().eval()

    def test_zero_refine_and_bridge(self):
        self.assertTrue(v5_step0_deltas_zero(self.branch))
        x = torch.randn(1, 3, 40, 48)
        y0 = torch.randn(1, 3, 40, 48)
        r = torch.randn(1, 3, 40, 48)
        with torch.no_grad():
            y_base = self.net(x, use_reference=False, apply_illum=False,
                              apply_ref_illum=False)
            y_br = bridge_zero_delta(self.net, x)
            y_v5 = tiled_v5_forward(self.net, self.branch, x, y0, r)
        self.assertLessEqual(float((y_base - y_br).abs().max()), 1e-6)
        self.assertLessEqual(float((y_base - y_v5).abs().max()), 1e-6)

    def test_d2_shape(self):
        x = torch.randn(2, 3, 32, 36)
        d2, ctx = prefix_to_h4(self.net, x)
        self.assertEqual(tuple(d2.shape), (2, 160, 8, 9))

    def test_no_gt_leakage(self):
        assert_no_gt_in_forward(tiled_v5_forward)
        names = set(inspect.signature(V5Model.refine_at_h4).parameters)
        self.assertFalse(names & {'H', 'h', 'q_star', 'D', 'g_v2'})
