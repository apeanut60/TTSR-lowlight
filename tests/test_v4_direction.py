"""V4 step0 canvas identity and no GT leakage."""

import inspect
import unittest

import torch

from model.V3BResidualFusion import V3B0ResidualFusion
from v3b_runtime import assert_no_gt_in_forward


class TestV4Direction(unittest.TestCase):
    def test_a0_step0_is_y0(self):
        h = V3B0ResidualFusion(96)
        y0 = torch.randn(1, 3, 16, 16)
        f0 = torch.randn(1, 32, 8, 8)
        t = torch.randn(1, 32, 8, 8)
        y = y0 + h(f0, t, y0.shape[-2:])
        self.assertLessEqual(float((y - y0).abs().max()), 1e-7)

    def test_a1_step0_is_r(self):
        h = V3B0ResidualFusion(96)
        r = torch.randn(1, 3, 16, 16)
        fr = torch.randn(1, 32, 8, 8)
        t = torch.randn(1, 32, 8, 8)
        y = r + h(fr, t, r.shape[-2:])
        self.assertLessEqual(float((y - r).abs().max()), 1e-7)

    def test_no_gt_leakage(self):
        assert_no_gt_in_forward(V3B0ResidualFusion.forward)
        from model.V4RefCanvas import features_a1, features_a1_mode
        for fn in (features_a1, features_a1_mode):
            names = set(inspect.signature(fn).parameters)
            for bad in ('H', 'q_star', 'U', 'AO', 'D', 'g_v2'):
                self.assertNotIn(bad, names)
