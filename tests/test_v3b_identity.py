"""T1 identity, T5 no GT leakage."""

import inspect
import unittest

import torch

from model.V3BResidualFusion import V3B0ResidualFusion
from v3b_runtime import assert_no_gt_in_forward, b0_forward, match_features, match_features_mode


class TestV3BIdentity(unittest.TestCase):
    def test_step0_delta_zero(self):
        torch.manual_seed(0)
        head = V3B0ResidualFusion()
        f0 = torch.randn(1, 32, 20, 30)
        t = torch.randn(1, 32, 20, 30)
        y0 = torch.randn(1, 3, 40, 60)
        y, d = b0_forward(head, f0, t, y0)
        self.assertLessEqual(float(d.abs().max()), 1e-7)
        self.assertLessEqual(float((y - y0).abs().max()), 1e-7)

    def test_forward_signature_no_gt(self):
        assert_no_gt_in_forward(V3B0ResidualFusion.forward)
        assert_no_gt_in_forward(match_features)
        names = inspect.signature(b0_forward).parameters
        for bad in ('H', 'q_star', 'U', 'D', 'g_v2'):
            self.assertNotIn(bad, names)


if __name__ == '__main__':
    unittest.main()
