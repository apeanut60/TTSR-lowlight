"""DCN zero residual offset ≈ coarse warp (integer flow, identity kernel, mask≈1)."""

import unittest

import torch

from model.V5DeformAlign import FlowGuidedDeformAlign
from model.V5FlowAlign import warp_by_flow


class TestV5DCN(unittest.TestCase):
    def test_zero_residual_follows_coarse_flow(self):
        torch.manual_seed(0)
        ch = 8
        x = torch.randn(1, ch, 16, 16)
        tgt = torch.randn(1, ch, 16, 16)
        flow = torch.zeros(1, 2, 16, 16)
        flow[:, 0] = 2
        flow[:, 1] = -1
        net = FlowGuidedDeformAlign(ch).eval()
        with torch.no_grad():
            deform, residual, mask = net(x, tgt, flow)
            warped = warp_by_flow(x, flow, mode='bilinear')
        self.assertLessEqual(float(residual.abs().max()), 1e-7)
        self.assertGreater(float(mask.mean()), 0.99)
        interior = (slice(None), slice(None), slice(2, -2), slice(2, -2))
        d = float((deform[interior] - warped[interior]).abs().mean())
        self.assertLess(d, 0.15, 'DCN vs warp mean abs %s' % d)
