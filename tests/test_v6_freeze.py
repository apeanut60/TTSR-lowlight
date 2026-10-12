"""Frozen Base must not receive grads; Ref branch must."""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from model.LLFormerBridge import LLFormerBridge
from model.V6LLFormerRefBridge import tiled_v6_forward
from model.V6Model import V6Model


class TestV6Freeze(unittest.TestCase):
    def test_grad_routing(self):
        base = LLFormerBridge().eval()
        for p in base.parameters():
            p.requires_grad_(False)
        branch = V6Model().train()
        x = torch.randn(1, 3, 128, 128)
        y0 = torch.randn(1, 3, 128, 128)
        r = torch.randn(1, 3, 128, 128)
        h = torch.randn(1, 3, 128, 128)
        y = tiled_v6_forward(base, branch, x, y0, r)
        loss = F.mse_loss(y, h)
        loss.backward()
        for p in base.parameters():
            self.assertIsNone(p.grad)
        gsum = sum(float(p.grad.abs().sum()) for p in branch.parameters()
                   if p.grad is not None)
        self.assertGreater(gsum, 0.0)


if __name__ == '__main__':
    unittest.main()
