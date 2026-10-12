"""V6 step0: ΔD=0 ⇒ Y equals frozen LLFormer Base."""

import unittest

import torch

from model.LLFormerBridge import LLFormerBridge
from model.V6LLFormerRefBridge import base_forward_m11, tiled_v6_forward
from model.V6Model import V6Model, v6_step0_deltas_zero


class TestV6Identity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = LLFormerBridge().eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.branch = V6Model().eval()

    def test_zero_refine(self):
        self.assertTrue(v6_step0_deltas_zero(self.branch))
        x = torch.randn(1, 3, 128, 128)
        y0 = torch.randn(1, 3, 128, 128)
        r = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            y_base = base_forward_m11(self.base, x)
            y_v6 = tiled_v6_forward(self.base, self.branch, x, y0, r)
        self.assertLessEqual(float((y_base - y_v6).abs().max()), 1e-5)


if __name__ == '__main__':
    unittest.main()
