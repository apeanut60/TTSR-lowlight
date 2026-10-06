"""Gradients reach the Ref branch; frozen Base gets none."""

import unittest

import torch

from model.RetinexRefMainNet import RetinexRefMainNet
from model.V5Model import V5Model
from model.V5RetinexBridge import tiled_v5_forward


class TestV5Gradient(unittest.TestCase):
    def test_grad_reaches_refine_not_base(self):
        torch.manual_seed(0)
        net = RetinexRefMainNet(n_feat=40, num_blocks=(1, 2, 2), level=2).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        branch = V5Model()
        with torch.no_grad():
            branch.refine_h4.out.weight.fill_(0.01)
            branch.refine_h2.out.weight.fill_(0.01)
        x = torch.randn(1, 3, 32, 32)
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        y = tiled_v5_forward(net, branch, x, y0, r)
        y.pow(2).mean().backward()
        g_ref = 0.0
        for p in branch.parameters():
            if p.grad is not None:
                g_ref += float(p.grad.abs().sum())
        self.assertGreater(g_ref, 0.0)
        for p in net.parameters():
            self.assertIsNone(p.grad)
        self.assertIsNotNone(branch.texture_encoder.stem.weight.grad)
        self.assertIsNotNone(branch.match_encoder.net[0].weight.grad)
