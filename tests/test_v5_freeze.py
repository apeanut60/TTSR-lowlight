"""Frozen Base stays bit-equal after a V5 step."""

import unittest

import torch

from model.RetinexRefMainNet import RetinexRefMainNet
from model.V5Model import V5Model
from model.V5RetinexBridge import tiled_v5_forward
from v3a5_runtime import bit_equal, snapshot_


class TestV5Freeze(unittest.TestCase):
    def test_base_bit_equal_after_step(self):
        torch.manual_seed(0)
        net = RetinexRefMainNet(n_feat=40, num_blocks=(1, 2, 2), level=2).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        snap = snapshot_(net)
        branch = V5Model()
        with torch.no_grad():
            branch.refine_h4.out.weight.fill_(0.01)
        opt = torch.optim.Adam(branch.parameters(), lr=1e-4)
        x = torch.randn(1, 3, 32, 32)
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        h = torch.randn(1, 3, 32, 32)
        opt.zero_grad(set_to_none=True)
        y = tiled_v5_forward(net, branch, x, y0, r)
        (y - h).pow(2).mean().backward()
        opt.step()
        self.assertTrue(bit_equal(snap, snapshot_(net)))
