"""B4 Base/matcher stay frozen across an adapter step."""

import unittest

import torch

from model.RetinexRefMainNet import RetinexRefMainNet
from model.V3BFeatureBridge import tiled_bridge_decode
from model.V3BFeatureResidual import FeatureResidualAdapter, delta_fn_from_adapter
from v3a5_runtime import bit_equal, snapshot_


class TestV3B4Freeze(unittest.TestCase):
    def test_base_bit_equal_after_adapter_step(self):
        torch.manual_seed(0)
        net = RetinexRefMainNet(n_feat=40, num_blocks=(1, 2, 2), level=2).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        snap = snapshot_(net)
        ad = FeatureResidualAdapter()
        with torch.no_grad():
            ad.net[-1].weight.fill_(0.001)
        x = torch.randn(1, 3, 32, 32)
        f0 = torch.randn(1, 32, 16, 16)
        t = torch.randn(1, 32, 16, 16)
        y = tiled_bridge_decode(net, x, delta_fn=delta_fn_from_adapter(ad, f0, t))
        loss = y.pow(2).mean()
        loss.backward()
        opt = torch.optim.Adam(ad.parameters(), lr=1e-4)
        opt.step()
        self.assertTrue(bit_equal(snap, snapshot_(net)))
        self.assertIsNotNone(ad.net[-1].weight.grad)
