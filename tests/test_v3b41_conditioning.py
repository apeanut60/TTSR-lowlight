"""B4.1 base-conditioning connectivity and zero/shuffled modes."""

import unittest

import torch

from model.RetinexRefMainNet import RetinexRefMainNet
from model.V3BBaseConditionedFeatureResidual import (
    BaseConditionedFeatureResidual, delta_fn_conditioned)
from model.V3BFeatureBridge import tiled_bridge_decode


class TestV3B41Conditioning(unittest.TestCase):
    def _unzero(self, ad):
        with torch.no_grad():
            ad.net[-1].weight.fill_(0.01)
            ad.net[-1].bias.zero_()
        return ad

    def test_changing_fdec_moves_delta(self):
        torch.manual_seed(0)
        ad = self._unzero(BaseConditionedFeatureResidual())
        f0 = torch.randn(1, 32, 8, 8)
        t = torch.randn(1, 32, 8, 8)
        a = torch.randn(1, 80, 8, 8)
        b = torch.zeros_like(a)
        self.assertGreater(float((ad(a, f0, t) - ad(b, f0, t)).abs().max()), 1e-8)

    def test_zero_and_shuffled_cond_modes(self):
        torch.manual_seed(1)
        ad = self._unzero(BaseConditionedFeatureResidual())
        net = RetinexRefMainNet(n_feat=40, num_blocks=(1, 2, 2), level=2).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        x = torch.randn(1, 3, 32, 32)
        f0 = torch.randn(1, 32, 16, 16)
        t = torch.randn(1, 32, 16, 16)
        f_donor = torch.randn(1, 80, 16, 16)
        y_n = tiled_bridge_decode(
            net, x, delta_fn=delta_fn_conditioned(ad, f0, t, 'normal'))
        y_z = tiled_bridge_decode(
            net, x, delta_fn=delta_fn_conditioned(ad, f0, t, 'zero'))
        y_s = tiled_bridge_decode(
            net, x, delta_fn=delta_fn_conditioned(
                ad, f0, t, 'shuffled', F_cond_full=f_donor))
        self.assertEqual(tuple(y_n.shape), tuple(x.shape))
        self.assertGreater(float((y_n - y_z).abs().max()), 1e-8)
        self.assertGreater(float((y_n - y_s).abs().max()), 1e-8)

    def test_step0_y_equals_bridge_base(self):
        ad = BaseConditionedFeatureResidual()  # zero last layer
        net = RetinexRefMainNet(n_feat=40, num_blocks=(1, 2, 2), level=2).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        x = torch.randn(1, 3, 32, 32)
        f0 = torch.randn(1, 32, 16, 16)
        t = torch.randn(1, 32, 16, 16)
        y0 = tiled_bridge_decode(net, x, delta_fn=None)
        y1 = tiled_bridge_decode(
            net, x, delta_fn=delta_fn_conditioned(ad, f0, t, 'normal'))
        self.assertLessEqual(float((y0 - y1).abs().max()), 1e-6)
