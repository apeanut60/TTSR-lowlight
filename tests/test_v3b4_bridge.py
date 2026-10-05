"""B4 frozen Base prefix/tail equals original forward when ΔF=0."""

import unittest

import torch

from model.RetinexRefMainNet import RetinexRefMainNet
from model.V3BFeatureBridge import bridge_zero_delta, prefix_to_h2


class TestV3B4Bridge(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = RetinexRefMainNet(n_feat=40, num_blocks=(1, 2, 2), level=2)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def test_zero_delta_matches_base(self):
        x = torch.randn(1, 3, 40, 48)
        with torch.no_grad():
            y0 = self.net(x, use_reference=False, apply_illum=False,
                          apply_ref_illum=False)
            yb = bridge_zero_delta(self.net, x)
        self.assertLessEqual(float((y0 - yb).abs().max()), 1e-6)

    def test_fh2_shape_and_align(self):
        x = torch.randn(2, 3, 32, 36)
        f, ctx = prefix_to_h2(self.net, x)
        self.assertEqual(tuple(f.shape), (2, 80, 16, 18))
        self.assertEqual(tuple(ctx['fea_encoder0'].shape)[-2:], (32, 36))
