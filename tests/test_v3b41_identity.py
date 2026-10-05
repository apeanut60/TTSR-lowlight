"""B4.1 identity / leakage / input channels / T response."""

import inspect
import unittest

import torch

from model.V3BBaseConditionedFeatureResidual import (
    IN_CH, BaseConditionedFeatureResidual)
from model.V3BResidualFusion import V3B0ResidualFusion
from v3b_runtime import assert_no_gt_in_forward


class TestV3B41Identity(unittest.TestCase):
    def test_in_ch_176(self):
        self.assertEqual(IN_CH, 176)
        ad = BaseConditionedFeatureResidual()
        self.assertEqual(ad.in_ch, 176)
        self.assertEqual(ad.out_ch, 80)

    def test_step0_delta_f_zero(self):
        torch.manual_seed(0)
        ad = BaseConditionedFeatureResidual()
        fd = torch.randn(1, 80, 10, 12)
        f0 = torch.randn(1, 32, 10, 12)
        t = torch.randn(1, 32, 10, 12)
        d = ad(fd, f0, t)
        self.assertEqual(tuple(d.shape), (1, 80, 10, 12))
        self.assertLessEqual(float(d.abs().max()), 1e-7)

    def test_a0_step0_rgb_zero(self):
        h = V3B0ResidualFusion(96)
        f0 = torch.randn(1, 32, 8, 8)
        t = torch.randn(1, 32, 8, 8)
        self.assertLessEqual(float(h(f0, t, (16, 16)).abs().max()), 1e-7)

    def test_spatial_align_hard(self):
        ad = BaseConditionedFeatureResidual()
        fd = torch.randn(1, 80, 8, 8)
        f0 = torch.randn(1, 32, 7, 8)
        t = torch.randn(1, 32, 7, 8)
        with self.assertRaises(SystemExit):
            ad(fd, f0, t)

    def test_no_gt_leakage(self):
        assert_no_gt_in_forward(BaseConditionedFeatureResidual.forward)
        names = set(inspect.signature(
            BaseConditionedFeatureResidual.forward).parameters)
        for bad in ('H', 'q_star', 'U', 'AO', 'D', 'g_v2', 'X', 'E'):
            self.assertNotIn(bad, names)

    def test_changing_t_moves_output_after_unzero(self):
        torch.manual_seed(1)
        ad = BaseConditionedFeatureResidual()
        with torch.no_grad():
            ad.net[-1].weight.fill_(0.01)
        fd = torch.randn(1, 80, 8, 8)
        f0 = torch.randn(1, 32, 8, 8)
        t1 = torch.randn(1, 32, 8, 8)
        t2 = torch.zeros_like(t1)
        self.assertGreater(float((ad(fd, f0, t1) - ad(fd, f0, t2)).abs().max()), 1e-8)

    def test_zero_t_forward(self):
        ad = BaseConditionedFeatureResidual()
        fd = torch.randn(1, 80, 8, 8)
        f0 = torch.randn(1, 32, 8, 8)
        z = torch.zeros_like(f0)
        d = ad(fd, f0, z)
        self.assertEqual(tuple(d.shape), (1, 80, 8, 8))
