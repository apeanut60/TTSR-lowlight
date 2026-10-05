"""B4 adapter identity / leakage / T response."""

import inspect
import unittest

import torch

from model.V3BFeatureResidual import FeatureResidualAdapter
from model.V3BResidualFusion import V3B0ResidualFusion
from v3b_runtime import assert_no_gt_in_forward


class TestV3B4Identity(unittest.TestCase):
    def test_step0_delta_f_zero(self):
        torch.manual_seed(0)
        ad = FeatureResidualAdapter()
        f0 = torch.randn(1, 32, 10, 12)
        t = torch.randn(1, 32, 10, 12)
        d = ad(f0, t)
        self.assertEqual(tuple(d.shape), (1, 80, 10, 12))
        self.assertLessEqual(float(d.abs().max()), 1e-7)

    def test_a0_step0_rgb_zero(self):
        h = V3B0ResidualFusion(96)
        f0 = torch.randn(1, 32, 8, 8)
        t = torch.randn(1, 32, 8, 8)
        self.assertLessEqual(float(h(f0, t, (16, 16)).abs().max()), 1e-7)

    def test_no_gt_leakage(self):
        assert_no_gt_in_forward(FeatureResidualAdapter.forward)
        names = set(inspect.signature(FeatureResidualAdapter.forward).parameters)
        for bad in ('H', 'q_star', 'U', 'AO', 'D', 'g_v2', 'X'):
            self.assertNotIn(bad, names)

    def test_changing_t_moves_output_after_unzero(self):
        torch.manual_seed(1)
        ad = FeatureResidualAdapter()
        with torch.no_grad():
            ad.net[-1].weight.fill_(0.01)
        f0 = torch.randn(1, 32, 8, 8)
        t1 = torch.randn(1, 32, 8, 8)
        t2 = torch.zeros_like(t1)
        self.assertGreater(float((ad(f0, t1) - ad(f0, t2)).abs().max()), 1e-8)

    def test_zero_t_forward(self):
        ad = FeatureResidualAdapter()
        f0 = torch.randn(1, 32, 8, 8)
        z = torch.zeros_like(f0)
        d = ad(f0, z)
        self.assertEqual(tuple(d.shape), tuple(f0.shape[:1]) + (80,) + tuple(f0.shape[-2:]))
