"""T4 B1a identity; T6 reference path; T7 diagnostic modes."""

import unittest

import torch

from model.LocalRefine import Refiner
from model.V3BResidualFusion import V3B0ResidualFusion
from model.V3BReferenceAdapt import (
    V3B1Model, V3B1aIdentityAdapt, V3B1bMasaAdapt)
from v3b_runtime import assert_no_gt_in_forward, b0_forward, match_features, match_features_mode


class TestV3BReferencePath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.core = Refiner(ch=32, radius=4, tau=0.1, chunk=64).eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.head = V3B0ResidualFusion()
        # break identity so output can move with T: set a tiny out weight
        with torch.no_grad():
            self.head.out.weight.fill_(0.01)
            self.head.out.bias.zero_()
        self.y0 = torch.randn(1, 3, 40, 60)

    def test_b1a_identity_at_init(self):
        adapt = V3B1aIdentityAdapt(32)
        f0 = torch.randn(2, 32, 8, 8)
        t = torch.randn(2, 32, 8, 8)
        ta = adapt(f0, t)
        self.assertLessEqual(float((ta - t).abs().max()), 1e-6)

    def test_b1b_step0_not_identity(self):
        adapt = V3B1bMasaAdapt(32)
        f0 = torch.randn(2, 32, 8, 8)
        t = torch.randn(2, 32, 8, 8) + 3.0
        ta = adapt(f0, t)
        self.assertGreater(float((ta - t).abs().max()), 1e-3)

    def test_b1_model_step0_y_equals_y0(self):
        f0 = torch.randn(1, 32, 8, 8)
        t = torch.randn(1, 32, 8, 8) + 2.0
        y0 = torch.randn(1, 3, 16, 16)
        ma = V3B1Model('B1a_identity_adapt')
        mb = V3B1Model('B1b_masa_adapt')
        da, auxa = ma(f0, t, y0.shape[-2:], return_aux=True)
        db, auxb = mb(f0, t, y0.shape[-2:], return_aux=True)
        self.assertLessEqual(float(da.abs().max()), 1e-7)
        self.assertLessEqual(float(db.abs().max()), 1e-7)
        self.assertLessEqual(float((auxa['t_adapt'] - t).abs().max()), 1e-6)
        self.assertGreater(float((auxb['t_adapt'] - t).abs().max()), 1e-3)
        assert_no_gt_in_forward(V3B1Model.forward)

    def test_different_r_changes_t_and_output(self):
        r1 = torch.randn(1, 3, 40, 60)
        r2 = torch.randn(1, 3, 40, 60)
        f0a, t1 = match_features(self.core, self.y0, r1)
        f0b, t2 = match_features(self.core, self.y0, r2)
        self.assertLessEqual(float((f0a - f0b).abs().max()), 1e-6)
        self.assertGreater(float((t1 - t2).abs().max()), 1e-6)
        y1, _ = b0_forward(self.head, f0a, t1, self.y0)
        y2, _ = b0_forward(self.head, f0b, t2, self.y0)
        self.assertGreater(float((y1 - y2).abs().max()), 1e-8)

    def test_diagnostic_modes_run(self):
        r = torch.randn(1, 3, 40, 60)
        for mode in ('normal', 'self', 'zero', 'shuffled'):
            f0, t = match_features_mode(self.core, self.y0, r, mode=mode)
            self.assertEqual(tuple(f0.shape), tuple(t.shape))
            y, d = b0_forward(self.head, f0, t, self.y0)
            self.assertEqual(tuple(y.shape), tuple(self.y0.shape))
            if mode == 'zero':
                self.assertEqual(float(t.abs().max()), 0.0)


if __name__ == '__main__':
    unittest.main()
