"""B3 shared A0→A1 initialization."""

import unittest

import torch

from model.V3BGlobalStats import V3B3A1, assert_shared_init, copy_a0_to_a1_head, global_first_layer_norms
from model.V3BResidualFusion import V3B0ResidualFusion


class TestV3B3SharedInit(unittest.TestCase):
    def test_a0_a1_step0_equal(self):
        torch.manual_seed(0)
        a0 = V3B0ResidualFusion(96)
        a1 = V3B3A1()
        copy_a0_to_a1_head(a0, a1.head)
        f0 = torch.randn(2, 32, 10, 12)
        t = torch.randn(2, 32, 10, 12)
        s = torch.randn(2, 6)
        diff = assert_shared_init(a0, a1, f0, t, s, (20, 24))
        self.assertLessEqual(diff, 1e-7)
        self.assertLessEqual(float(a1(f0, t, (20, 24), s).abs().max()), 1e-7)

    def test_global_slice_zero_and_common(self):
        a0 = V3B0ResidualFusion(96)
        a1 = V3B3A1()
        copy_a0_to_a1_head(a0, a1.head)
        w = a1.head.stem[0].weight
        self.assertEqual(tuple(w.shape), (64, 160, 3, 3))
        self.assertEqual(float(w[:, 96:].abs().max()), 0.0)
        self.assertTrue(torch.equal(w[:, :96], a0.stem[0].weight))
        n = global_first_layer_norms(a1)
        self.assertEqual(n['l1'], 0.0)
        self.assertEqual(n['l2'], 0.0)

    def test_changing_global_can_change_after_nonzero_slice(self):
        torch.manual_seed(1)
        a1 = V3B3A1()
        with torch.no_grad():
            a1.head.stem[0].weight[:, 96:].fill_(0.01)
            a1.head.out.weight.fill_(0.01)
        f0 = torch.randn(1, 32, 8, 8)
        t = torch.randn(1, 32, 8, 8)
        s1 = torch.zeros(1, 6)
        s2 = torch.ones(1, 6)
        d1 = a1(f0, t, (16, 16), s1)
        d2 = a1(f0, t, (16, 16), s2)
        self.assertGreater(float((d1 - d2).abs().max()), 1e-8)


if __name__ == '__main__':
    unittest.main()
