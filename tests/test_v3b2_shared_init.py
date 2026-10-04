"""B2 shared A0→A1 initialization."""

import unittest

import torch

from model.V3BEvidence import assert_shared_init, copy_a0_to_a1, evidence_first_layer_norms
from model.V3BResidualFusion import V3B0ResidualFusion


class TestV3B2SharedInit(unittest.TestCase):
    def test_a0_a1_step0_equal(self):
        torch.manual_seed(0)
        a0 = V3B0ResidualFusion(96)
        a1 = V3B0ResidualFusion(100)
        copy_a0_to_a1(a0, a1)
        f0 = torch.randn(2, 32, 10, 12)
        t = torch.randn(2, 32, 10, 12)
        e = torch.randn(2, 4, 10, 12) * 3
        diff = assert_shared_init(a0, a1, f0, t, e, (20, 24))
        self.assertLessEqual(diff, 1e-7)
        # nonzero evidence input, still zero delta at init
        self.assertLessEqual(float(a1(f0, t, (20, 24), E=e).abs().max()), 1e-7)

    def test_evidence_weight_slice_zero(self):
        a0 = V3B0ResidualFusion(96)
        a1 = V3B0ResidualFusion(100)
        copy_a0_to_a1(a0, a1)
        w = a1.stem[0].weight
        self.assertEqual(tuple(w.shape), (64, 100, 3, 3))
        self.assertEqual(float(w[:, 96:].abs().max()), 0.0)
        self.assertTrue(torch.equal(w[:, :96], a0.stem[0].weight))
        self.assertTrue(torch.equal(a1.stem[0].bias, a0.stem[0].bias))
        # later layers bit-equal
        for (n0, p0), (n1, p1) in zip(a0.named_parameters(), a1.named_parameters()):
            if n0 == 'stem.0.weight':
                continue
            self.assertEqual(n0, n1)
            self.assertTrue(torch.equal(p0, p1), msg=n0)

    def test_norms_zero_at_init(self):
        a0 = V3B0ResidualFusion(96)
        a1 = V3B0ResidualFusion(100)
        copy_a0_to_a1(a0, a1)
        n = evidence_first_layer_norms(a1)
        self.assertEqual(n['l1'], 0.0)
        self.assertEqual(n['l2'], 0.0)


if __name__ == '__main__':
    unittest.main()
