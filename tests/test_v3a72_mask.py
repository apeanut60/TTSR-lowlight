"""T1: invalid-energy blocks must have q==0."""

import unittest

import numpy as np

from v3a72_runtime import apply_selective_q, per_image_topk_q


class TestV3A72Mask(unittest.TestCase):
    def test_invalid_forced_zero_even_if_score_high(self):
        score = np.array([10.0, 10.0, 10.0, -1.0])
        mask = np.array([1.0, 0.0, 1.0, 1.0])
        q = apply_selective_q(score, tau=0.0, mask=mask)
        self.assertEqual(q[1], 0.0)
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[2], 1.0)
        self.assertEqual(q[3], 0.0)

    def test_topk_ignores_invalid(self):
        score = np.array([5.0, 100.0, 4.0, 3.0])
        mask = np.array([1.0, 0.0, 1.0, 1.0])
        q = per_image_topk_q(score, mask, coverage=1.0 / 3.0)
        self.assertEqual(q[1], 0.0)
        self.assertEqual(float(q.sum()), 1.0)
        self.assertEqual(q[0], 1.0)


if __name__ == '__main__':
    unittest.main()
