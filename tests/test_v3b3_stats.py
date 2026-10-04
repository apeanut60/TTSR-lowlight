"""B3 RGB global stats exactness / normalize / shape."""

import inspect
import unittest

import torch

from model.V3BGlobalStats import (GLOBAL_STAT_NAMES, broadcast_g, normalize_global_stats,
                                  rgb01_mean_std)
from v3b_runtime import assert_no_gt_in_forward


class TestV3B3Stats(unittest.TestCase):
    def test_no_gt_in_rgb_stats(self):
        assert_no_gt_in_forward(rgb01_mean_std)
        names = set(inspect.signature(rgb01_mean_std).parameters)
        for bad in ('H', 'q_star', 'U', 'AO', 'D', 'g_v2'):
            self.assertNotIn(bad, names)

    def test_synthetic_mean_std_exact(self):
        x = torch.zeros(1, 3, 4, 6)
        x[:, 0] = 1.0   # R01 = 1.0
        x[:, 1] = -1.0  # G01 = 0.0
        x[:, 2] = 0.0   # B01 = 0.5
        s = rgb01_mean_std(x)
        self.assertEqual(tuple(s.shape), (1, 6))
        self.assertAlmostEqual(float(s[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(s[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(s[0, 2]), 0.5, places=6)
        self.assertAlmostEqual(float(s[0, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(s[0, 4]), 0.0, places=6)
        self.assertAlmostEqual(float(s[0, 5]), 0.0, places=6)

    def test_std_unbiased_false(self):
        x = torch.zeros(1, 3, 2, 2)
        x[0, 0, 0, 0] = 1.0
        x[0, 0, 0, 1] = -1.0
        x[0, 0, 1, 0] = 1.0
        x[0, 0, 1, 1] = -1.0
        # R01 values: 1,0,1,0  mean=0.5  population std = 0.5
        s = rgb01_mean_std(x)
        r01 = torch.tensor([1.0, 0.0, 1.0, 0.0])
        want = float(r01.std(unbiased=False))
        self.assertAlmostEqual(float(s[0, 3]), want, places=6)
        self.assertNotAlmostEqual(want, float(r01.std(unbiased=True)), places=6)

    def test_changing_r_changes_stats(self):
        a = torch.zeros(1, 3, 8, 8)
        b = torch.ones(1, 3, 8, 8)
        self.assertGreater(float((rgb01_mean_std(a) - rgb01_mean_std(b)).abs().max()),
                           1e-6)

    def test_normalize_clip_and_broadcast(self):
        stats = {n: dict(mean=0.0, std=1.0) for n in GLOBAL_STAT_NAMES}
        s = torch.ones(2, 6) * 10
        out = normalize_global_stats(s, stats)
        self.assertLessEqual(float(out.max()), 5.0)
        g = torch.randn(2, 64)
        G = broadcast_g(g, (10, 12))
        self.assertEqual(tuple(G.shape), (2, 64, 10, 12))
        self.assertTrue(torch.allclose(G[:, :, 0, 0], g))
        self.assertTrue(torch.allclose(G[:, :, 3, 4], g))


if __name__ == '__main__':
    unittest.main()
