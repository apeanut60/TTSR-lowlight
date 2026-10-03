"""V3-A.7.1 unit tests: U vs q_raw, image U, regret, no-H-in-features."""

import inspect
import unittest

import numpy as np
import torch

from v3a5_runtime import action_optimal_target, prepare_geometry, target_geometry
from v3a71_runtime import (decision_regret, image_utility, mse_image,
                           ranked_bins)
from v3a7_runtime import block_utility, q_raw_unclipped


class TestV3A71UtilityMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.geom = prepare_geometry(target_geometry(40, 60, 'g64'), 'cpu')
        self.y0 = torch.randn(1, 3, 40, 60)
        self.D = torch.randn(1, 3, 40, 60) * 0.15
        self.H = self.y0 + 0.4 * self.D + 0.05 * torch.randn(1, 3, 40, 60)

    def test_u_pos_iff_qraw_gt_half(self):
        U = block_utility(self.y0, self.H, self.D, self.geom)
        qraw = q_raw_unclipped(self.y0, self.H, self.D, self.geom)
        # ignore near-zero D blocks
        z = (self.D ** 2).sum(dim=1, keepdim=True)
        from v3a5_runtime import block_mean
        zb = block_mean(z, self.geom).reshape_as(U)
        m = zb.reshape(-1) > 1e-8
        u = U.reshape(-1)[m]
        q = qraw.reshape(-1)[m]
        lhs = (u > 0)
        rhs = (q > 0.5)
        self.assertGreater(int(m.sum()), 10)
        self.assertTrue(torch.equal(lhs, rhs))

        tgt = action_optimal_target(self.y0, self.H, self.D, self.geom)
        # clipped q* agrees on sign except when q_raw outside [0,1]
        qstar = tgt['q_grid'].reshape(-1)[m]
        interior = (q > 0) & (q < 1)
        self.assertTrue(torch.equal((qstar[interior] > 0.5), (q[interior] > 0.5)))

    def test_positive_utility_when_h_is_y0_plus_d(self):
        H = self.y0 + self.D
        U = block_utility(self.y0, H, self.D, self.geom)
        self.assertGreater(float(U.min()), 0.0)
        self.assertGreater(float(image_utility(self.y0, H, self.D)), 0.0)

    def test_negative_utility_when_h_is_y0(self):
        U = block_utility(self.y0, self.y0, self.D, self.geom)
        self.assertLess(float(U.max()), 0.0)
        self.assertLess(float(image_utility(self.y0, self.y0, self.D)), 0.0)


class TestV3A71ImageUtility(unittest.TestCase):
    def test_matches_direct_mse(self):
        torch.manual_seed(2)
        y0 = torch.randn(1, 3, 16, 16)
        H = torch.randn(1, 3, 16, 16)
        D = torch.randn(1, 3, 16, 16) * 0.2
        u = float(image_utility(y0, H, D))
        direct = float(mse_image(y0, H) - mse_image(y0 + D, H))
        self.assertAlmostEqual(u, direct, places=6)


class TestV3A71Regret(unittest.TestCase):
    def test_oracle_pick_has_zero_regret(self):
        mse_b, mse_r = 0.10, 0.04
        self.assertAlmostEqual(decision_regret(mse_r, mse_b, mse_r), 0.0)
        self.assertGreater(decision_regret(mse_b, mse_b, mse_r), 0.0)

    def test_ranked_bins_order(self):
        pred = np.array([3, 2, 1, 0], dtype=np.float64)
        true = np.array([1.0, 0.5, -0.2, -1.0])
        b = ranked_bins(pred, true, fractions=(0.25, 0.50))
        self.assertGreater(b['top_25']['mean_u'], b['bottom_50']['mean_u'])


class TestV3A71NoHInFeatures(unittest.TestCase):
    def test_extract_z_signature(self):
        from scripts.audit_v3a71_utility_predictability import extract_z
        names = inspect.signature(extract_z).parameters
        for bad in ('H', 'h', 'q_star', 'U', 'qstar'):
            self.assertNotIn(bad, names)


if __name__ == '__main__':
    unittest.main()
