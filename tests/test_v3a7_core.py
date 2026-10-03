"""V3-A.7: utility identities, independence from q*, dependence on H."""

import unittest

import torch

from v3a5_runtime import prepare_geometry, target_geometry
from v3a7_runtime import accept_target, arm_loss, block_utility, utility_bce_loss


class TestV3A7Utility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.geom = prepare_geometry(target_geometry(40, 60, 'g64'), 'cpu')
        H, W = 40, 60
        self.y0 = torch.zeros(1, 3, H, W)
        self.Hgt = torch.zeros(1, 3, H, W)
        self.D = torch.ones(1, 3, H, W) * 0.2
        self.mask = torch.ones(1, 1, *self.geom['shape'])

    def test_utility_positive_when_D_helps(self):
        # H = Y0+D → applying D is perfect
        H = self.y0 + self.D
        U = block_utility(self.y0, H, self.D, self.geom)
        self.assertGreater(float(U.min()), 0.0)
        self.assertTrue(torch.equal(accept_target(U), torch.ones_like(U)))

    def test_utility_negative_when_D_hurts(self):
        # H = Y0 → applying D moves away
        U = block_utility(self.y0, self.y0, self.D, self.geom)
        self.assertLess(float(U.max()), 0.0)
        self.assertTrue(torch.equal(accept_target(U), torch.zeros_like(U)))

    def test_loss_independent_of_qstar(self):
        q = torch.rand(1, 1, *self.geom['shape'], requires_grad=True)
        H = self.y0 + 0.5 * self.D
        loss_a, _ = arm_loss('A1_utility_bce', q, self.y0, H, self.D,
                             self.geom, self.mask, q_star=torch.zeros_like(q))
        g_a = torch.autograd.grad(loss_a, q, retain_graph=True)[0].clone()
        loss_b, _ = arm_loss('A1_utility_bce', q, self.y0, H, self.D,
                             self.geom, self.mask, q_star=torch.ones_like(q))
        g_b = torch.autograd.grad(loss_b, q)[0]
        self.assertTrue(torch.allclose(loss_a, loss_b))
        self.assertTrue(torch.allclose(g_a, g_b))

    def test_loss_depends_on_H(self):
        q = torch.full((1, 1, *self.geom['shape']), 0.7, requires_grad=True)
        loss_help, _ = utility_bce_loss(
            q, self.y0, self.y0 + self.D, self.D, self.geom, self.mask)
        loss_hurt, _ = utility_bce_loss(
            q, self.y0, self.y0, self.D, self.geom, self.mask)
        self.assertFalse(torch.allclose(loss_help, loss_hurt))
        # q=0.7 should prefer the helpful target
        self.assertLess(float(loss_help), float(loss_hurt))


if __name__ == '__main__':
    unittest.main()
