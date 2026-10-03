"""V3-A.6 unit tests: loss identities, init equality, freeze, q* independence."""

import os
import tempfile
import unittest

import torch

from model.V3A6DecisionVerifier import (build_v3a6_model,
                                        build_v3a6_shared_init)
from v3a5_runtime import (action_optimal_target, block_energy, energy_mask,
                          expand_gate, prepare_geometry, snapshot_,
                          target_geometry)
from v3a6_runtime import (arm_loss, decision_mse_loss, qstar_regression_loss,
                          require_ckpt)


class TestV3A6Loss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = 'cpu'
        self.geom = prepare_geometry(target_geometry(40, 60, 'g64'), self.device)
        # tiny synthetic tensors matching geom H/W
        H, W = 40, 60
        self.y0 = torch.randn(1, 3, H, W)
        self.Hgt = torch.randn(1, 3, H, W)
        self.D = torch.randn(1, 3, H, W) * 0.1
        self.mask = torch.ones(1, 1, *self.geom['shape'])

    def test_q0_identity(self):
        q = torch.zeros(1, 1, *self.geom['shape'])
        qf = expand_gate(q, self.geom)
        y = self.y0 + qf * self.D
        self.assertTrue(torch.allclose(y, self.y0))

    def test_q1_identity(self):
        q = torch.ones(1, 1, *self.geom['shape'])
        qf = expand_gate(q, self.geom)
        y = self.y0 + qf * self.D
        self.assertTrue(torch.allclose(y, self.y0 + self.D))

    def test_a1_independent_of_qstar(self):
        q = torch.rand(1, 1, *self.geom['shape'], requires_grad=True)
        qstar_a = torch.rand(1, 1, *self.geom['shape'])
        qstar_b = torch.rand(1, 1, *self.geom['shape'])
        loss_a, _ = arm_loss('A1_decision_mse', q, self.y0, self.Hgt, self.D,
                             self.geom, self.mask, q_star=qstar_a)
        g_a = torch.autograd.grad(loss_a, q, retain_graph=True)[0].clone()
        loss_b, _ = arm_loss('A1_decision_mse', q, self.y0, self.Hgt, self.D,
                             self.geom, self.mask, q_star=qstar_b)
        g_b = torch.autograd.grad(loss_b, q)[0]
        self.assertTrue(torch.allclose(loss_a, loss_b))
        self.assertTrue(torch.allclose(g_a, g_b))

    def test_a0_depends_on_qstar(self):
        q = torch.rand(1, 1, *self.geom['shape'], requires_grad=True)
        qstar_a = torch.zeros(1, 1, *self.geom['shape'])
        qstar_b = torch.ones(1, 1, *self.geom['shape'])
        loss_a = arm_loss('A0_qstar', q, self.y0, self.Hgt, self.D,
                          self.geom, self.mask, q_star=qstar_a)[0]
        loss_b = arm_loss('A0_qstar', q, self.y0, self.Hgt, self.D,
                          self.geom, self.mask, q_star=qstar_b)[0]
        self.assertFalse(torch.allclose(loss_a, loss_b))


class TestV3A6Init(unittest.TestCase):
    def test_step0_arm_equality(self):
        init = build_v3a6_shared_init(seed=42)
        a0 = build_v3a6_model('A0_qstar')
        a1 = build_v3a6_model('A1_decision_mse')
        a0.load_state_dict(init['A0_qstar'], strict=True)
        a1.load_state_dict(init['A1_decision_mse'], strict=True)
        a0.eval(); a1.eval()
        geom = prepare_geometry(target_geometry(40, 60, 'g64'), 'cpu')
        x = torch.randn(1, 3, 40, 60)
        y0 = torch.randn(1, 3, 40, 60)
        r = torch.randn(1, 3, 40, 60)
        with torch.no_grad():
            q0 = a0(x, y0, r, geom=geom)
            q1 = a1(x, y0, r, geom=geom)
        self.assertLessEqual(float((q0 - q1).abs().max()), 1e-6)
        # residual zero
        self.assertEqual(a0.context_residual_max_abs(), 0.0)


class TestV3A6Lock(unittest.TestCase):
    def test_exact_checkpoint_missing(self):
        with self.assertRaises(SystemExit):
            require_ckpt('/tmp/definitely_missing_v3a6_ckpt.pt', formal=True)


if __name__ == '__main__':
    unittest.main()
