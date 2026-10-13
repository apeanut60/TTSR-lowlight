"""CPU unit tests for V7-RIVER fusion."""
import unittest
import torch

from model.V7Model import V7Model, v7_step0_deltas_zero
from model.V7RiverFusion import V7RiverFusion


class TestV7River(unittest.TestCase):
    def test_step0_identity(self):
        torch.manual_seed(0)
        m = V7Model()
        self.assertTrue(v7_step0_deltas_zero(m))
        d2 = torch.randn(1, 64, 16, 16)
        a2 = torch.randn(1, 64, 16, 16)
        d2s, delta = m.refine_h4(d2, a2)
        self.assertLessEqual(float(delta.abs().max()), 1e-7)
        self.assertLessEqual(float((d2s - d2).abs().max()), 1e-7)

    def test_shapes_d1_d2(self):
        m = V7Model()
        d2, a2 = torch.randn(1, 64, 8, 8), torch.randn(1, 64, 8, 8)
        d1, a1 = torch.randn(1, 32, 16, 16), torch.randn(1, 32, 16, 16)
        o2, _ = m.refine_h4(d2, a2)
        o1, _ = m.refine_h2(d1, a1)
        self.assertEqual(o2.shape, d2.shape)
        self.assertEqual(o1.shape, d1.shape)

    def test_setting_a_freeze(self):
        m = V7Model()
        names = m.apply_setting_freeze('A')
        for n, p in m.named_parameters():
            if n.startswith(('match_encoder', 'texture_encoder', 'align_')):
                self.assertFalse(p.requires_grad, n)
            if n.startswith('refine_'):
                self.assertTrue(p.requires_grad, n)
        self.assertTrue(any('refine_' in n for n in names))

    def test_fusion_grad_from_ref(self):
        torch.manual_seed(1)
        f = V7RiverFusion(64, 64, n_heads=4, head_dim=16)
        # leave zero-init: nudge out so grad flows to q/k/v
        with torch.no_grad():
            f.out_conv.weight.add_(1e-3)
        d = torch.randn(1, 64, 8, 8, requires_grad=False)
        a = torch.randn(1, 64, 8, 8, requires_grad=True)
        out, delta = f(d, a)
        out.sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
