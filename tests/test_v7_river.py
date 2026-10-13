"""CPU unit tests for V7-RIVER fusion."""
import unittest
import torch

from model.V7Model import V7Model, v7_step0_deltas_zero
from model.V7RiverFusion import V7RiverFusion


def _grad_norm(module, prefix):
    total = 0.0
    for n, p in module.named_parameters():
        if n.startswith(prefix) and p.grad is not None:
            total += float(p.grad.detach().norm())
    return total


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
        m = V7Model(spatial_window=3)
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

    def test_d1_disabled_identity(self):
        m = V7Model(d1_enabled=False)
        d1 = torch.randn(1, 32, 16, 16)
        d1s, a1, aux = m.refine_at_h2(d1, None, None, {})
        self.assertTrue(aux.get('d1_disabled'))
        self.assertLessEqual(float((d1s - d1).abs().max()), 0)

    def test_fusion_grad_from_ref(self):
        torch.manual_seed(1)
        f = V7RiverFusion(64, 64, n_heads=4, head_dim=16, spatial_window=3)
        with torch.no_grad():
            f.out_conv.weight.add_(1e-3)
        d = torch.randn(1, 64, 8, 8, requires_grad=False)
        a = torch.randn(1, 64, 8, 8, requires_grad=True)
        out, delta = f(d, a)
        out.sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0)

    def test_n1_qk_grad_zero(self):
        """Legacy N=1: Softmax trivial → Q/K attn-path grads are 0."""
        torch.manual_seed(2)
        f = V7RiverFusion(64, 64, n_heads=4, head_dim=16, spatial_window=1)
        with torch.no_grad():
            f.out_conv.weight.add_(1e-2)
        d = torch.randn(1, 64, 6, 6)
        a = torch.randn(1, 64, 6, 6)
        out, _ = f(d, a)
        out.sum().backward()
        self.assertEqual(_grad_norm(f, 'q_proj'), 0.0)
        self.assertEqual(_grad_norm(f, 'k_proj'), 0.0)
        self.assertGreater(_grad_norm(f, 'v_proj'), 0.0)

    def test_window3_qk_grad_nonzero(self):
        """Window K=3 → N=9 keys; Q/K must receive nonzero grads."""
        torch.manual_seed(3)
        f = V7RiverFusion(64, 64, n_heads=4, head_dim=16, spatial_window=3)
        with torch.no_grad():
            f.out_conv.weight.add_(1e-2)
        d = torch.randn(1, 64, 6, 6)
        a = torch.randn(1, 64, 6, 6)
        out, _ = f(d, a)
        out.sum().backward()
        self.assertGreater(_grad_norm(f, 'q_proj'), 0.0)
        self.assertGreater(_grad_norm(f, 'k_proj'), 0.0)
        self.assertGreater(_grad_norm(f, 'v_proj'), 0.0)
        self.assertEqual(f.n_keys, 9)

    def test_window_kv_layout_center_and_neighbors(self):
        """Exact token values: center + 3x3 neighborhood match padded feat."""
        torch.manual_seed(0)
        b, c, h, w, k = 2, 8, 5, 6, 3
        f = V7RiverFusion(c, c, n_heads=2, head_dim=4, spatial_window=k)
        # Unique values so layout mistakes cannot accidentally match
        feat = torch.arange(b * c * h * w, dtype=torch.float32).reshape(b, c, h, w)
        tok = f._window_kv(feat)  # [B, HW, K*K, C]
        self.assertEqual(tuple(tok.shape), (b, h * w, k * k, c))

        pad = k // 2
        feat_pad = torch.nn.functional.pad(feat, (pad, pad, pad, pad), mode='constant')
        # unfold zero-pads by default (same as F.unfold padding)
        center = (k * k) // 2  # index 4 for 3x3
        for bi in range(b):
            for yi in range(h):
                for xi in range(w):
                    hw = yi * w + xi
                    # center token == feat[bi, :, yi, xi]
                    self.assertTrue(torch.equal(
                        tok[bi, hw, center], feat[bi, :, yi, xi]),
                        'center mismatch at (%d,%d,%d)' % (bi, yi, xi))
                    # full neighborhood
                    n = 0
                    for dy in range(-pad, pad + 1):
                        for dx in range(-pad, pad + 1):
                            expect = feat_pad[bi, :, yi + pad + dy, xi + pad + dx]
                            self.assertTrue(
                                torch.equal(tok[bi, hw, n], expect),
                                'nbr n=%d at (%d,%d,%d)' % (n, bi, yi, xi))
                            n += 1

    def test_window_kv_matches_manual_gather(self):
        """_window_kv equals explicit gather for interior pixels (no pad)."""
        c, h, w, k = 4, 4, 4, 3
        f = V7RiverFusion(c, c, n_heads=1, head_dim=4, spatial_window=k)
        feat = torch.randn(1, c, h, w)
        tok = f._window_kv(feat)[0]  # HW,9,C
        # pixel (1,1): neighborhood feat[:,0:3,0:3]
        patch = feat[0, :, 0:3, 0:3].permute(1, 2, 0).reshape(9, c)
        self.assertTrue(torch.allclose(tok[1 * w + 1], patch, atol=0))

    def test_win3_forward_shapes(self):
        f = V7RiverFusion(64, 64, n_heads=4, head_dim=16, spatial_window=3)
        d = torch.randn(1, 64, 7, 9)
        a = torch.randn(1, 64, 7, 9)
        out, delta = f(d, a)
        self.assertEqual(out.shape, d.shape)
        self.assertEqual(delta.shape, d.shape)


if __name__ == '__main__':
    unittest.main()
