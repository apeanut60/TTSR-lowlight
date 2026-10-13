"""Run: python -m unittest discover -s tests -v"""
import unittest
import torch
from model.td_transfer import TaskDecoupledTransferHead, td_reconstruction_losses

class TestTaskDecoupledTransfer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.net = TaskDecoupledTransferHead(16).eval()
        self.y0 = torch.rand(1, 3, 32, 32)
        self.tex = torch.randn(1, 16, 8, 8)
        self.c = torch.rand(1, 1, 8, 8)

    def test_two_outputs_and_identity_init(self):
        with torch.no_grad():
            d = self.net(self.y0, self.tex, self.c)
        self.assertEqual(d['i_tex'].shape, self.y0.shape)
        self.assertEqual(d['y'].shape, self.y0.shape)
        self.assertLessEqual(float((d['y'] - self.y0).abs().max()), 1e-7)

    def test_eq10_gate_zero_confidence(self):
        with torch.no_grad():
            d = self.net(self.y0, self.tex, torch.zeros_like(self.c))
            base = self.net.content(self.y0)
        self.assertLessEqual(float((d['fused'] - base).abs().max()), 1e-7)
        self.assertLessEqual(float(d['gated_texture'].abs().max()), 1e-7)

    def test_eq11_loss_and_grad(self):
        gt = torch.rand_like(self.y0)
        d = self.net(self.y0, self.tex, self.c)
        loss = td_reconstruction_losses(d, gt)
        self.assertTrue(torch.isfinite(loss['loss']))
        torch.testing.assert_close(loss['l_tex'],
            (gt-(self.y0 + d['i_tex'])).abs().mean())
        loss['loss'].backward()
        self.assertGreater(float(self.net.texture_decoder.out.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.net.att.rgb_out.weight.grad.abs().sum()), 0)

    def test_confidence_range_and_shapes(self):
        d=self.net(self.y0, self.tex, -torch.ones_like(self.c))
        self.assertEqual(float(d['confidence'].max()), 0.0)
        with self.assertRaises(ValueError):
            self.net(self.y0, self.tex, self.c[..., :-1])

if __name__ == '__main__':
    unittest.main()
