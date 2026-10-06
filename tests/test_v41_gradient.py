"""V4.1a gradient through frozen matcher to Grounder; frozen params have no grad."""

import unittest

import torch

from model.LocalRefine import LocalSoftMatch, SharedEncoder
from model.V3BResidualFusion import V3B0ResidualFusion
from model.V4RefGrounder import RefGrounder, frozen_pair_and_tlow, grounded_forward


class TestV41Gradient(unittest.TestCase):
    def test_grad_reaches_grounder_not_frozen(self):
        torch.manual_seed(0)
        enc = SharedEncoder(32).eval()
        mt = LocalSoftMatch(32, radius=2).eval()
        head = V3B0ResidualFusion(96).eval()
        with torch.no_grad():
            head.out.weight.fill_(0.01)
        for p in list(enc.parameters()) + list(mt.parameters()) + list(head.parameters()):
            p.requires_grad_(False)

        class W:
            pass
        W.encoder = enc
        W.match = mt
        g = RefGrounder()
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        h = torch.randn(1, 3, 32, 32)
        f0, fr, t_low, _tr = frozen_pair_and_tlow(W, y0, r)
        y, _d, _fs, _ts = grounded_forward(mt, head, f0, fr, t_low, y0, g)
        (y - h).pow(2).mean().backward()
        gsum = 0.0
        for p in g.parameters():
            if p.grad is not None:
                gsum += float(p.grad.abs().sum())
        self.assertGreater(gsum, 0.0)
        for name, mod in (('match', mt), ('encoder', enc), ('b0', head)):
            for p in mod.parameters():
                self.assertIsNone(p.grad, msg='%s has grad' % name)
