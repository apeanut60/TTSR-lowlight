"""V4.1a frozen encoder/matcher/B0 stay bit-equal after a grounder step."""

import unittest

import torch

from model.LocalRefine import LocalSoftMatch, SharedEncoder
from model.V3BResidualFusion import V3B0ResidualFusion
from model.V4RefGrounder import RefGrounder, frozen_pair_and_tlow, grounded_forward
from v3a5_runtime import bit_equal, snapshot_


class TestV41Freeze(unittest.TestCase):
    def test_frozen_bit_equal_after_step(self):
        torch.manual_seed(0)
        enc = SharedEncoder(32).eval()
        mt = LocalSoftMatch(32, radius=2).eval()
        head = V3B0ResidualFusion(96).eval()
        for p in list(enc.parameters()) + list(mt.parameters()) + list(head.parameters()):
            p.requires_grad_(False)

        class W:
            pass
        W.encoder = enc
        W.match = mt
        snap = dict(e=snapshot_(enc), m=snapshot_(mt), h=snapshot_(head))
        g = RefGrounder()
        with torch.no_grad():
            g.net[-1].weight.fill_(0.001)
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        f0, fr, t_low, _tr = frozen_pair_and_tlow(W, y0, r)
        y, _d, _fs, _ts = grounded_forward(mt, head, f0, fr, t_low, y0, g)
        y.pow(2).mean().backward()
        torch.optim.Adam(g.parameters(), lr=1e-4).step()
        self.assertTrue(bit_equal(snap['e'], snapshot_(enc)))
        self.assertTrue(bit_equal(snap['m'], snapshot_(mt)))
        self.assertTrue(bit_equal(snap['h'], snapshot_(head)))
        self.assertIsNotNone(g.net[-1].weight.grad)
