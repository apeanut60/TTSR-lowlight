"""V4 encoder/matcher stay frozen across a head step."""

import unittest

import torch

from model.LocalRefine import LocalSoftMatch, SharedEncoder
from model.V3BResidualFusion import V3B0ResidualFusion
from model.V4RefCanvas import features_a1
from v3a5_runtime import bit_equal, snapshot_


class TestV4Freeze(unittest.TestCase):
    def test_encoder_match_bit_equal(self):
        torch.manual_seed(0)
        enc = SharedEncoder(32).eval()
        mt = LocalSoftMatch(32, radius=2).eval()
        for p in list(enc.parameters()) + list(mt.parameters()):
            p.requires_grad_(False)

        class W:
            pass
        W.encoder = enc
        W.match = mt
        snap_e, snap_m = snapshot_(enc), snapshot_(mt)
        head = V3B0ResidualFusion(96)
        with torch.no_grad():
            head.out.weight.fill_(0.01)
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        fr, t = features_a1(W, y0, r)
        y = r + head(fr.detach(), t.detach(), r.shape[-2:])
        y.pow(2).mean().backward()
        torch.optim.Adam(head.parameters(), lr=1e-4).step()
        self.assertTrue(bit_equal(snap_e, snapshot_(enc)))
        self.assertTrue(bit_equal(snap_m, snapshot_(mt)))
        self.assertIsNotNone(head.out.weight.grad)
