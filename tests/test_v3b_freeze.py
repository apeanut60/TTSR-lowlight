"""T2/T3: optimizer step must not change frozen encoder/match."""

import copy
import unittest

import torch

from model.V3BResidualFusion import V3B0ResidualFusion
from v3a5_runtime import bit_equal, snapshot_
from v3b_runtime import b0_forward, b0_loss
from model.LocalRefine import Refiner


class TestV3BFreeze(unittest.TestCase):
    def test_only_head_changes(self):
        torch.manual_seed(0)
        core = Refiner(ch=32, radius=4, tau=0.1, chunk=64)
        for p in core.parameters():
            p.requires_grad_(False)
        head = V3B0ResidualFusion()
        before_enc = snapshot_(core.encoder)
        before_match = snapshot_(core.match)
        y0 = torch.randn(1, 3, 40, 60)
        r = torch.randn(1, 3, 40, 60)
        h = y0 + 0.05 * torch.randn_like(y0)
        with torch.no_grad():
            f0 = core.encoder((y0 + 1) * 0.5)
            t = core.match(f0, core.encoder((r + 1) * 0.5))
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        y, _ = b0_forward(head, f0.detach(), t.detach(), y0.detach())
        loss = b0_loss(y, h)
        opt.zero_grad()
        loss.backward()
        opt.step()
        self.assertTrue(bit_equal(before_enc, snapshot_(core.encoder)))
        self.assertTrue(bit_equal(before_match, snapshot_(core.match)))
        self.assertGreater(float(head.out.weight.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
