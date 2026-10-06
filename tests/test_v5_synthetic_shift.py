"""Synthetic integer shift must be recovered by hard correspondence. HARD STOP otherwise."""

import unittest

import torch
import torch.nn.functional as F

from model.V5Correspondence import ChunkedHardMatcher


def _unique_feat(h=16, w=20, c=8):
    y = torch.arange(h).float().view(1, 1, h, 1).expand(1, 1, h, w)
    x = torch.arange(w).float().view(1, 1, 1, w).expand(1, 1, h, w)
    extra = torch.randn(1, c - 2, h, w)
    return torch.cat([y, x, extra], dim=1)


class TestV5SyntheticShift(unittest.TestCase):
    def test_recover_dx3_dy_minus2(self):
        torch.manual_seed(1)
        dx, dy = 3, -2
        tgt = _unique_feat()
        ref = torch.roll(tgt, shifts=(dy, dx), dims=(-2, -1))
        m = ChunkedHardMatcher(chunk_size=32)
        out = m(tgt, ref)
        h, w = tgt.shape[-2:]
        ys = slice(max(0, -dy) + 1, h - max(0, dy) - 1)
        xs = slice(max(0, -dx) + 1, w - max(0, dx) - 1)
        fx = out.flow[0, 0, ys, xs]
        fy = out.flow[0, 1, ys, xs]
        ok = (fx.round() == dx) & (fy.round() == dy)
        rate = float(ok.float().mean())
        err = torch.stack([fx - dx, fy - dy], 0).pow(2).sum(0).sqrt()
        self.assertGreaterEqual(rate, 0.95, 'recovery=%s' % rate)
        self.assertLessEqual(float(err.median()), 0.5)
