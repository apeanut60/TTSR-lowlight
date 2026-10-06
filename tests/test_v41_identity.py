"""V4.1a identity: ΔFR=0, FR*=FR, T*=T_raw, Y=B0 at init."""

import inspect
import unittest

import torch

from model.LocalRefine import LocalSoftMatch, SharedEncoder
from model.V3BResidualFusion import V3B0ResidualFusion
from model.V4RefGrounder import RefGrounder, frozen_pair_and_tlow, grounded_forward
from v3b_runtime import assert_no_gt_in_forward


class _W(object):
    def __init__(self):
        self.encoder = SharedEncoder(32).eval()
        self.match = LocalSoftMatch(32, radius=2).eval()
        for p in list(self.encoder.parameters()) + list(self.match.parameters()):
            p.requires_grad_(False)


class TestV41Identity(unittest.TestCase):
    def test_delta_zero_and_fr_star(self):
        torch.manual_seed(0)
        g = RefGrounder()
        fr = torch.randn(1, 32, 10, 12)
        tl = torch.randn(1, 32, 10, 12)
        d = g(fr, tl)
        self.assertEqual(tuple(d.shape), (1, 32, 10, 12))
        self.assertLessEqual(float(d.abs().max()), 1e-7)
        self.assertLessEqual(float(((fr + d) - fr).abs().max()), 1e-7)

    def test_t_and_y_match_b0(self):
        torch.manual_seed(1)
        w = _W()
        head = V3B0ResidualFusion(96).eval()
        for p in head.parameters():
            p.requires_grad_(False)
        g = RefGrounder().eval()
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        f0, fr, t_low, t_raw = frozen_pair_and_tlow(w, y0, r)
        y_b0 = y0 + head(f0, t_raw, y0.shape[-2:])
        y, dfr, frs, ts = grounded_forward(w.match, head, f0, fr, t_low, y0, g)
        self.assertLessEqual(float(dfr.abs().max()), 1e-7)
        self.assertLessEqual(float((frs - fr).abs().max()), 1e-7)
        self.assertLessEqual(float((ts - t_raw).abs().max()), 1e-6)
        self.assertLessEqual(float((y - y_b0).abs().max()), 1e-6)
        self.assertEqual(tuple(t_low.shape), (1, 32, 16, 16))
        self.assertEqual(tuple(ts.shape), (1, 32, 16, 16))

    def test_no_gt_leakage(self):
        assert_no_gt_in_forward(RefGrounder.forward)
        names = set(inspect.signature(RefGrounder.forward).parameters)
        for bad in ('H', 'q_star', 'U', 'AO', 'D', 'g_v2', 'X'):
            self.assertNotIn(bad, names)
        names2 = set(inspect.signature(grounded_forward).parameters)
        for bad in ('H', 'q_star', 'U', 'AO'):
            self.assertNotIn(bad, names2)
