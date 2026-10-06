"""V4.1a grounding connectivity and T_low ablations."""

import unittest

import torch

from model.LocalRefine import LocalSoftMatch, SharedEncoder
from model.V4RefGrounder import RefGrounder, frozen_pair_and_tlow


class _W(object):
    def __init__(self):
        self.encoder = SharedEncoder(32).eval()
        self.match = LocalSoftMatch(32, radius=2).eval()
        for p in list(self.encoder.parameters()) + list(self.match.parameters()):
            p.requires_grad_(False)


class TestV41Grounding(unittest.TestCase):
    def test_changing_f0_changes_tlow(self):
        torch.manual_seed(0)
        w = _W()
        y0a = torch.randn(1, 3, 32, 32)
        y0b = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        _f, fr_a, tl_a, _t = frozen_pair_and_tlow(w, y0a, r)
        _f, fr_b, tl_b, _t = frozen_pair_and_tlow(w, y0b, r)
        self.assertLessEqual(float((fr_a - fr_b).abs().max()), 1e-6)
        self.assertGreater(float((tl_a - tl_b).abs().max()), 1e-5)

    def test_changing_tlow_moves_trained_delta(self):
        torch.manual_seed(1)
        g = RefGrounder()
        with torch.no_grad():
            g.net[-1].weight.fill_(0.01)
        fr = torch.randn(1, 32, 8, 8)
        a = torch.randn(1, 32, 8, 8)
        b = torch.zeros_like(a)
        self.assertGreater(float((g(fr, a) - g(fr, b)).abs().max()), 1e-8)

    def test_self_zero_shuffled_modes(self):
        torch.manual_seed(2)
        w = _W()
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        donor = torch.randn(1, 3, 32, 32)
        _f, fr_n, tl_n, _t = frozen_pair_and_tlow(w, y0, r, tlow_mode='normal')
        _f, fr_s, tl_s, _t = frozen_pair_and_tlow(w, y0, r, tlow_mode='self')
        _f, fr_z, tl_z, _t = frozen_pair_and_tlow(w, y0, r, tlow_mode='zero')
        _f, fr_d, tl_d, _t = frozen_pair_and_tlow(
            w, y0, r, y0_donor=donor, tlow_mode='shuffled')
        self.assertLessEqual(float((fr_n - fr_s).abs().max()), 1e-6)
        self.assertEqual(float(tl_z.abs().max()), 0.0)
        self.assertGreater(float((tl_n - tl_s).abs().max()), 1e-5)
        self.assertGreater(float((tl_n - tl_d).abs().max()), 1e-5)
