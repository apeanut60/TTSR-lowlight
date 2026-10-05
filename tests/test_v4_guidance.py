"""V4 reverse-guidance diagnostic modes."""

import unittest

import torch
from model.LocalRefine import LocalSoftMatch, SharedEncoder
from model.V4RefCanvas import features_a1_mode


class _Wrap(object):
    def __init__(self):
        self.encoder = SharedEncoder(32).eval()
        self.match = LocalSoftMatch(32, radius=2).eval()
        for p in list(self.encoder.parameters()) + list(self.match.parameters()):
            p.requires_grad_(False)


class TestV4GuidanceModes(unittest.TestCase):
    def test_self_zero_shuffled(self):
        torch.manual_seed(0)
        w = _Wrap()
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        donor = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            fr_n, t_n = features_a1_mode(w, y0, r, 'normal')
            fr_s, t_s = features_a1_mode(w, y0, r, 'self')
            fr_z, t_z = features_a1_mode(w, y0, r, 'zero')
            fr_d, t_d = features_a1_mode(w, y0, r, 'shuffled_target', y0_donor=donor)
        self.assertLessEqual(float((fr_n - fr_s).abs().max()), 1e-6)
        self.assertLessEqual(float((fr_n - fr_z).abs().max()), 1e-6)
        self.assertEqual(float(t_z.abs().max()), 0.0)
        self.assertGreater(float((t_n - t_s).abs().max()), 1e-5)
        self.assertGreater(float((t_n - t_d).abs().max()), 1e-5)
