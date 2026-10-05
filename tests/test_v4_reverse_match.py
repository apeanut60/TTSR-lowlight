"""V4 reverse-match actually swaps query/source."""

import unittest

import torch
from model.LocalRefine import LocalSoftMatch, SharedEncoder
from model.V4RefCanvas import encode_m11, features_a0, features_a1, match_qv


class _Wrap(object):
    def __init__(self):
        self.encoder = SharedEncoder(32).eval()
        self.match = LocalSoftMatch(32, radius=2).eval()
        for p in list(self.encoder.parameters()) + list(self.match.parameters()):
            p.requires_grad_(False)


class TestV4ReverseMatch(unittest.TestCase):
    def test_shape(self):
        m = LocalSoftMatch(8, radius=1)
        q = torch.randn(2, 8, 6, 8)
        s = torch.randn(2, 8, 6, 8)
        t = match_qv(m, q, s)
        self.assertEqual(tuple(t.shape), (2, 8, 6, 8))

    def test_direction_not_symmetric(self):
        torch.manual_seed(0)
        m = LocalSoftMatch(8, radius=1)
        a = torch.randn(1, 8, 6, 8)
        b = torch.randn(1, 8, 6, 8)
        t_ab = match_qv(m, a, b)
        t_ba = match_qv(m, b, a)
        self.assertGreater(float((t_ab - t_ba).abs().max()), 1e-5)

    def test_changing_f0_changes_t_low(self):
        torch.manual_seed(1)
        w = _Wrap()
        y0a = torch.randn(1, 3, 32, 32)
        y0b = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            fr_a, t_a = features_a1(w, y0a, r)
            fr_b, t_b = features_a1(w, y0b, r)
        self.assertLessEqual(float((fr_a - fr_b).abs().max()), 1e-6)
        self.assertGreater(float((t_a - t_b).abs().max()), 1e-5)

    def test_changing_r_changes_query(self):
        torch.manual_seed(2)
        w = _Wrap()
        y0 = torch.randn(1, 3, 32, 32)
        r1 = torch.randn(1, 3, 32, 32)
        r2 = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            fr1, _ = features_a1(w, y0, r1)
            fr2, _ = features_a1(w, y0, r2)
        self.assertGreater(float((fr1 - fr2).abs().max()), 1e-5)

    def test_a0_query_is_f0(self):
        torch.manual_seed(3)
        w = _Wrap()
        y0 = torch.randn(1, 3, 32, 32)
        r = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            f0, t = features_a0(w, y0, r)
            f0e = encode_m11(w.encoder, y0)
            t2 = w.match(f0e, encode_m11(w.encoder, r))
        self.assertLessEqual(float((f0 - f0e).abs().max()), 1e-6)
        self.assertLessEqual(float((t - t2).abs().max()), 1e-6)
