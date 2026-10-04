"""B2 evidence extractor / normalize / shape."""

import inspect
import unittest

import torch

from model.LocalRefine import Refiner
from model.V3BEvidence import (EVIDENCE_NAMES, attach_match_evidence,
                               extract_match_maps, normalize_evidence)
from v3b2_runtime import assert_no_gt_in_forward


# re-export check: ensure extract signature has no GT
def _check_extract_sig():
    assert_no_gt_in_forward(extract_match_maps)


class TestV3B2Evidence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.core = Refiner(ch=32, radius=4, tau=0.1, chunk=64).eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.match_ev = attach_match_evidence(self.core)
        self.y0 = torch.randn(1, 3, 40, 60)
        self.r = torch.randn(1, 3, 40, 60)

    def test_no_gt_in_extract(self):
        _check_extract_sig()
        names = set(inspect.signature(extract_match_maps).parameters)
        for bad in ('H', 'q_star', 'U', 'AO', 'D', 'g_v2'):
            self.assertNotIn(bad, names)

    def test_spatial_shape_h2(self):
        f0, t, e = extract_match_maps(
            self.core.encoder, self.match_ev, self.y0, self.r)
        self.assertEqual(tuple(f0.shape), (1, 32, 20, 30))
        self.assertEqual(tuple(t.shape), tuple(f0.shape))
        self.assertEqual(tuple(e.shape), (1, 4, 20, 30))
        self.assertEqual(e.shape[1], len(EVIDENCE_NAMES))

    def test_changing_r_changes_evidence(self):
        r2 = torch.randn(1, 3, 40, 60)
        _, _, e1 = extract_match_maps(
            self.core.encoder, self.match_ev, self.y0, self.r)
        _, _, e2 = extract_match_maps(
            self.core.encoder, self.match_ev, self.y0, r2)
        self.assertGreater(float((e1 - e2).abs().max()), 1e-6)

    def test_normalize_train_stats(self):
        stats = {
            name: dict(mean=0.0, std=1.0) for name in EVIDENCE_NAMES
        }
        e = torch.ones(1, 4, 4, 4) * 10
        out = normalize_evidence(e, stats)
        self.assertLessEqual(float(out.max()), 5.0)
        self.assertGreaterEqual(float(out.min()), -5.0)
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 5.0, places=5)


if __name__ == '__main__':
    unittest.main()
