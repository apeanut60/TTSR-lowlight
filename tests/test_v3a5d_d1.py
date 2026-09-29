"""V3-A.5D1 unit tests: zero-init evidence, step0 A0≡A1, verdict thresholds."""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.EvidenceFusion import evidence_weight_max_abs                 # noqa: E402
from model.V3A5DVerifier import (EVIDENCE_NAMES, V3A5DVerifier,         # noqa: E402
                                 build_d1_shared_init)
from v3a5d1_runtime import verdict_d1                                   # noqa: E402


class TestD1Init(unittest.TestCase):
    def test_evidence_names_locked(self):
        self.assertEqual(EVIDENCE_NAMES, ('sim_max', 'f0_minus_t', 'gate_v2'))

    def test_shared_init_zero_evidence_and_step0(self):
        init = build_d1_shared_init(seed=42, ev_mean=[0.5, 0.1, 0.5],
                                    ev_std=[0.2, 0.05, 1.0])
        a0 = V3A5DVerifier('A0_control')
        a1 = V3A5DVerifier('A1_evidence')
        a0.load_state_dict(init['A0_control'], strict=True)
        a1.load_state_dict(init['A1_evidence'], strict=True)
        a0.set_evidence_norm([0.5, 0.1, 0.5], [0.2, 0.05, 1.0], [1, 1, 0])
        a1.set_evidence_norm([0.5, 0.1, 0.5], [0.2, 0.05, 1.0], [1, 1, 0])
        self.assertEqual(a1.evidence_weight_max_abs(), 0.0)
        self.assertEqual(evidence_weight_max_abs(a1.ev_head), 0.0)

        x = torch.randn(1, 3, 64, 96)
        y0 = torch.randn(1, 3, 64, 96)
        r = torch.randn(1, 3, 64, 96)
        # native H/4 = 16x24; use geom=None for unit test (no pool)
        with torch.no_grad():
            q0 = a0(x, y0, r, geom=None)
            ev = torch.randn(1, 3, *q0.shape[-2:])
            q1 = a1(x, y0, r, geom=None, evidence_g64=ev)
        self.assertLessEqual(float((q0 - q1).abs().max()), 1e-5)
        self.assertAlmostEqual(float(q0.mean()), 0.5, places=5)


class TestVerdict(unittest.TestCase):
    def test_success(self):
        a0 = dict(masked_MAE=0.25, masked_corr=0.57, decision_accuracy=0.83)
        a1 = dict(masked_MAE=0.16, masked_corr=0.73, decision_accuracy=0.91)
        v = verdict_d1(a0, a1)
        self.assertTrue(v['success'])
        self.assertEqual(v['next_step'], 'D1-full')

    def test_close_route(self):
        a0 = dict(masked_MAE=0.25, masked_corr=0.57, decision_accuracy=0.83)
        a1 = dict(masked_MAE=0.24, masked_corr=0.60, decision_accuracy=0.84)
        v = verdict_d1(a0, a1)
        self.assertTrue(v['close_evidence_route'])
        self.assertEqual(v['next_step'], 'D2')

    def test_insufficient_not_close(self):
        # medium bump but acc fails success gate
        a0 = dict(masked_MAE=0.25, masked_corr=0.57, decision_accuracy=0.83)
        a1 = dict(masked_MAE=0.16, masked_corr=0.73, decision_accuracy=0.85)
        v = verdict_d1(a0, a1)
        self.assertFalse(v['success'])
        self.assertFalse(v['close_evidence_route'])
        self.assertEqual(v['label'], 'D1_insufficient')


if __name__ == '__main__':
    unittest.main()
