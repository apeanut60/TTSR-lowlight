"""V3-A.5D2 unit tests: zero residual context, step0 A0≡A1, verdict labels."""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.V3A5D2Verifier import V3A5D2Verifier, build_d2_shared_init    # noqa: E402
from v3a5d2_runtime import verdict_d2                                   # noqa: E402


class TestD2Init(unittest.TestCase):
    def test_shared_init_step0_identical(self):
        init = build_d2_shared_init(seed=42)
        a0 = V3A5D2Verifier('A0_control')
        a1 = V3A5D2Verifier('A1_multiscale')
        a0.load_state_dict(init['A0_control'], strict=True)
        a1.load_state_dict(init['A1_multiscale'], strict=True)
        self.assertEqual(a1.context_residual_max_abs(), 0.0)

        x = torch.randn(1, 3, 64, 96)
        y0 = torch.randn(1, 3, 64, 96)
        r = torch.randn(1, 3, 64, 96)
        with torch.no_grad():
            q0 = a0(x, y0, r, geom=None)
            q1 = a1(x, y0, r, geom=None)
        self.assertLessEqual(float((q0 - q1).abs().max()), 1e-5)
        self.assertAlmostEqual(float(q0.mean()), 0.5, places=5)

    def test_context_changes_after_noise(self):
        a1 = V3A5D2Verifier('A1_multiscale')
        with torch.no_grad():
            a1.context.fuse_h4[-1].weight.normal_(0, 0.01)
        self.assertGreater(a1.context_residual_max_abs(), 0.0)


class TestVerdict(unittest.TestCase):
    def test_success_maps_to_d2(self):
        a0 = dict(masked_MAE=0.25, masked_corr=0.57, decision_accuracy=0.83)
        a1 = dict(masked_MAE=0.16, masked_corr=0.73, decision_accuracy=0.91)
        v = verdict_d2(a0, a1)
        self.assertEqual(v['label'], 'D2_success')
        self.assertEqual(v['next_step'], 'D2-full')

    def test_null_close_rf(self):
        a0 = dict(masked_MAE=0.25, masked_corr=0.57, decision_accuracy=0.83)
        a1 = dict(masked_MAE=0.24, masked_corr=0.60, decision_accuracy=0.84)
        v = verdict_d2(a0, a1)
        self.assertTrue(v['close_rf_route'])
        self.assertEqual(v['label'], 'D2_null_close_rf')


if __name__ == '__main__':
    unittest.main()
