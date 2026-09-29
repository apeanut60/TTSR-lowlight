#!/usr/bin/env python
"""V3-A.5C acceptance tests (CPU-friendly unit checks + optional CUDA smoke)."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from v3a5c_runtime import (choose_tiny_ids, make_pair_schedule,  # noqa: E402
                           polarized_decision_accuracy, slice_mismatch_map,
                           verdict_c0, verdict_c1)


class TinySubsetTests(unittest.TestCase):
    def test_ids_deterministic(self):
        ids = ['low%05d.png' % i for i in range(575)]
        a = choose_tiny_ids(ids, n=16, seed=20260928)
        b = choose_tiny_ids(ids, n=16, seed=20260928)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 16)

    def test_mismatch_donors_in_train(self):
        full = {'a.png': 'b.png', 'b.png': 'c.png', 'c.png': 'a.png',
                'd.png': 'a.png'}
        sub = slice_mismatch_map(full, ['a.png', 'd.png'])
        self.assertEqual(sub['a.png'], 'b.png')
        with self.assertRaises(KeyError):
            slice_mismatch_map(full, ['missing.png'])


class MetricsTests(unittest.TestCase):
    def test_decision_accuracy_perfect(self):
        q_star = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5])  # last mid ignored
        q_v = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.5])
        d = polarized_decision_accuracy(q_v, q_star)
        self.assertEqual(d['n_polar'], 4)
        self.assertAlmostEqual(d['decision_accuracy'], 1.0)

    def test_pair_schedule_equal_freq(self):
        sched = make_pair_schedule(n_updates=12, grad_accum=4, n_pairs=48,
                                   seed=42)
        self.assertEqual(len(sched), 48)
        # first epoch permutation covers all
        self.assertEqual(sorted(sched.tolist()), list(range(48)))


class VerdictTests(unittest.TestCase):
    def test_c0_success(self):
        overall = dict(masked_MAE=0.08, masked_corr=0.85,
                       decision_accuracy=0.92, std_ratio=0.5)
        by = {s: dict(masked_corr=0.7) for s in
              ('correct', 'true_dark_g0.5', 'mismatch')}
        v = verdict_c0(overall, by)
        self.assertEqual(v['label'], 'C0_success')
        self.assertEqual(v['action'], 'stop_no_c1_predictability')

    def test_c0_partial_goes_to_c1(self):
        overall = dict(masked_MAE=0.15, masked_corr=0.60,
                       decision_accuracy=0.75, std_ratio=0.5)
        by = {s: dict(masked_corr=0.7) for s in
              ('correct', 'true_dark_g0.5', 'mismatch')}
        v = verdict_c0(overall, by)
        self.assertEqual(v['label'], 'C0_partial_fail')
        self.assertEqual(v['action'], 'run_c1')

    def test_c1_requires_all_three(self):
        c0 = dict(masked_MAE=0.30, masked_corr=0.20, decision_accuracy=0.5)
        c1 = dict(masked_MAE=0.15, masked_corr=0.55, decision_accuracy=0.95)
        # mae improve 0.15 ok, corr +0.35 ok, acc ok -> success
        v = verdict_c1(c0, c1)
        self.assertEqual(v['label'], 'C1_success')
        # drop corr improvement
        c1b = dict(masked_MAE=0.15, masked_corr=0.40, decision_accuracy=0.95)
        v2 = verdict_c1(c0, c1b)
        self.assertEqual(v2['label'], 'C1_partial')


class OptionalCudaSmoke(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gradient_nonzero_on_random(self):
        from model.V3A5Verifier import V3A5Verifier
        m = V3A5Verifier('g64').cuda()
        x = torch.randn(1, 3, 400, 600, device='cuda')
        y0 = torch.randn(1, 3, 400, 600, device='cuda')
        r = torch.randn(1, 3, 400, 600, device='cuda')
        from v3a5_runtime import prepare_geometry, target_geometry
        geom = prepare_geometry(target_geometry(400, 600, 'g64'), 'cuda')
        q = m(x, y0, r, geom=geom)
        q.mean().backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))


if __name__ == '__main__':
    unittest.main()
