"""V3-A.5D0: T-equivalence + evidence schema invariants."""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0]]

from model.LocalRefine import Refiner                                    # noqa: E402
from model.ReferenceEvidence import LocalSoftMatchWithEvidence          # noqa: E402
from model.V3A5DEvidenceProbe import V3A5DEvidenceProbe                 # noqa: E402
from v3a5d_runtime import (ANALYSIS_CHANNELS, evidence_schema_payload,  # noqa: E402
                           evidence_schema_sha)


CKPT = ('/root/data/experiments/v3a1_lolv2real/'
        'R1_v2stable_naive_s42/checkpoint_03000.pt')


class TestEvidenceSchema(unittest.TestCase):
    def test_schema_locked(self):
        p = evidence_schema_payload()
        self.assertEqual(len(p['analysis_channels']), 11)
        self.assertEqual(p['match_names'][0], 'sim_max')
        sha = evidence_schema_sha()
        self.assertEqual(len(sha), 64)
        self.assertEqual(sha, evidence_schema_sha())


class TestTEquivalence(unittest.TestCase):
    def test_match_only_bit_close(self):
        p = Refiner()
        ev = LocalSoftMatchWithEvidence(32, 4, 0.1)
        ev.load_state_dict(p.match.state_dict(), strict=True)
        f0 = torch.randn(1, 32, 32, 48)
        fr = torch.randn(1, 32, 32, 48)
        with torch.no_grad():
            t0 = p.match(f0, fr)
            t1, e = ev(f0, fr, return_evidence=True)
        self.assertEqual(float((t0 - t1).abs().max()), 0.0)
        for k in ('sim_max', 'pmax', 'entropy', 'margin', 'dx', 'dy', 'disp_var'):
            self.assertIn(k, e)
            self.assertTrue(torch.isfinite(e[k]).all())

    @unittest.skipUnless(torch.cuda.is_available() and os.path.isfile(CKPT),
                         'needs CUDA + R1 ckpt')
    def test_probe_vs_frozen_proposal(self):
        from model.V3A4Verifier import V3A4Refiner
        from v3a4_runtime import load_r1_proposal_strict

        wrap = V3A4Refiner('none').cuda().eval()
        load_r1_proposal_strict(wrap, CKPT, 'cuda')
        for p in wrap.parameters():
            p.requires_grad_(False)
        probe = V3A5DEvidenceProbe(wrap.proposal).cuda().eval()
        y0 = torch.randn(1, 3, 64, 96, device='cuda')
        r = torch.randn(1, 3, 64, 96, device='cuda')
        with torch.no_grad():
            sr, aux, evidence = probe(y0, r, check_equiv=True)
        self.assertTrue(aux['equiv']['ok'])
        self.assertLessEqual(aux['equiv']['T_max_abs'], 1e-6)
        self.assertIn('sim_max', evidence['match'])
        self.assertIn('gate_v2', evidence['proposal']['full'])
        self.assertIn('f0_minus_t', evidence['proposal']['feature'])
        # confidence transform present in analysis list
        names = [c[0] for c in ANALYSIS_CHANNELS]
        self.assertIn('confidence_entropy', names)


if __name__ == '__main__':
    unittest.main()
