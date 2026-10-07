"""V5 checkpoint hygiene and script guards."""

import os
import unittest

from v3b_runtime import SEED
from v5_runtime import ARM_A1, require_ckpt_blob_v5
from model.V5RetinexBridge import INJECTION_POINT


class TestV5Checkpoint(unittest.TestCase):
    def _good(self):
        return dict(
            step=30000, arm=ARM_A1, objective='mse_reconstruction',
            seed=SEED, init_sha='abc', repo_commit='deadbeef',
            injection_point=INJECTION_POINT, frozen_base=True, loss='MSE',
        )

    def test_ok(self):
        require_ckpt_blob_v5(
            self._good(), arm=ARM_A1, step=30000,
            init_sha='abc', repo_commit='deadbeef')

    def test_wrong_commit_and_injection(self):
        bad = self._good()
        bad['repo_commit'] = 'ffff'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v5(
                bad, arm=ARM_A1, step=30000,
                init_sha='abc', repo_commit='deadbeef')
        bad = self._good()
        bad['injection_point'] = 'h2_only'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v5(
                bad, arm=ARM_A1, step=30000,
                init_sha='abc', repo_commit='deadbeef')

    def test_scripts_guards(self):
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
        train = open(os.path.join(root, 'train_v5.py'), encoding='utf-8').read()
        setup = open(os.path.join(root, 'setup_v5.py'), encoding='utf-8').read()
        ev = open(os.path.join(root, 'eval_v5.py'), encoding='utf-8').read()
        self.assertIn('formal V5.0 forbids start_step>0', train)
        self.assertIn('official test forbidden', ev)
        self.assertIn('official_test_allowed=False', setup)
        self.assertIn('HARD STOP', setup)
        self.assertIn('Ref branch grad is zero', train)
        self.assertIn('frozen Base received grad', train)
        self.assertNotIn('load_frozen_n0', train)
        self.assertNotIn('PerPixelMultiRefAttention', train)
