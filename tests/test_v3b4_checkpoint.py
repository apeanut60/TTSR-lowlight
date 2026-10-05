"""B4 checkpoint / resume / injection-point / Test forbidden."""

import os
import unittest

from model.V3BFeatureBridge import INJECTION_POINT
from v3b2_runtime import SEED
from v3b4_runtime import require_ckpt_blob_b4


class TestV3B4Checkpoint(unittest.TestCase):
    def _good(self):
        return dict(
            step=20000, arm='A1_h2_feature', objective='mse_reconstruction',
            seed=SEED, init_sha='abc', proposal_sha='def',
            repo_commit='deadbeef', injection_point=INJECTION_POINT,
        )

    def test_ok(self):
        require_ckpt_blob_b4(
            self._good(), arm='A1_h2_feature', step=20000,
            init_sha='abc', proposal_sha='def', repo_commit='deadbeef')

    def test_wrong_injection_and_commit(self):
        bad = self._good()
        bad['injection_point'] = 'h4_bottleneck'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_b4(
                bad, arm='A1_h2_feature', step=20000,
                init_sha='abc', proposal_sha='def', repo_commit='deadbeef')
        with self.assertRaises(SystemExit):
            require_ckpt_blob_b4(
                self._good(), arm='A1_h2_feature', step=20000,
                init_sha='abc', proposal_sha='def', repo_commit='other')

    def test_scripts_guards(self):
        root = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        train = open(os.path.join(root, 'train_v3b4.py'), encoding='utf-8').read()
        ev = open(os.path.join(root, 'eval_v3b4.py'), encoding='utf-8').read()
        setup = open(os.path.join(root, 'setup_v3b4.py'), encoding='utf-8').read()
        self.assertIn('formal B4 forbids start_step>0', train)
        self.assertIn('official test forbidden', ev)
        self.assertIn("official_test_allowed=False", setup)
        self.assertIn('HARD STOP', setup)
        self.assertNotIn('RefTextureAdapter', train)
