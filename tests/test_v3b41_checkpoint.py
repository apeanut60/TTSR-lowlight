"""B4.1 checkpoint / resume / injection / conditioning metadata / Test forbidden."""

import os
import unittest

from model.V3BFeatureBridge import INJECTION_POINT
from v3b2_runtime import SEED
from v3b41_runtime import ARM_A1, require_ckpt_blob_b41


class TestV3B41Checkpoint(unittest.TestCase):
    def _good(self):
        return dict(
            step=20000, arm=ARM_A1, objective='mse_reconstruction',
            seed=SEED, init_sha='abc', proposal_sha='def',
            repo_commit='deadbeef', injection_point=INJECTION_POINT,
            base_conditioned=True, base_feature_ch=80, ref_input_ch=96,
        )

    def test_ok(self):
        require_ckpt_blob_b41(
            self._good(), arm=ARM_A1, step=20000,
            init_sha='abc', proposal_sha='def', repo_commit='deadbeef')

    def test_wrong_injection_commit_cond(self):
        bad = self._good()
        bad['injection_point'] = 'h4_bottleneck'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_b41(
                bad, arm=ARM_A1, step=20000,
                init_sha='abc', proposal_sha='def', repo_commit='deadbeef')
        with self.assertRaises(SystemExit):
            require_ckpt_blob_b41(
                self._good(), arm=ARM_A1, step=20000,
                init_sha='abc', proposal_sha='def', repo_commit='other')
        bad2 = self._good()
        bad2['base_conditioned'] = False
        with self.assertRaises(SystemExit):
            require_ckpt_blob_b41(
                bad2, arm=ARM_A1, step=20000,
                init_sha='abc', proposal_sha='def', repo_commit='deadbeef')

    def test_scripts_guards(self):
        root = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        with open(os.path.join(root, 'train_v3b41.py'), encoding='utf-8') as f:
            train = f.read()
        with open(os.path.join(root, 'eval_v3b41.py'), encoding='utf-8') as f:
            ev = f.read()
        with open(os.path.join(root, 'setup_v3b41.py'), encoding='utf-8') as f:
            setup = f.read()
        self.assertIn('formal B4.1 forbids start_step>0', train)
        self.assertIn('official test forbidden', ev)
        self.assertIn('official_test_allowed=False', setup)
        self.assertIn('HARD STOP', setup)
        self.assertIn('bridge_file_sha', setup)
        self.assertNotIn('RefTextureAdapter', train)
        self.assertNotIn('MHA', train)
