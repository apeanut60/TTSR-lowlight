"""V4.1a checkpoint B0 SHA / repo commit / Test forbidden / no resume."""

import os
import unittest

from model.V4RefGrounder import FORWARD_DIR, GROUND_DIR
from v3b2_runtime import SEED
from v41_runtime import ARM_A1, require_ckpt_blob_v41


class TestV41Checkpoint(unittest.TestCase):
    def _good(self):
        return dict(
            step=20000, arm=ARM_A1, objective='mse_reconstruction',
            seed=SEED, init_sha='abc', proposal_sha='def',
            repo_commit='deadbeef', b0_head_state_sha='b0sha',
            ground_direction=GROUND_DIR, forward_direction=FORWARD_DIR,
            frozen_b0=True,
        )

    def test_ok(self):
        require_ckpt_blob_v41(
            self._good(), arm=ARM_A1, step=20000, init_sha='abc',
            proposal_sha='def', repo_commit='deadbeef', b0_head_state_sha='b0sha')

    def test_wrong_b0_and_commit(self):
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v41(
                self._good(), arm=ARM_A1, step=20000, init_sha='abc',
                proposal_sha='def', repo_commit='deadbeef', b0_head_state_sha='other')
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v41(
                self._good(), arm=ARM_A1, step=20000, init_sha='abc',
                proposal_sha='def', repo_commit='other', b0_head_state_sha='b0sha')
        bad = self._good()
        bad['ground_direction'] = 'F0_query_FR_source'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v41(
                bad, arm=ARM_A1, step=20000, init_sha='abc',
                proposal_sha='def', repo_commit='deadbeef', b0_head_state_sha='b0sha')

    def test_scripts_guards(self):
        root = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        with open(os.path.join(root, 'train_v41.py'), encoding='utf-8') as f:
            train = f.read()
        with open(os.path.join(root, 'eval_v41.py'), encoding='utf-8') as f:
            ev = f.read()
        with open(os.path.join(root, 'setup_v41.py'), encoding='utf-8') as f:
            setup = f.read()
        self.assertIn('formal V4.1a forbids start_step>0', train)
        self.assertIn('official test forbidden', ev)
        self.assertIn('official_test_allowed=False', setup)
        self.assertIn('HARD STOP', setup)
        self.assertIn('grounder grad is zero', train)
        self.assertIn('frozen %s received grad', train)
        self.assertNotIn('RefTextureAdapter', train)
        self.assertNotIn('MHA', train)
