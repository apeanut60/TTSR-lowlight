"""V4 checkpoint direction/canvas/commit / Test forbidden / no resume."""

import os
import unittest

from model.V4RefCanvas import CANVAS_A1, DIR_A1
from v3b2_runtime import SEED
from v4_runtime import ARM_A1, require_ckpt_blob_v4


class TestV4Checkpoint(unittest.TestCase):
    def _good(self):
        return dict(
            step=20000, arm=ARM_A1, objective='mse_reconstruction',
            seed=SEED, init_sha='abc', proposal_sha='def',
            repo_commit='deadbeef', direction=DIR_A1, canvas=CANVAS_A1,
            shared_head_init_sha='abc',
        )

    def test_ok(self):
        require_ckpt_blob_v4(
            self._good(), arm=ARM_A1, step=20000, init_sha='abc',
            proposal_sha='def', repo_commit='deadbeef',
            direction=DIR_A1, canvas=CANVAS_A1, shared_init_sha='abc')

    def test_wrong_direction_canvas_commit(self):
        bad = self._good()
        bad['direction'] = 'F0_query_FR_source'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v4(
                bad, arm=ARM_A1, step=20000, init_sha='abc',
                proposal_sha='def', repo_commit='deadbeef',
                direction=DIR_A1, canvas=CANVAS_A1, shared_init_sha='abc')
        bad2 = self._good()
        bad2['canvas'] = 'Y0'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v4(
                bad2, arm=ARM_A1, step=20000, init_sha='abc',
                proposal_sha='def', repo_commit='deadbeef',
                direction=DIR_A1, canvas=CANVAS_A1, shared_init_sha='abc')
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v4(
                self._good(), arm=ARM_A1, step=20000, init_sha='abc',
                proposal_sha='def', repo_commit='other',
                direction=DIR_A1, canvas=CANVAS_A1, shared_init_sha='abc')

    def test_scripts_guards(self):
        root = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        with open(os.path.join(root, 'train_v4.py'), encoding='utf-8') as f:
            train = f.read()
        with open(os.path.join(root, 'eval_v4.py'), encoding='utf-8') as f:
            ev = f.read()
        with open(os.path.join(root, 'setup_v4.py'), encoding='utf-8') as f:
            setup = f.read()
        self.assertIn('formal V4 forbids start_step>0', train)
        self.assertIn('official test forbidden', ev)
        self.assertIn('official_test_allowed=False', setup)
        self.assertIn('shared_head_init_sha', setup)
        self.assertNotIn('RefTextureAdapter', train)
        self.assertNotIn('MHA', train)
