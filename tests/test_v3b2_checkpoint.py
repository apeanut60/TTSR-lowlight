"""B2 checkpoint integrity + formal resume refuse."""

import unittest

from v3b2_runtime import SEED, require_ckpt_blob


class TestV3B2Checkpoint(unittest.TestCase):
    def _good(self):
        return dict(
            step=20000, arm='A1_evidence', objective='mse_reconstruction',
            seed=SEED, init_sha='abc', proposal_sha='def',
            repo_commit='deadbeef',
        )

    def test_ok(self):
        require_ckpt_blob(
            self._good(), arm='A1_evidence', step=20000,
            init_sha='abc', proposal_sha='def', repo_commit='deadbeef')

    def test_wrong_arm(self):
        with self.assertRaises(SystemExit):
            require_ckpt_blob(
                self._good(), arm='A0_b0_replay', step=20000,
                init_sha='abc', proposal_sha='def')

    def test_wrong_step(self):
        with self.assertRaises(SystemExit):
            require_ckpt_blob(
                self._good(), arm='A1_evidence', step=10000,
                init_sha='abc', proposal_sha='def')

    def test_wrong_init_sha(self):
        with self.assertRaises(SystemExit):
            require_ckpt_blob(
                self._good(), arm='A1_evidence', step=20000,
                init_sha='zzz', proposal_sha='def')

    def test_formal_resume_guard_in_train(self):
        # source-level contract: train_v3b2 refuses start_step>0 when formal
        import ast
        import os
        path = os.path.join(
            os.path.dirname(__file__), '..', 'scripts', 'train_v3b2.py')
        src = open(path, encoding='utf-8').read()
        self.assertIn("formal B2 forbids start_step>0", src)
        self.assertIn('start_step > 0', src)


if __name__ == '__main__':
    unittest.main()
