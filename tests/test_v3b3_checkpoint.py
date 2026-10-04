"""B3 checkpoint integrity, resume refuse, official test forbidden."""

import os
import unittest

from v3b2_runtime import SEED, require_ckpt_blob


class TestV3B3Checkpoint(unittest.TestCase):
    def _good(self):
        return dict(
            step=20000, arm='A1_global_stats', objective='mse_reconstruction',
            seed=SEED, init_sha='abc', proposal_sha='def',
            repo_commit='deadbeef',
        )

    def test_wrong_arm_step_sha(self):
        with self.assertRaises(SystemExit):
            require_ckpt_blob(
                self._good(), arm='A0_b0_replay', step=20000,
                init_sha='abc', proposal_sha='def')
        with self.assertRaises(SystemExit):
            require_ckpt_blob(
                self._good(), arm='A1_global_stats', step=10000,
                init_sha='abc', proposal_sha='def')
        with self.assertRaises(SystemExit):
            require_ckpt_blob(
                self._good(), arm='A1_global_stats', step=20000,
                init_sha='zzz', proposal_sha='def')

    def test_formal_resume_and_test_forbidden(self):
        root = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        train = open(os.path.join(root, 'train_v3b3.py'), encoding='utf-8').read()
        evals = open(os.path.join(root, 'eval_v3b3.py'), encoding='utf-8').read()
        setup = open(os.path.join(root, 'setup_v3b3.py'), encoding='utf-8').read()
        self.assertIn('formal B3 forbids start_step>0', train)
        self.assertIn("live_head != lock['repo_commit']", train)
        self.assertIn('official test forbidden', evals)
        self.assertIn('allow_eval_code_drift', evals)
        self.assertIn("official_test_allowed=False", setup)


if __name__ == '__main__':
    unittest.main()
