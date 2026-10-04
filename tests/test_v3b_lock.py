"""Formal lock keys for V3-B.0."""

import unittest

from v3b_runtime import FORMAL_LOCK_KEYS, FORMAL_LOCK_KEYS_B1


class TestV3BLock(unittest.TestCase):
    def test_required_keys(self):
        need = {
            'repo_commit', 'proposal_sha256', 'split_sha256',
            'mismatch_train_sha256', 'mismatch_dev_sha256',
            'reference_variant', 'architecture', 'arm', 'init_sha',
            'optimizer', 'lr', 'updates', 'grad_accum', 'seed',
            'official_test_allowed',
        }
        self.assertTrue(need.issubset(set(FORMAL_LOCK_KEYS)))

    def test_b1_keys(self):
        self.assertTrue(set(FORMAL_LOCK_KEYS).issubset(set(FORMAL_LOCK_KEYS_B1)))
        self.assertIn('b0_init_sha', FORMAL_LOCK_KEYS_B1)
        self.assertIn('b1_arms', FORMAL_LOCK_KEYS_B1)


if __name__ == '__main__':
    unittest.main()
