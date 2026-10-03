"""Formal lock key presence for V3-A.7.1."""

import unittest

from v3a71_runtime import FORMAL_LOCK_KEYS


class TestV3A71Lock(unittest.TestCase):
    def test_required_keys(self):
        need = {
            'repo_commit', 'proposal_sha256', 'cache_metadata_sha256',
            'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
            'energy_stats_sha256', 'reference_variant', 'geometry',
            'architecture', 'bottleneck', 'init_sha', 'official_test_allowed',
        }
        self.assertTrue(need.issubset(set(FORMAL_LOCK_KEYS)))


if __name__ == '__main__':
    unittest.main()
