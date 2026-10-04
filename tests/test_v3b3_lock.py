"""B3 formal lock keys."""

import unittest

from v3b3_runtime import FORMAL_LOCK_KEYS_B3


class TestV3B3Lock(unittest.TestCase):
    def test_required_keys(self):
        need = {
            'repo_commit', 'proposal_sha256', 'split_sha256',
            'mismatch_train_sha256', 'mismatch_dev_sha256',
            'reference_variant', 'architecture', 'arms',
            'a0_init_sha', 'a1_init_sha', 'common_weight_sha',
            'global_stat_names', 'stat_source_space', 'std_unbiased',
            'global_stats_sha256', 'normalization_method',
            'mlp_architecture', 'broadcast_channels',
            'optimizer', 'lr', 'updates', 'grad_accum', 'seed',
            'pair_schedule_seed', 'official_test_allowed',
        }
        self.assertTrue(need.issubset(set(FORMAL_LOCK_KEYS_B3)))


if __name__ == '__main__':
    unittest.main()
