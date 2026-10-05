"""B4 lock keys and official test flag."""

import unittest

from v3b4_runtime import FORMAL_LOCK_KEYS_B4


class TestV3B4Lock(unittest.TestCase):
    def test_required_keys(self):
        need = {
            'repo_commit', 'proposal_sha256', 'split_sha256',
            'mismatch_train_sha256', 'mismatch_dev_sha256',
            'reference_variant', 'injection_point', 'base_feature_ch',
            'reference_feature_ch', 'architecture_a0', 'architecture_a1',
            'arms', 'a0_init_sha', 'a1_init_sha', 'adapter_n_params',
            'optimizer', 'lr', 'updates', 'grad_accum', 'seed',
            'pair_schedule_seed', 'official_test_allowed',
        }
        self.assertTrue(need.issubset(set(FORMAL_LOCK_KEYS_B4)))
