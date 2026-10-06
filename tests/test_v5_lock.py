"""V5.0 lock keys."""

import unittest

from v5_runtime import FORMAL_LOCK_KEYS_V5


class TestV5Lock(unittest.TestCase):
    def test_required_keys(self):
        need = {
            'stage', 'repo_commit', 'base_ckpt_sha256', 'split_sha256',
            'mismatch_train_sha256', 'mismatch_dev_sha256', 'reference_variant',
            'architecture', 'match_encoder', 'texture_encoder',
            'search_range', 'correlation_type', 'warp_type', 'dcn_kernel',
            'refine_scales', 'zero_init', 'loss', 'optimizer', 'lr',
            'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
            'official_test_allowed', 'frozen_base', 'trainable_modules',
            'v5_init_sha', 'injection_point',
        }
        self.assertTrue(need.issubset(set(FORMAL_LOCK_KEYS_V5)))
        self.assertIn('official_test_allowed', FORMAL_LOCK_KEYS_V5)
