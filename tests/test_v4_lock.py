"""V4 lock keys."""

import unittest

from v4_runtime import FORMAL_LOCK_KEYS_V4


class TestV4Lock(unittest.TestCase):
    def test_required_keys(self):
        need = {
            'repo_commit', 'proposal_sha256', 'split_sha256',
            'mismatch_train_sha256', 'mismatch_dev_sha256',
            'reference_variant', 'direction_A0', 'direction_A1',
            'canvas_A0', 'canvas_A1', 'architecture', 'arms',
            'shared_head_init_sha', 'a0_init_sha', 'a1_init_sha',
            'optimizer', 'lr', 'updates', 'grad_accum', 'seed',
            'pair_schedule_seed', 'official_test_allowed',
        }
        self.assertTrue(need.issubset(set(FORMAL_LOCK_KEYS_V4)))
