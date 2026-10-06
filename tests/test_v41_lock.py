"""V4.1a lock keys."""

import unittest

from v41_runtime import FORMAL_LOCK_KEYS_V41


class TestV41Lock(unittest.TestCase):
    def test_required_keys(self):
        need = {
            'repo_commit', 'proposal_sha256', 'split_sha256',
            'mismatch_train_sha256', 'mismatch_dev_sha256',
            'reference_variant', 'b0_head_ckpt_sha256', 'b0_head_state_sha',
            'ground_direction', 'forward_direction', 'architecture',
            'grounder_zero_output', 'frozen_b0', 'trainable_modules',
            'grounder_init_sha', 'optimizer', 'lr', 'updates', 'grad_accum',
            'seed', 'pair_schedule_seed', 'official_test_allowed',
        }
        self.assertTrue(need.issubset(set(FORMAL_LOCK_KEYS_V41)))
