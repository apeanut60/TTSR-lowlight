"""V6 checkpoint / lock field guards."""

import ast
import os
import unittest

from model.V6LLFormerRefBridge import INJECTION_POINT
from v6_runtime import ARM_A1, FORMAL_LOCK_KEYS_V6, SEED, require_ckpt_blob_v6


class TestV6Lock(unittest.TestCase):
    def test_ckpt_blob(self):
        good = dict(
            step=30000, arm=ARM_A1, objective='mse_reconstruction',
            seed=SEED, init_sha='abc', repo_commit='deadbeef',
            injection_point=INJECTION_POINT, frozen_base=True, loss='MSE',
        )
        require_ckpt_blob_v6(
            good, arm=ARM_A1, step=30000, init_sha='abc',
            repo_commit='deadbeef')
        bad = dict(good)
        bad['repo_commit'] = 'ffff'
        with self.assertRaises(SystemExit):
            require_ckpt_blob_v6(
                bad, arm=ARM_A1, step=30000, init_sha='abc',
                repo_commit='deadbeef')

    def test_scripts_guards(self):
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
        with open(os.path.join(root, 'train_v6.py'), encoding='utf-8') as f:
            train = f.read()
        self.assertNotIn('VGG19', train)
        self.assertNotIn('load_frozen_n0', train)
        self.assertNotIn('V3B0ResidualFusion', train)
        self.assertNotIn('perceptual', train)
        self.assertIn('b0_loss', train)
        ast.parse(train)
        self.assertIn('v6_init_sha', FORMAL_LOCK_KEYS_V6)


if __name__ == '__main__':
    unittest.main()
