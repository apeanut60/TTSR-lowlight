"""SmoothL1 is the only train loss; no VGG / Nano in train script."""

import ast
import os
import unittest

import torch
import torch.nn as nn


class TestLoss(unittest.TestCase):
    def test_smooth_l1(self):
        crit = nn.SmoothL1Loss()
        a = torch.zeros(1, 3, 8, 8)
        b = torch.ones(1, 3, 8, 8)
        loss = crit(a, b)
        self.assertGreater(float(loss), 0)

    def test_train_script_guards(self):
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
        path = os.path.join(root, 'train_llformer_lolv2real.py')
        self.assertTrue(os.path.isfile(path), 'train script missing: %s' % path)
        src = open(path, encoding='utf-8').read()
        self.assertIn('SmoothL1Loss', src)
        for ban in ('VGG19', 'perceptual', 'lpips.LPIPS', 'Nano', 'GAN',
                    'load_frozen_n0', 'V5Model', 'V3B0ResidualFusion'):
            self.assertNotIn(ban, src)
        # parse ok
        ast.parse(src)


if __name__ == '__main__':
    unittest.main()
