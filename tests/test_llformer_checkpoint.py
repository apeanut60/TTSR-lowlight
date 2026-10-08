"""Official LOL checkpoint must fully load into LLFormerBridge."""

import os
import unittest

import torch

from model.LLFormerBridge import (OFFICIAL_LOL_CKPT, build_official_llformer,
                                  load_into_llformer, load_llformer_bridge)


class TestLLFormerCheckpoint(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(os.path.isfile(OFFICIAL_LOL_CKPT), OFFICIAL_LOL_CKPT)

    def test_full_strict_load(self):
        net = build_official_llformer()
        info = load_into_llformer(net, OFFICIAL_LOL_CKPT)
        self.assertEqual(info['missing'], 0)
        self.assertEqual(info['unexpected'], 0)
        self.assertEqual(info['shape_mismatch'], 0)
        self.assertEqual(info['loaded'], 1485)

    def test_bridge_load(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net, info = load_llformer_bridge(OFFICIAL_LOL_CKPT, device=device,
                                         train=False)
        self.assertEqual(info['loaded'], 1485)
        n = sum(p.numel() for p in net.parameters())
        self.assertGreater(n, 0)


if __name__ == '__main__':
    unittest.main()
