"""Train tensors are [0,1]; paired crop/aug exact."""

import unittest

import torch

from llformer_runtime import LOLv2PairDataset, build_train625_dev64, paired_augment


class TestDataRange(unittest.TestCase):
    def test_pair_range_and_aug(self):
        train, _ = build_train625_dev64()
        ds = LOLv2PairDataset(train[:4], train=True, patch=128)
        for i in range(len(ds)):
            t = ds[i]
            self.assertEqual(tuple(t['low'].shape), (3, 128, 128))
            self.assertGreaterEqual(float(t['low'].min()), 0.0)
            self.assertLessEqual(float(t['low'].max()), 1.0)
            self.assertGreaterEqual(float(t['high'].min()), 0.0)
            self.assertLessEqual(float(t['high'].max()), 1.0)

    def test_paired_aug_same_geom(self):
        torch.manual_seed(0)
        a = torch.rand(3, 200, 200)
        b = a.clone()
        # force deterministic by fixing RNG around call
        import random
        random.seed(0)
        torch.manual_seed(0)
        la, ha = paired_augment(a, b, 128)
        random.seed(0)
        torch.manual_seed(0)
        lb, hb = paired_augment(a, b, 128)
        self.assertTrue(torch.equal(la, lb))
        self.assertTrue(torch.equal(ha, hb))
        self.assertTrue(torch.equal(la, ha))  # same source content


if __name__ == '__main__':
    unittest.main()
