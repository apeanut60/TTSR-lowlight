"""train625/dev64 hard lock assertions."""

import unittest

from llformer_runtime import build_train625_dev64, list_train_lows


class TestSplitLock(unittest.TestCase):
    def test_counts_and_partition(self):
        train, dev = build_train625_dev64()
        self.assertEqual(len(train), 625)
        self.assertEqual(len(dev), 64)
        self.assertEqual(len(set(train) & set(dev)), 0)
        all_lows = set(list_train_lows())
        self.assertEqual(set(train) | set(dev), all_lows)
        self.assertEqual(len(all_lows), 689)


if __name__ == '__main__':
    unittest.main()
