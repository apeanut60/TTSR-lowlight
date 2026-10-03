"""T5: image-cluster bootstrap resamples whole images, not independent blocks."""

import unittest

import numpy as np

from v3a72_runtime import cluster_resample_indices


class TestV3A72Bootstrap(unittest.TestCase):
    def test_repeat_image_repeats_all_its_blocks(self):
        image_ids = np.array(['a', 'a', 'a', 'b'])
        idx = cluster_resample_indices(image_ids, ['a', 'a'])
        self.assertEqual(idx.size, 6)
        self.assertTrue(np.all(image_ids[idx] == 'a'))

    def test_chosen_b_only_one_block(self):
        image_ids = np.array(['a', 'a', 'a', 'b'])
        idx = cluster_resample_indices(image_ids, ['b'])
        self.assertEqual(list(idx), [3])


if __name__ == '__main__':
    unittest.main()
