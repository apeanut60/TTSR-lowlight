"""V3-A.5D0: exact G64 pooling coverage + block-mean ground truth."""

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from v3a5_runtime import prepare_geometry, target_geometry                 # noqa: E402
from v3a5d_runtime import (pool_geometry_coverage,                        # noqa: E402
                           pool_spatial_map_to_geom,
                           pool_spatial_map_to_geom_python)


H, W = 400, 600


class TestGeometryPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geom = prepare_geometry(target_geometry(H, W, 'g64'), 'cpu')

    def test_coverage_full_and_half(self):
        for s in (1, 2):
            cov = pool_geometry_coverage(s, self.geom)
            self.assertTrue(cov['all_pixels_covered_once'], cov)
            self.assertTrue(cov['no_empty_block'], cov)
            self.assertEqual(cov['g64_shape'], list(self.geom['shape']))

    def test_block_mean_ones(self):
        for s in (1, 2):
            sh, sw = H // s, W // s
            ones = torch.ones(1, 1, sh, sw)
            out = pool_spatial_map_to_geom(ones, s, self.geom)
            self.assertEqual(tuple(out.shape[-2:]), tuple(self.geom['shape']))
            self.assertLessEqual(float((out - 1).abs().max()), 1e-7)

    def test_block_mean_vs_python_coords(self):
        rng = np.random.RandomState(0)
        for s in (1, 2):
            sh, sw = H // s, W // s
            # coordinate maps
            yy = np.arange(sh, dtype=np.float64)[:, None] + np.zeros(sw)
            xx = np.arange(sw, dtype=np.float64)[None, :] + np.zeros((sh, 1))
            for name, arr in (('ones', np.ones((sh, sw))),
                              ('coord_y', yy), ('coord_x', xx),
                              ('rand', rng.randn(sh, sw))):
                gt = pool_spatial_map_to_geom_python(arr, s, self.geom)
                t = torch.as_tensor(arr, dtype=torch.float64)[None, None]
                got = pool_spatial_map_to_geom(t, s, self.geom).numpy()[0, 0]
                err = float(np.max(np.abs(got - gt[0])))
                self.assertLessEqual(err, 1e-7, msg='%s scale=%d err=%g' % (name, s, err))

    def test_rejects_bad_shape(self):
        bad = torch.ones(1, 1, 100, 150)
        with self.assertRaises(SystemExit):
            pool_spatial_map_to_geom(bad, 2, self.geom)


if __name__ == '__main__':
    unittest.main()
