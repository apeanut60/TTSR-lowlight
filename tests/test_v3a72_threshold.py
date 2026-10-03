"""T3: threshold calibration has no dev scores."""

import inspect
import unittest

import numpy as np

from v3a72_runtime import calibrate_train_thresholds


class TestV3A72Threshold(unittest.TestCase):
    def test_signature_has_no_dev(self):
        names = inspect.signature(calibrate_train_thresholds).parameters
        self.assertNotIn('dev', names)
        self.assertNotIn('dev_scores', names)
        self.assertNotIn('Xdv', names)

    def test_matches_train_percentile(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=1000)
        taus = calibrate_train_thresholds(s, coverages=(0.05, 0.10))
        self.assertAlmostEqual(taus[0.05], float(np.percentile(s, 95)), places=8)
        self.assertAlmostEqual(taus[0.10], float(np.percentile(s, 90)), places=8)


if __name__ == '__main__':
    unittest.main()
