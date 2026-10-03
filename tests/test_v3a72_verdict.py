"""T4: overall ranking without correct-only ranking is not regional GO."""

import unittest

from v3a72_runtime import verdict_v3a72


class TestV3A72Verdict(unittest.TestCase):
    def test_overall_strong_correct_weak_is_not_case_a(self):
        payload = dict(
            bootstrap_correct={
                'top10_mean_u': dict(lo=-0.01, hi=0.02, mean=0.0),
                'top10_minus_bottom50': dict(lo=-0.02, hi=0.01, mean=0.0),
            },
            bins_correct={
                'top_10': dict(mean_u=0.0001, pos_rate=0.51),
                'top_50': dict(mean_u=0.0002),
                'bottom_50': dict(mean_u=0.0003),
            },
            composition_top10={'correct': 0.92, 'true_dark_g0.5': 0.04,
                               'mismatch': 0.04},
            p1_primary=dict(mean_psnr=19.85),
            psnr_base=19.70,
            psnr_v3a6_a1=19.83,
            safety_p1={
                'true_dark_g0.5': dict(mean=0.0, large_harm_rate=0.0),
                'mismatch': dict(mean=0.0, large_harm_rate=0.0),
            },
            safety_r1={
                'true_dark_g0.5': dict(large_harm_rate=0.1),
                'mismatch': dict(large_harm_rate=0.1),
            },
            coverage_monotonic=True,
            other_state_top10_negative=False,
        )
        v = verdict_v3a72(payload)
        self.assertNotEqual(v['label'], 'V3A72_CASE_A_REGIONAL_SELECTIVE_GO')
        self.assertEqual(v['label'], 'V3A72_CASE_B_COARSE_STATE_ONLY')

    def test_case_a_requires_ci_and_p1(self):
        payload = dict(
            bootstrap_correct={
                'top10_mean_u': dict(lo=0.002, hi=0.01, mean=0.005),
                'top10_minus_bottom50': dict(lo=0.003, hi=0.02, mean=0.01),
            },
            bins_correct={
                'top_10': dict(mean_u=0.01, pos_rate=0.75),
                'top_50': dict(mean_u=0.004),
                'bottom_50': dict(mean_u=-0.002),
            },
            composition_top10={'correct': 0.4},
            p1_primary=dict(mean_psnr=19.90),
            psnr_base=19.70,
            psnr_v3a6_a1=19.83,
            safety_p1={
                'true_dark_g0.5': dict(mean=0.01, large_harm_rate=0.0),
                'mismatch': dict(mean=0.01, large_harm_rate=0.0),
            },
            safety_r1={
                'true_dark_g0.5': dict(large_harm_rate=0.2),
                'mismatch': dict(large_harm_rate=0.2),
            },
            coverage_monotonic=True,
            other_state_top10_negative=False,
        )
        v = verdict_v3a72(payload)
        self.assertEqual(v['label'], 'V3A72_CASE_A_REGIONAL_SELECTIVE_GO')


if __name__ == '__main__':
    unittest.main()
