"""V4 A0/A1 heads share one init state_dict."""

import unittest

import torch

from model.V3BResidualFusion import V3B0ResidualFusion
from v3a5_runtime import bit_equal, snapshot_, state_dict_sha


class TestV4SharedInit(unittest.TestCase):
    def test_bit_equal_from_one_sd(self):
        torch.manual_seed(42)
        shared = V3B0ResidualFusion(96)
        sd = snapshot_(shared)
        a0 = V3B0ResidualFusion(96)
        a1 = V3B0ResidualFusion(96)
        a0.load_state_dict(sd, strict=True)
        a1.load_state_dict(sd, strict=True)
        self.assertTrue(bit_equal(snapshot_(a0), snapshot_(a1)))
        self.assertEqual(state_dict_sha(snapshot_(a0)), state_dict_sha(snapshot_(a1)))
        self.assertEqual(state_dict_sha(snapshot_(a0)), state_dict_sha(sd))
