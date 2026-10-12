"""D2=64 D1=32 texture pyramid matches."""

import unittest

import torch

from model.LLFormerBridge import D1_CHANNELS, D2_CHANNELS, LLFormerBridge
from model.V6LLFormerRefBridge import v6_decode_tile
from model.V6Model import V6Model
from model.V6TextureEncoder import TEXTURE_CHANNELS


class TestV6Shapes(unittest.TestCase):
    def test_channels(self):
        self.assertEqual(TEXTURE_CHANNELS, (16, 32, 64))
        self.assertEqual(D2_CHANNELS, 64)
        self.assertEqual(D1_CHANNELS, 32)
        m = V6Model()
        self.assertEqual(m.d2_ch, 64)
        self.assertEqual(m.d1_ch, 32)

    def test_forward_shapes(self):
        base = LLFormerBridge().eval()
        branch = V6Model().eval()
        x = torch.randn(1, 3, 128, 128)
        y0 = torch.randn(1, 3, 128, 128)
        r = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            y, aux = v6_decode_tile(base, branch, x, y0, r)
        self.assertEqual(tuple(y.shape), (1, 3, 128, 128))
        self.assertEqual(aux['d2'].shape[1], 64)
        self.assertEqual(aux['d1'].shape[1], 32)


if __name__ == '__main__':
    unittest.main()
