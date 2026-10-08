"""D2=64ch@H/4, D1=32ch@H/2."""

import unittest

import torch

from model.LLFormerBridge import (D1_CHANNELS, D2_CHANNELS, OFFICIAL_LOL_CKPT,
                                  LLFormerBridge, load_into_llformer,
                                  reflect_pad_to_multiple)


class TestFeatureShapes(unittest.TestCase):
    def test_shapes(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = LLFormerBridge().to(device).eval()
        load_into_llformer(net, OFFICIAL_LOL_CKPT)
        for h, w in ((128, 128), (400, 600), (256, 384)):
            x = torch.rand(1, 3, h, w, device=device)
            xp, h0, w0 = reflect_pad_to_multiple(x)
            with torch.no_grad():
                feats = net.forward_features(xp)
            d2, d1 = feats['d2'], feats['d1']
            self.assertEqual(tuple(d2.shape[:2]), (1, D2_CHANNELS))
            self.assertEqual(tuple(d1.shape[:2]), (1, D1_CHANNELS))
            self.assertEqual(d2.shape[-2], xp.shape[-2] // 4)
            self.assertEqual(d2.shape[-1], xp.shape[-1] // 4)
            self.assertEqual(d1.shape[-2], xp.shape[-2] // 2)
            self.assertEqual(d1.shape[-1], xp.shape[-1] // 2)
            self.assertEqual(tuple(feats['y0'].shape), tuple(xp.shape))


if __name__ == '__main__':
    unittest.main()
