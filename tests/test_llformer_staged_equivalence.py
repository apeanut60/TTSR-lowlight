"""Normal Bridge forward == staged pass-through."""

import unittest

import torch

from model.LLFormerBridge import (OFFICIAL_LOL_CKPT, LLFormerBridge,
                                  load_into_llformer)

ABS_TOL = 1e-6


class TestStagedEquivalence(unittest.TestCase):
    def test_staged_equals_forward(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = LLFormerBridge().to(device).eval()
        load_into_llformer(net, OFFICIAL_LOL_CKPT)
        x = torch.rand(1, 3, 128, 128, device=device)
        with torch.no_grad():
            y0 = net.forward(x)
            st = net.encode_to_d2(x)
            st = net.decode_d2_to_d1(st, st['d2'])
            y1 = net.decode_d1_to_rgb(st, st['d1'])
            y2 = net.forward_staged(x)
        self.assertLessEqual(float((y0 - y1).abs().max()), ABS_TOL)
        self.assertLessEqual(float((y0 - y2).abs().max()), ABS_TOL)


if __name__ == '__main__':
    unittest.main()
