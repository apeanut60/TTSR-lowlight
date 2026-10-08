"""HARD: official LLFormer.forward == LLFormerBridge.forward (max_abs <= 1e-6)."""

import os
import unittest

import torch

from llformer_runtime import DATA_DIR, load_rgb01
from model.LLFormerBridge import (OFFICIAL_LOL_CKPT, build_official_llformer,
                                  load_into_llformer, LLFormerBridge,
                                  reflect_pad_to_multiple)

ABS_TOL = 1e-6


def _sample_paths(n=8):
    low_dir = os.path.join(DATA_DIR, 'Train', 'Low')
    names = sorted(f for f in os.listdir(low_dir) if f.endswith('.png'))
    # mix early + mid + late
    picks = names[:n // 2] + names[len(names) // 2:len(names) // 2 + n // 2]
    return [os.path.join(low_dir, n) for n in picks[:n]]


class TestBridgeEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.off = build_official_llformer().to(cls.device).eval()
        load_into_llformer(cls.off, OFFICIAL_LOL_CKPT)
        cls.br = LLFormerBridge().to(cls.device).eval()
        load_into_llformer(cls.br, OFFICIAL_LOL_CKPT)
        for p in cls.off.parameters():
            p.requires_grad_(False)
        for p in cls.br.parameters():
            p.requires_grad_(False)

    def _max_abs(self, a, b):
        return float((a - b).abs().max())

    def test_train_images(self):
        worst = 0.0
        with torch.no_grad():
            for path in _sample_paths(8):
                x = load_rgb01(path)[None].to(self.device)
                xp, _, _ = reflect_pad_to_multiple(x)
                ya = self.off(xp)
                yb = self.br(xp)
                d = self._max_abs(ya, yb)
                worst = max(worst, d)
                self.assertLessEqual(d, ABS_TOL, path)
        print('[equiv] train8 worst max_abs=%.3e' % worst, flush=True)

    def test_non_multiple_of_16(self):
        # synthetic odd sizes
        worst = 0.0
        with torch.no_grad():
            for h, w in ((400, 600), (401, 603), (127, 255), (513, 257)):
                x = torch.rand(1, 3, h, w, device=self.device)
                xp, h0, w0 = reflect_pad_to_multiple(x)
                ya = self.off(xp)[..., :h0, :w0]
                yb = self.br(xp)[..., :h0, :w0]
                d = self._max_abs(ya, yb)
                worst = max(worst, d)
                self.assertLessEqual(d, ABS_TOL, (h, w))
        print('[equiv] odd-size worst max_abs=%.3e' % worst, flush=True)

    def test_bridge_matches_parent_path(self):
        x = torch.rand(1, 3, 128, 128, device=self.device)
        with torch.no_grad():
            y_staged = self.br.forward_staged(x)
            y_parent = self.br.forward_official_path(x)
        self.assertLessEqual(self._max_abs(y_staged, y_parent), ABS_TOL)


if __name__ == '__main__':
    unittest.main()
