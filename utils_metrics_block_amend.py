"""Metrics / export protocol ``rgb_float_clipped_v1`` for block-amend."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from utils import calc_psnr_rgb, calc_ssim


PROTOCOL_NAME = 'rgb_float_clipped_v1'


def sr_to_01_raw(sr_neg11: torch.Tensor) -> torch.Tensor:
    return (sr_neg11 + 1.0) * 0.5


def clip01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def overrun_stats(sr01_raw: torch.Tensor) -> Dict[str, float]:
    flat = sr01_raw.detach().float()
    n = flat.numel()
    return {
        'frac_lt0': float((flat < 0).sum().item()) / max(n, 1),
        'frac_gt1': float((flat > 1).sum().item()) / max(n, 1),
        'raw_min': float(flat.min().item()),
        'raw_max': float(flat.max().item()),
    }


def _to_hwc_u8_float255(t01_clipped: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] or [3,H,W] in [0,1] → HWC float64 in [0,255] (not quantized)."""
    x = t01_clipped.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    arr = x.permute(1, 2, 0).numpy().astype(np.float64) * 255.0
    return arr


def metrics_rgb_float_clipped_v1(
    sr_neg11: torch.Tensor, hr_neg11: torch.Tensor,
) -> Dict[str, float]:
    """Per-image RGB float PSNR + mean channel SSIM on clipped images.

    Protocol: map [-1,1]→[0,1], clip, scale to [0,255] float (no uint8 round)
    for PSNR/SSIM. Sizes must match exactly.
    """
    if tuple(sr_neg11.shape) != tuple(hr_neg11.shape):
        raise ValueError(
            'sr/GT shape mismatch: %s vs %s (auto-crop forbidden)'
            % (tuple(sr_neg11.shape), tuple(hr_neg11.shape)))
    sr_raw = sr_to_01_raw(sr_neg11)
    hr_raw = sr_to_01_raw(hr_neg11)
    stats = overrun_stats(sr_raw)
    sr_c = _to_hwc_u8_float255(clip01(sr_raw))
    hr_c = _to_hwc_u8_float255(clip01(hr_raw))
    psnr = float(calc_psnr_rgb(sr_c, hr_c))
    ssim = float(calc_ssim(sr_c, hr_c))
    out = {'psnr_rgb': psnr, 'ssim_rgb': ssim, 'protocol': PROTOCOL_NAME}
    out.update(stats)
    return out


def to_uint8_png(sr_neg11: torch.Tensor) -> np.ndarray:
    """Clip then round to uint8 HWC. Never wrap via unchecked astype."""
    sr01 = clip01(sr_to_01_raw(sr_neg11.detach().float().cpu()))
    if sr01.dim() == 4:
        sr01 = sr01[0]
    arr = sr01.permute(1, 2, 0).numpy()
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
