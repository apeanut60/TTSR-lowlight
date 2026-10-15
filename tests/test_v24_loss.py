#!/usr/bin/env python
"""v2.4: MSE and audited L1+MS-SSIM loss properties."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402


def test_msssim_identity_and_grad():
    mss = MSSSIMLoss()
    x = torch.rand(1, 3, 128, 128) * 2 - 1
    x = x.clone().requires_grad_(True)
    # identical → MS-SSIM≈1 → loss≈0
    loss0 = mss(x, x.detach())
    assert float(loss0) < 1e-3, float(loss0)
    y = (x.detach() + 0.2).clamp(-1, 1)
    loss = mss(x, y)
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert torch.isfinite(loss)
    print('MSSSIM identity+grad OK loss0=%.6f loss=%.4f' % (float(loss0), float(loss)))


def test_l1_msssim_combo():
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss()
    pred = torch.randn(1, 3, 96, 96, requires_grad=True)
    gt = torch.randn(1, 3, 96, 96)
    loss = rec(pred, gt) + 1.0 * mss(pred, gt)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    print('L1+MS-SSIM combo OK')


def test_mse_grad():
    pred = torch.randn(1, 3, 64, 64, requires_grad=True)
    gt = torch.randn(1, 3, 64, 64)
    F.mse_loss(pred, gt).backward()
    assert pred.grad is not None


if __name__ == '__main__':
    test_msssim_identity_and_grad()
    test_l1_msssim_combo()
    test_mse_grad()
    print('test_v24_loss OK')
