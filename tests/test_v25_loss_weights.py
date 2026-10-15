#!/usr/bin/env python
"""v2.5: λ weighting, identity MS-SSIM, λ=1 matches A3 recipe."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from train_v25_loss_balance import compute_l1_mss  # noqa: E402


def test_msssim_identity():
    mss = MSSSIMLoss()
    x = torch.rand(1, 3, 128, 128) * 2 - 1
    assert float(mss(x, x)) < 1e-3


def test_lambda_scales_total_not_components():
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss()
    pred = torch.randn(1, 3, 96, 96, requires_grad=True)
    gt = torch.randn(1, 3, 96, 96)
    l1 = float(rec(pred, gt).detach())
    ms = float(mss(pred, gt).detach())
    for lam in (1.0, 0.3, 0.1):
        loss, parts = compute_l1_mss(pred, gt, rec, mss, lam)
        assert abs(parts['l1'] - l1) < 1e-5
        assert abs(parts['ms_ssim'] - ms) < 1e-5
        assert abs(float(loss) - (l1 + lam * ms)) < 1e-5
    # grads change with λ
    pred2 = torch.randn(1, 3, 96, 96, requires_grad=True)
    gt2 = torch.randn(1, 3, 96, 96)
    loss1, _ = compute_l1_mss(pred2, gt2, rec, mss, 1.0)
    loss1.backward()
    g1 = pred2.grad.detach().clone()
    pred2.grad = None
    loss01, _ = compute_l1_mss(pred2, gt2, rec, mss, 0.1)
    loss01.backward()
    g01 = pred2.grad.detach()
    assert not torch.allclose(g1, g01)
    print('lambda scale OK')


if __name__ == '__main__':
    test_msssim_identity()
    test_lambda_scales_total_not_components()
    print('test_v25_loss_weights OK')
