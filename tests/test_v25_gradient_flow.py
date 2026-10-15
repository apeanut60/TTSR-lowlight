#!/usr/bin/env python
"""v2.5: L1+λ MS-SSIM backprops to P2 for λ in {1,0.3,0.1}."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from train_v24_end2end import h3_forward, load_h3, load_p2, p2_forward  # noqa: E402
from train_v25_loss_balance import compute_l1_mss  # noqa: E402


def test_p2_grads_all_lambdas():
    if not torch.cuda.is_available():
        print('skip'); return
    device = 'cuda'
    p2, _ = load_p2(device, joint=True)
    enc, mat, head, _ = load_h3(device, chunk=256)
    torch.nn.init.normal_(head.out.weight, std=1e-3)
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)
    low = torch.randn(1, 3, 128, 128, device=device)
    nano = torch.randn(1, 3, 128, 128, device=device)
    gt = torch.randn(1, 3, 128, 128, device=device)
    for lam in (1.0, 0.3, 0.1):
        p2.zero_grad(set_to_none=True)
        enc.zero_grad(set_to_none=True)
        mat.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        y0 = p2_forward(p2, low, nano, joint=True)
        assert y0.requires_grad
        y, _ = h3_forward(enc, mat, head, y0, nano)
        loss, _ = compute_l1_mss(y, gt, rec, mss, lam)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in p2.parameters() if p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in head.parameters())
        print('λ=%.1f OK loss=%.4f' % (lam, float(loss)))


if __name__ == '__main__':
    test_p2_grads_all_lambdas()
    print('test_v25_gradient_flow OK')
