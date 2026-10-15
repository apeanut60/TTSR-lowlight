#!/usr/bin/env python
"""v2.4: end-to-end Loss → H3 → Y0 → P2 gradient path."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    compute_loss, h3_forward, load_h3, load_p2, p2_forward,
)


def test_e2e_mse_and_l1msssim():
    if not torch.cuda.is_available():
        print('skip (no CUDA)'); return
    device = 'cuda'
    p2, _ = load_p2(device, joint=True)
    enc, mat, head, _ = load_h3(device, chunk=256)
    torch.nn.init.normal_(head.out.weight, std=1e-3)
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)
    low = torch.randn(1, 3, 128, 128, device=device)
    nano = torch.randn(1, 3, 128, 128, device=device)
    gt = torch.randn(1, 3, 128, 128, device=device)
    for kind in ('mse', 'l1msssim'):
        p2.zero_grad(set_to_none=True)
        enc.zero_grad(set_to_none=True)
        mat.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        y0 = p2_forward(p2, low, nano, joint=True)
        assert y0.requires_grad
        y, _ = h3_forward(enc, mat, head, y0, nano)
        loss, _ = compute_loss(y, gt, kind, rec, mss)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in p2.parameters() if p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in enc.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in mat.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in head.parameters())
        print('e2e %s OK loss=%.4f' % (kind, float(loss)))


if __name__ == '__main__':
    test_e2e_mse_and_l1msssim()
    print('test_v24_gradient_flow OK')
