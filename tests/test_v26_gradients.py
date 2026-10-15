#!/usr/bin/env python
"""v2.6: joint path grads to P2 under L1+MS-SSIM."""
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


def test_joint_grads():
    if not torch.cuda.is_available():
        print('skip'); return
    device = 'cuda'
    p2, _ = load_p2(device, joint=True)
    enc, mat, head, _ = load_h3(device, chunk=256)
    torch.nn.init.normal_(head.out.weight, std=1e-3)
    rec = ReconstructionLoss('l1'); mss = MSSSIMLoss().to(device)
    low = torch.randn(1, 3, 192, 192, device=device)
    nano = torch.randn(1, 3, 192, 192, device=device)
    gt = torch.randn(1, 3, 192, 192, device=device)
    y0 = p2_forward(p2, low, nano, joint=True)
    assert y0.requires_grad
    y, _ = h3_forward(enc, mat, head, y0, nano)
    loss, _ = compute_l1_mss(y, gt, rec, mss, 1.0)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in p2.parameters() if p.requires_grad)
    # LTE frozen
    assert all(not p.requires_grad for n, p in p2.named_parameters() if 'LTE' in n)
    print('joint grads OK')


if __name__ == '__main__':
    test_joint_grads()
    print('test_v26_gradients OK')
