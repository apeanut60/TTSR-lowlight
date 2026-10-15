#!/usr/bin/env python
"""v2.4: A0/A2 freeze vs joint parameter groups."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from train_v24_end2end import (  # noqa: E402
    LV2_UNUSED, configure_p2, h3_forward, load_h3, load_p2, p2_forward,
)


def test_a0_p2_frozen_h3_grads():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p2, _ = load_p2(device, joint=False)
    enc, mat, head, _ = load_h3(device, chunk=256)
    low = torch.randn(1, 3, 128, 128, device=device)
    nano = torch.randn(1, 3, 128, 128, device=device)
    gt = torch.randn(1, 3, 128, 128, device=device)
    before = {n: p.detach().cpu().clone() for n, p in p2.named_parameters()}
    y0 = p2_forward(p2, low, nano, joint=False)
    assert not y0.requires_grad
    y, _ = h3_forward(enc, mat, head, y0, nano)
    torch.nn.functional.mse_loss(y, gt).backward()
    assert all(p.grad is None for p in p2.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    # no param change without optimizer; freeze check is requires_grad
    assert all(not p.requires_grad for p in p2.parameters())
    print('A0 freeze OK')


def test_a2_p2_grads():
    if not torch.cuda.is_available():
        print('skip A2 (no CUDA)'); return
    device = 'cuda'
    p2, _ = load_p2(device, joint=True)
    enc, mat, head, _ = load_h3(device, chunk=256)
    # break head zero-init so grads reach P2
    torch.nn.init.normal_(head.out.weight, std=1e-3)
    low = torch.randn(1, 3, 128, 128, device=device)
    nano = torch.randn(1, 3, 128, 128, device=device)
    gt = torch.randn(1, 3, 128, 128, device=device)
    y0 = p2_forward(p2, low, nano, joint=True)
    assert y0.requires_grad
    y, _ = h3_forward(enc, mat, head, y0, nano)
    torch.nn.functional.mse_loss(y, gt).backward()
    n_g = sum(1 for p in p2.parameters() if p.requires_grad and p.grad is not None
              and p.grad.abs().sum() > 0)
    n_t = sum(1 for p in p2.parameters() if p.requires_grad)
    assert n_g == n_t, (n_g, n_t)
    for n in LV2_UNUSED:
        p = dict(p2.named_parameters())[n]
        assert not p.requires_grad
    assert any('LTE' in n and not p.requires_grad for n, p in p2.named_parameters())
    # optimizer step changes P2
    opt = torch.optim.Adam([p for p in p2.parameters() if p.requires_grad], lr=1e-6)
    before = {n: p.detach().clone() for n, p in p2.named_parameters() if p.requires_grad}
    opt.step()
    changed = any(not torch.allclose(p, before[n])
                  for n, p in p2.named_parameters() if n in before)
    assert changed
    print('A2 joint grads OK n_g=%d' % n_g)


if __name__ == '__main__':
    test_a0_p2_frozen_h3_grads()
    test_a2_p2_grads()
    print('test_v24_freeze_groups OK')
