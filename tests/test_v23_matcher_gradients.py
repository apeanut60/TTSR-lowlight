#!/usr/bin/env python
"""v2.3: dense vs chunked LocalSoftMatch forward + gradient equivalence."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.LocalRefine import LocalSoftMatch  # noqa: E402


def _clone_match(src, chunk):
    m = LocalSoftMatch(32, 4, 0.1, chunk=chunk)
    m.load_state_dict(src.state_dict())
    return m


def test_dense_chunk_forward_grad(atol=1e-5, rtol=1e-4):
    torch.manual_seed(0)
    # small spatial so dense is feasible
    h, w = 24, 32
    f0 = torch.randn(1, 32, h, w, requires_grad=True)
    fr = torch.randn(1, 32, h, w, requires_grad=True)

    dense = LocalSoftMatch(32, 4, 0.1, chunk=None)
    chunk = _clone_match(dense, chunk=48)

    t_d = dense(f0, fr)
    t_c = chunk(f0.clone().detach().requires_grad_(True),
                fr.clone().detach().requires_grad_(True))
    # rebind for clean graphs
    f0d = f0.clone().detach().requires_grad_(True)
    frd = fr.clone().detach().requires_grad_(True)
    f0c = f0.clone().detach().requires_grad_(True)
    frc = fr.clone().detach().requires_grad_(True)
    t_d = dense(f0d, frd)
    t_c = chunk(f0c, frc)
    assert torch.allclose(t_d, t_c, atol=atol, rtol=rtol), (
        float((t_d - t_c).abs().max()))

    loss_d = t_d.pow(2).mean()
    loss_c = t_c.pow(2).mean()
    loss_d.backward()
    loss_c.backward()

    assert torch.allclose(f0d.grad, f0c.grad, atol=atol, rtol=rtol), (
        float((f0d.grad - f0c.grad).abs().max()))
    assert torch.allclose(frd.grad, frc.grad, atol=atol, rtol=rtol), (
        float((frd.grad - frc.grad).abs().max()))

    for (nd, pd), (nc, pc) in zip(dense.named_parameters(),
                                  chunk.named_parameters()):
        assert nd == nc
        assert pd.grad is not None and pc.grad is not None
        assert torch.allclose(pd.grad, pc.grad, atol=atol, rtol=rtol), (
            nd, float((pd.grad - pc.grad).abs().max()))
    print('dense/chunk forward+grad OK max_fwd=%.3e' %
          float((t_d - t_c).abs().max()))


def test_qkv_update_nonzero():
    torch.manual_seed(1)
    m = LocalSoftMatch(32, 4, 0.1, chunk=64)
    opt = torch.optim.Adam(m.parameters(), lr=2e-6)
    f0 = torch.randn(1, 32, 32, 32)
    fr = torch.randn(1, 32, 32, 32)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    t = m(f0, fr)
    t.pow(2).mean().backward()
    for p in m.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    opt.step()
    changed = False
    for n, p in m.named_parameters():
        if not torch.allclose(p, before[n]):
            changed = True
    assert changed


if __name__ == '__main__':
    test_dense_chunk_forward_grad()
    test_qkv_update_nonzero()
    print('test_v23_matcher_gradients OK')
