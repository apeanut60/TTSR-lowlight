#!/usr/bin/env python3
"""Full vs chunked LocalSoftMatch + head forward equivalence (small images)."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402


def test_match_dense_chunk_and_grad():
    torch.manual_seed(0)
    enc = SharedEncoder(32)
    dense = LocalSoftMatch(32, 4, 0.1, chunk=None)
    chunked = LocalSoftMatch(32, 4, 0.1, chunk=64)
    chunked.load_state_dict(dense.state_dict())
    head = V3B0ResidualFusion(96)
    # make head non-zero so grads flow
    with torch.no_grad():
        head.out.weight.normal_(0, 0.01)
        head.out.bias.normal_(0, 0.01)

    y0 = torch.randn(1, 3, 64, 80, requires_grad=False)
    nano = torch.randn(1, 3, 64, 80)
    f0 = enc((y0 + 1) * 0.5)
    fr = enc((nano + 1) * 0.5)

    # forward eq
    with torch.no_grad():
        td = dense(f0, fr)
        tc = chunked(f0, fr)
    assert torch.allclose(td, tc, atol=1e-5, rtol=1e-5)

    # gradient eq through head
    f0d = f0.detach().requires_grad_(True)
    td = dense(f0d, fr.detach())
    dd = head(f0d, td, (64, 80))
    loss_d = dd.pow(2).mean()
    loss_d.backward()
    g_dense = f0d.grad.detach().clone()

    f0c = f0.detach().requires_grad_(True)
    tc = chunked(f0c, fr.detach())
    dc = head(f0c, tc, (64, 80))
    loss_c = dc.pow(2).mean()
    loss_c.backward()
    g_chunk = f0c.grad.detach().clone()
    assert torch.allclose(g_dense, g_chunk, atol=1e-4, rtol=1e-4), float(
        (g_dense - g_chunk).abs().max())


def test_eval30_not_in_train_manifest():
    import csv
    tr = '/root/data/experiments/TTSR-lowlight-v2/manifests/manifests_train.csv'
    ev = '/root/data/experiments/TTSR-lowlight-v2/manifests/manifests_eval.csv'
    if not os.path.isfile(tr):
        return
    t = {r['sample_id'] for r in csv.DictReader(open(tr))}
    e = {r['sample_id'] for r in csv.DictReader(open(ev))}
    assert len(t) == 500 and len(e) == 30
    assert t.isdisjoint(e)


if __name__ == '__main__':
    test_match_dense_chunk_and_grad(); print('OK full/chunk')
    test_eval30_not_in_train_manifest(); print('OK split')
    print('ALL PASS')
