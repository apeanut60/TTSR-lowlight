#!/usr/bin/env python3
"""v2 LocalRefine / V3B0 engineering tests."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.LocalRefine import LocalSoftMatch, Refiner, count_params  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402


def test_param_count():
    assert count_params(Refiner()) == 78724


def test_step0_identity():
    m = Refiner().eval()
    y = torch.randn(2, 3, 64, 64)
    r = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        sr, aux = m(y, r)
    assert torch.allclose(sr, y, atol=1e-6)
    assert float(aux['delta'].abs().max()) < 1e-6


def test_dense_chunk_match():
    torch.manual_seed(0)
    f0 = torch.randn(1, 32, 24, 32)
    fr = torch.randn(1, 32, 24, 32)
    dense = LocalSoftMatch(32, radius=4, tau=0.1, chunk=None)
    chunked = LocalSoftMatch(32, radius=4, tau=0.1, chunk=128)
    chunked.load_state_dict(dense.state_dict())
    with torch.no_grad():
        td = dense(f0, fr)
        tc = chunked(f0, fr)
    assert torch.allclose(td, tc, atol=1e-5, rtol=1e-5)


def test_neighbour_no_row_wrap():
    # radius-4 indices stay in-bounds and do not wrap rows
    from model.LocalRefine import _neighbour_indices
    flat, valid = _neighbour_indices(16, 16, 4)
    assert flat.shape == (256, 81)
    assert valid.shape == (256, 81)
    # position (0,0): only bottom-right neighbourhood valid
    assert int(valid[0].sum()) < 81
    # center fully valid (y=8,x=8) with radius 4 inside 16x16
    mid = 8 * 16 + 8
    assert int(valid[mid].sum()) == 81
    # no row wrap: for left border, invalid left offsets are False
    assert int(valid[8 * 16 + 0].sum()) < 81


def test_v3b0_shapes_and_zero():
    head = V3B0ResidualFusion(in_ch=96)
    f0 = torch.randn(1, 32, 32, 40)
    t = torch.randn(1, 32, 32, 40)
    with torch.no_grad():
        dy = head(f0, t, (64, 80))
    assert dy.shape == (1, 3, 64, 80)
    assert float(dy.abs().max()) < 1e-6


def test_c_out_grad():
    m = Refiner()
    y = torch.randn(2, 3, 64, 64, requires_grad=False)
    r = torch.randn(2, 3, 64, 64)
    sr, _ = m(y, r)
    sr.mean().backward()
    assert m.c_out.weight.grad is not None
    assert float(m.c_out.weight.grad.abs().sum()) > 0


def test_huawei_manifest_500():
    import csv
    p = '/root/data/experiments/TTSR-lowlight-v2/manifests/manifests_train.csv'
    if not os.path.isfile(p):
        p = os.path.join(ROOT, 'diagnostics/v2/huawei_train_manifest.csv')
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 500
    for r in rows:
        assert r['camera'] == 'Huawei'
        assert os.path.isfile(r['low_path'])
        assert os.path.isfile(r.get('high_path') or r['gt_path'])
        assert os.path.isfile(r['nano_path'])
        # must not use GT as nano
        assert os.path.abspath(r['nano_path']) != os.path.abspath(
            r.get('high_path') or r['gt_path'])


if __name__ == '__main__':
    test_param_count()
    print('OK params')
    test_step0_identity()
    print('OK identity')
    test_dense_chunk_match()
    print('OK dense/chunk')
    test_neighbour_no_row_wrap()
    print('OK indices')
    test_v3b0_shapes_and_zero()
    print('OK v3b0')
    test_c_out_grad()
    print('OK grad')
    test_huawei_manifest_500()
    print('OK manifest')
    print('ALL PASS')
