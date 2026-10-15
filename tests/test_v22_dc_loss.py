#!/usr/bin/env python3
"""v2.2 DC-aware loss mathematical / gradient equivalence tests."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.v22_dc_loss import (  # noqa: E402
    dc_aware_residual_loss, full_mse_from_residual, mean_hw, split_dc,
)
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402


def test_mean_hw_shape():
    x = torch.randn(2, 3, 16, 20)
    m = mean_hw(x)
    assert m.shape == (2, 3, 1, 1)
    assert torch.allclose(m[0, 0], x[0, 0].mean())


def test_split_rebuild_and_zero_spatial_mean():
    d = torch.randn(2, 3, 32, 40)
    m, s = split_dc(d)
    assert torch.allclose(m + s, d, atol=1e-6)
    assert torch.allclose(mean_hw(s), torch.zeros_like(m), atol=1e-6)


def test_lambda1_equals_full_mse():
    torch.manual_seed(0)
    d_pred = torch.randn(2, 3, 64, 48, requires_grad=True)
    d_tgt = torch.randn(2, 3, 64, 48)
    total, l_sp, l_dc, _, _ = dc_aware_residual_loss(d_pred, d_tgt, lambda_dc=1.0)
    full = full_mse_from_residual(d_pred, d_tgt)
    assert torch.allclose(total, full, atol=1e-6), (float(total), float(full))


def test_lambda1_grad_equals_full_mse():
    torch.manual_seed(1)
    d_pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    d_tgt = torch.randn(2, 3, 32, 32)
    total, _, _, _, _ = dc_aware_residual_loss(d_pred, d_tgt, lambda_dc=1.0)
    total.backward()
    g1 = d_pred.grad.detach().clone()
    d_pred.grad = None
    full = full_mse_from_residual(d_pred, d_tgt)
    full.backward()
    g2 = d_pred.grad.detach().clone()
    assert torch.allclose(g1, g2, atol=1e-6), float((g1 - g2).abs().max())


def test_lambda2_changes_dc_grad_not_spatial_formula():
    torch.manual_seed(2)
    # Construct prediction with nontrivial mean AND spatial
    base = torch.randn(1, 3, 32, 32)
    m = torch.tensor([0.1, -0.05, 0.02]).view(1, 3, 1, 1)
    d_pred = (base - mean_hw(base) + m).detach().requires_grad_(True)
    d_tgt = torch.zeros_like(d_pred)
    t1, sp1, dc1, _, _ = dc_aware_residual_loss(d_pred, d_tgt, lambda_dc=1.0)
    t1.backward()
    g1 = d_pred.grad.detach().clone()
    d_pred.grad = None
    t2, sp2, dc2, _, _ = dc_aware_residual_loss(d_pred, d_tgt, lambda_dc=2.0)
    t2.backward()
    g2 = d_pred.grad.detach().clone()
    assert torch.allclose(sp1, sp2)
    assert torch.allclose(dc1, dc2)
    assert not torch.allclose(g1, g2, atol=1e-7)
    # DC component of gradient should scale: mean of g over HW
    # For λ=2, total = sp + 2*dc; grad w.r.t mean part doubles relative to λ=1
    # Check g2 mean magnitude > g1 for channels
    assert float(mean_hw(g2).abs().sum()) > float(mean_hw(g1).abs().sum())


def test_head_step0_zero_residual():
    head = V3B0ResidualFusion(96)
    f0 = torch.randn(1, 32, 16, 20)
    t = torch.randn(1, 32, 16, 20)
    with torch.no_grad():
        d = head(f0, t, (32, 40))
    assert float(d.abs().max()) < 1e-6


if __name__ == '__main__':
    test_mean_hw_shape(); print('OK mean')
    test_split_rebuild_and_zero_spatial_mean(); print('OK split')
    test_lambda1_equals_full_mse(); print('OK mse eq')
    test_lambda1_grad_equals_full_mse(); print('OK grad eq')
    test_lambda2_changes_dc_grad_not_spatial_formula(); print('OK lambda2')
    test_head_step0_zero_residual(); print('OK zero')
    print('ALL PASS')
