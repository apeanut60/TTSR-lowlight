#!/usr/bin/env python
"""v2.3: verify H0/H1/H2 requires_grad and non-zero grads on a toy batch."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from train_v23_joint_ft import configure_trainable, forward_h3  # noqa: E402


def _modules():
    enc = SharedEncoder(32)
    mat = LocalSoftMatch(32, 4, 0.1, chunk=64)
    head = V3B0ResidualFusion(96)
    # Break zero-init on last conv so grads reach Matcher/Encoder in unit tests.
    # Real training loads H3@2000 where out is already non-zero.
    torch.nn.init.normal_(head.out.weight, std=1e-3)
    torch.nn.init.zeros_(head.out.bias)
    return enc, mat, head


def _has_grad(mod):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())


def _any_grad_tensor(mod):
    return any(p.grad is not None for p in mod.parameters())


def _run(arm):
    enc, mat, head = _modules()
    configure_trainable(arm, enc, mat, head)
    y0 = torch.randn(1, 3, 64, 64)
    nano = torch.randn(1, 3, 64, 64)
    gt = torch.randn(1, 3, 64, 64)
    y, _, _, _ = forward_h3(enc, mat, head, y0, nano, arm)
    loss = torch.nn.functional.mse_loss(y, gt)
    loss.backward()
    return enc, mat, head


def test_h0_grads():
    enc, mat, head = _run('H0')
    assert _has_grad(head)
    assert not _any_grad_tensor(enc)
    assert not _any_grad_tensor(mat)


def test_h1_grads():
    enc, mat, head = _run('H1')
    assert _has_grad(head)
    assert _has_grad(mat)
    assert not _any_grad_tensor(enc)


def test_h2_grads():
    enc, mat, head = _run('H2')
    assert _has_grad(head)
    assert _has_grad(mat)
    assert _has_grad(enc)


def test_shared_encoder_single_instance():
    enc, mat, head = _modules()
    configure_trainable('H2', enc, mat, head)
    y0 = torch.randn(1, 3, 32, 32)
    nano = torch.randn(1, 3, 32, 32)
    # both paths must share weights: same module id
    f0 = enc((y0 + 1) * 0.5)
    fr = enc((nano + 1) * 0.5)
    assert f0.shape == fr.shape
    # one optimizer group for encoder
    groups, names = configure_trainable('H2', enc, mat, head)
    enc_groups = [g for g in groups if g['name'] == 'encoder']
    assert len(enc_groups) == 1
    assert 'SharedEncoder' in names


if __name__ == '__main__':
    test_h0_grads()
    test_h1_grads()
    test_h2_grads()
    test_shared_encoder_single_instance()
    print('test_v23_trainable_groups OK')
