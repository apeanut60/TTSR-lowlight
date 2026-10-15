#!/usr/bin/env python
"""v2.3: H0/H1/H2 step0 forward matches H3@2000 on Eval30 probe / full if GPU."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, load_cache, metrics, read_manifest_csv, _read_eval_tensor,
)
from train_v23_joint_ft import (  # noqa: E402
    V2_EXP, configure_trainable, forward_h3, load_models,
)

H3_REF_PSNR = 20.466


@torch.no_grad()
def _eval_psnr(enc, mat, head, rows, cache, device, limit=None):
    enc.eval(); mat.eval(); head.eval()
    vals = []
    for row in (rows if limit is None else rows[:limit]):
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(cache, row)))[None].to(device)
        gt = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        f0 = enc((y0 + 1.) * 0.5)
        fr = enc((nano + 1.) * 0.5)
        y = y0 + head(f0, mat(f0, fr), y0.shape[-2:])
        p, _, _ = metrics(y, gt)
        vals.append(p)
    return float(np.mean(vals))


def test_arms_identical_step0_toy():
    device = 'cpu'
    enc, mat, head, _ = load_models(device, chunk=64)
    y0 = torch.randn(1, 3, 48, 64)
    nano = torch.randn(1, 3, 48, 64)
    outs = []
    for arm in ('H0', 'H1', 'H2'):
        configure_trainable(arm, enc, mat, head)
        enc.eval(); mat.eval(); head.eval()
        y, d, _, _ = forward_h3(enc, mat, head, y0, nano, arm)
        outs.append(y.detach().clone())
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    assert torch.allclose(outs[0], outs[2], atol=1e-6)
    print('H0/H1/H2 step0 toy identical OK')


def test_h3_eval30_psnr():
    if not torch.cuda.is_available():
        print('skip full Eval30 (no CUDA)')
        return
    device = 'cuda'
    enc, mat, head, _ = load_models(device, chunk=EVAL_QUERY_CHUNK)
    enc.eval(); mat.eval(); head.eval()
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    cache = os.path.join(V2_EXP, 'cache_p2_eval')
    psnr = _eval_psnr(enc, mat, head, ev, cache, device)
    print('H3@2000 Eval30 PSNR=%.4f (ref≈%.3f)' % (psnr, H3_REF_PSNR))
    assert abs(psnr - H3_REF_PSNR) < 0.02, psnr


if __name__ == '__main__':
    test_arms_identical_step0_toy()
    test_h3_eval30_psnr()
    print('test_v23_step0_equivalence OK')
