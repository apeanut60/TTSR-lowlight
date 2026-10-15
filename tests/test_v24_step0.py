#!/usr/bin/env python
"""v2.4: step0 H3@2000 Eval30 ≈20.466; P2≈20.265 with frozen init."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import read_manifest_csv  # noqa: E402
from train_v24_end2end import V2_EXP, eval_full, load_h3, load_p2  # noqa: E402


def test_step0_eval30():
    if not torch.cuda.is_available():
        print('skip (no CUDA)'); return
    device = 'cuda'
    p2, args = load_p2(device, joint=False)
    enc, mat, head, _ = load_h3(device)
    enc.eval(); mat.eval(); head.eval()
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    summary, _ = eval_full(p2, args, enc, mat, head, ev, device)
    print('step0 Y=%.4f P2-only=%.4f' % (summary['psnr'], summary['p2_only_psnr']))
    assert abs(summary['psnr'] - 20.466) < 0.03, summary['psnr']
    assert abs(summary['p2_only_psnr'] - 20.265) < 0.05, summary['p2_only_psnr']


if __name__ == '__main__':
    test_step0_eval30()
    print('test_v24_step0 OK')
