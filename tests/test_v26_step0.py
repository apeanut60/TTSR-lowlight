#!/usr/bin/env python
"""v2.6 step0 ≈ 20.466 / 20.265."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import read_manifest_csv  # noqa: E402
from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from train_v24_end2end import V2_EXP, load_h3, load_p2  # noqa: E402
from train_v25_loss_balance import eval_full  # noqa: E402


def test_step0():
    if not torch.cuda.is_available():
        print('skip'); return
    device = 'cuda'
    p2, args = load_p2(device, joint=False)
    enc, mat, head, _ = load_h3(device)
    enc.eval(); mat.eval(); head.eval()
    rec = ReconstructionLoss('l1'); mss = MSSSIMLoss().to(device)
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    s, _ = eval_full(p2, args, enc, mat, head, ev, device, rec, mss)
    print('step0 Y=%.4f P2=%.4f' % (s['psnr'], s['p2_psnr']))
    assert abs(s['psnr'] - 20.466) < 0.03
    assert abs(s['p2_psnr'] - 20.265) < 0.05


if __name__ == '__main__':
    test_step0()
    print('test_v26_step0 OK')
