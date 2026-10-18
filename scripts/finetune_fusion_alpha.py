#!/usr/bin/env python
"""Fit fusion scale on LOLv2-Real train; one Official-Test eval.

Networks stay frozen. Scalar (and per-channel) alpha:
  Y = clip(YL + alpha * (YN - YS), 0, 1)
Least squares ignores the clip, then metrics use the clipped output.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

SOTA = '/root/projects/TTSR-lowlight-v1-data1SOTA'
sys.path.insert(0, SOTA)
sys.path.insert(0, os.path.join(SOTA, 'scripts'))

from infer_llformer_ttsr_fusion import (  # noqa: E402
    DATA, GT_DIR, H3_CKPT, LL_CKPT, LOW_DIR, NANO_DIR, OUT, P2_CKPT,
    fuse, infer_rgb, load_llformer_bridge, read_m11, rgb_metrics,
    to_01_from_m11, ttsr_pair,
)
from train_nikon_domain import load_h3_from, load_p2_from  # noqa: E402

TRAIN_LIST = '/root/data/experiments/llformer_lolv2real/splits/llformer_train625.txt'
DEV_LIST = '/root/data/experiments/llformer_lolv2real/splits/llformer_dev64.txt'
TRAIN_LOW = os.path.join(DATA, 'Train/Low')
TRAIN_GT = os.path.join(DATA, 'Train/Normal')
TRAIN_NANO = os.path.join(DATA, 'Train/nanobanana_ref_v2')
FIT_OUT = os.path.join(OUT, 'alpha_fit')


def names(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def accumulate(p2, args, enc, mat, head, bridge, device, low_dir, gt_dir, nano_dir, file_names):
    num = 0.0
    den = 0.0
    num_c = np.zeros(3, np.float64)
    den_c = np.zeros(3, np.float64)
    n = 0
    for name in file_names:
        nano = os.path.join(nano_dir, name)
        gt_name = name.replace('low', 'normal')
        gt_path = os.path.join(gt_dir, gt_name)
        low_path = os.path.join(low_dir, name)
        if not (os.path.isfile(nano) and os.path.isfile(gt_path) and os.path.isfile(low_path)):
            continue
        low = read_m11(low_path, device)
        ref = read_m11(nano, device, size_hw=low.shape[-2:])
        gt = read_m11(gt_path, device, size_hw=low.shape[-2:])
        gt01 = to_01_from_m11(gt)
        yl = infer_rgb(bridge, to_01_from_m11(low), clamp=True)
        if yl.shape[-2:] != low.shape[-2:]:
            yl = torch.nn.functional.interpolate(
                yl, size=low.shape[-2:], mode='bilinear', align_corners=False)
        yn, ys, _ = ttsr_pair(p2, args, enc, mat, head, low, ref)
        delta = (yn - ys).double()
        resid = (gt01 - yl).double()
        num += float((resid * delta).sum())
        den += float((delta * delta).sum())
        for c in range(3):
            num_c[c] += float((resid[:, c] * delta[:, c]).sum())
            den_c[c] += float((delta[:, c] * delta[:, c]).sum())
        n += 1
        if n % 25 == 0:
            print('  fit %d  alpha~%.4f' % (n, num / max(den, 1e-12)), flush=True)
    alpha = num / max(den, 1e-12)
    alpha_c = num_c / np.maximum(den_c, 1e-12)
    return dict(n=n, alpha=float(alpha), alpha_rgb=[float(x) for x in alpha_c])


@torch.no_grad()
def eval_test_once(p2, args, enc, mat, head, bridge, device, alphas):
    """One Official-Test forward; score every alpha."""
    lows = sorted(f for f in os.listdir(LOW_DIR) if f.endswith('.png') and not f.startswith('.'))
    scores = {k: [] for k in alphas}
    ssim_s = {k: [] for k in alphas}
    for name in lows:
        low = read_m11(os.path.join(LOW_DIR, name), device)
        nano = read_m11(os.path.join(NANO_DIR, name), device, size_hw=low.shape[-2:])
        gt = read_m11(os.path.join(GT_DIR, name.replace('low', 'normal')), device,
                      size_hw=low.shape[-2:])
        yl = infer_rgb(bridge, to_01_from_m11(low), clamp=True)
        if yl.shape[-2:] != low.shape[-2:]:
            yl = torch.nn.functional.interpolate(
                yl, size=low.shape[-2:], mode='bilinear', align_corners=False)
        yn, ys, _ = ttsr_pair(p2, args, enc, mat, head, low, nano)
        for key, alpha in alphas.items():
            if isinstance(alpha, float):
                a = alpha
            else:
                a = torch.tensor(alpha, device=device, dtype=yl.dtype).view(1, 3, 1, 1)
            p, s = rgb_metrics(fuse(yl, yn, ys, a), gt)
            scores[key].append(p)
            ssim_s[key].append(s)
    return {k: dict(psnr=float(np.mean(scores[k])), ssim=float(np.mean(ssim_s[k])),
                    n=len(scores[k]), alpha=alphas[k]) for k in alphas}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bridge, _ = load_llformer_bridge(LL_CKPT, device=device, train=False)
    p2, args = load_p2_from(P2_CKPT, device, joint=False)
    p2.eval()
    enc, mat, head = load_h3_from(H3_CKPT, device, trainable=False)
    t0 = time.time()
    print('fit train625', flush=True)
    fit = accumulate(
        p2, args, enc, mat, head, bridge, device,
        TRAIN_LOW, TRAIN_GT, TRAIN_NANO, names(TRAIN_LIST))
    print('fitted', fit, flush=True)
    print('eval test once', flush=True)
    results = {'fit': fit, 'seconds_fit': time.time() - t0}
    scored = eval_test_once(p2, args, enc, mat, head, bridge, device, {
        'alpha_0': 0.0,
        'alpha_0.1': 0.1,
        'alpha_fit': fit['alpha'],
        'alpha_fit_rgb': fit['alpha_rgb'],
    })
    results.update(scored)
    for k, v in scored.items():
        print('%s  PSNR=%.4f SSIM=%.4f  a=%s' % (k, v['psnr'], v['ssim'], v['alpha']), flush=True)
    results['seconds_total'] = time.time() - t0
    results['p2_ckpt'] = P2_CKPT
    results['h3_ckpt'] = H3_CKPT
    results['llformer_ckpt'] = LL_CKPT
    results['note'] = 'Networks frozen. Alpha least-squares on train625 images that have nanobanana_ref_v2. Official Test scored once.'
    os.makedirs(FIT_OUT, exist_ok=True)
    with open(os.path.join(FIT_OUT, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', os.path.join(FIT_OUT, 'summary.json'))


if __name__ == '__main__':
    main()
