#!/usr/bin/env python
"""Fixed-alpha fusion with LOLv2 H3@250, then a 2-epoch scalar-alpha fit.

Networks stay frozen. Official Test is forwarded once.
  Y = clip(YL + alpha * (YN - YS), 0, 1)
Alpha starts at 0.1 and is trained for 2 epochs on train625 images that
have nanobanana_ref_v2. Metrics are RGB (calc_psnr_rgb / mean-channel SSIM).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SOTA = '/root/projects/TTSR-lowlight-v1-data1SOTA'
sys.path.insert(0, SOTA)
sys.path.insert(0, os.path.join(SOTA, 'scripts'))

from infer_llformer_ttsr_fusion import (  # noqa: E402
    DATA, GT_DIR, LL_CKPT, LOW_DIR, NANO_DIR,
    fuse, infer_rgb, load_llformer_bridge, read_m11, rgb_metrics,
    to_01_from_m11, ttsr_pair,
)
from train_nikon_domain import load_h3_from, load_p2_from  # noqa: E402

CKPT = (
    '/root/data/experiments/TTSR-lowlight-lolv2real/checkpoints/'
    'H3_frozenP2_mse_s42/checkpoint_00250.pt')
TRAIN_LIST = '/root/data/experiments/llformer_lolv2real/splits/llformer_train625.txt'
TRAIN_LOW = os.path.join(DATA, 'Train/Low')
TRAIN_GT = os.path.join(DATA, 'Train/Normal')
TRAIN_NANO = os.path.join(DATA, 'Train/nanobanana_ref_v2')
OUT = '/root/data/experiments/TTSR-LLFormer-Fusion/h3ft250'


def names(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def branch(bridge, p2, args, enc, mat, head, low_path, gt_path, nano_path, device):
    low = read_m11(low_path, device)
    nano = read_m11(nano_path, device, size_hw=low.shape[-2:])
    gt = read_m11(gt_path, device, size_hw=low.shape[-2:])
    yl = infer_rgb(bridge, to_01_from_m11(low), clamp=True)
    if yl.shape[-2:] != low.shape[-2:]:
        yl = torch.nn.functional.interpolate(
            yl, size=low.shape[-2:], mode='bilinear', align_corners=False)
    yn, ys, _ = ttsr_pair(p2, args, enc, mat, head, low, nano)
    return yl, yn - ys, gt


def pack(yl, delta, gt):
    return (
        yl.squeeze(0).half().cpu(),
        delta.squeeze(0).half().cpu(),
        gt.squeeze(0).half().cpu(),
    )


def score_cached(cache, alpha, device):
    psnrs, ssims = [], []
    a = alpha
    if not isinstance(alpha, float):
        a = float(alpha)
    for yl, delta, gt in cache:
        pred = fuse(
            yl.float()[None].to(device),
            delta.float()[None].to(device),
            torch.zeros(1, device=device),
            a)
        p, s = rgb_metrics(pred, gt.float()[None].to(device))
        psnrs.append(p)
        ssims.append(s)
    return dict(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)), n=len(psnrs), alpha=a)


def train_alpha(cache, device, epochs=2, lr=1e-3, init=0.1):
    alpha = torch.nn.Parameter(torch.tensor(init, device=device))
    opt = torch.optim.Adam([alpha], lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        for yl, delta, gt in cache:
            yl_d = yl.float()[None].to(device)
            d = delta.float()[None].to(device)
            gt01 = to_01_from_m11(gt.float()[None].to(device))
            pred = torch.clamp(yl_d + alpha * d, 0.0, 1.0)
            loss = F.mse_loss(pred, gt01)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        rec = dict(epoch=epoch, alpha=float(alpha.detach()), train_mse=total / len(cache))
        history.append(rec)
        print('  epoch %d alpha=%.5f train_mse=%.6f' % (
            epoch, rec['alpha'], rec['train_mse']), flush=True)
    return float(alpha.detach()), history


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    bridge, _ = load_llformer_bridge(LL_CKPT, device=device, train=False)
    p2, args = load_p2_from(CKPT, device, joint=False)
    p2.eval()
    enc, mat, head = load_h3_from(CKPT, device, trainable=False)

    print('cache official test', flush=True)
    test_cache = []
    lows = sorted(f for f in os.listdir(LOW_DIR) if f.endswith('.png') and not f.startswith('.'))
    for i, name in enumerate(lows):
        yl, delta, gt = branch(
            bridge, p2, args, enc, mat, head,
            os.path.join(LOW_DIR, name),
            os.path.join(GT_DIR, name.replace('low', 'normal')),
            os.path.join(NANO_DIR, name), device)
        test_cache.append(pack(yl, delta, gt))
        if (i + 1) % 25 == 0:
            print('  test %d' % (i + 1), flush=True)
    fixed = score_cached(test_cache, 0.1, device)
    base = score_cached(test_cache, 0.0, device)
    print('alpha_0    PSNR=%.4f SSIM=%.4f' % (base['psnr'], base['ssim']), flush=True)
    print('alpha_0.1  PSNR=%.4f SSIM=%.4f' % (fixed['psnr'], fixed['ssim']), flush=True)

    print('cache train625', flush=True)
    train_cache = []
    for name in names(TRAIN_LIST):
        nano = os.path.join(TRAIN_NANO, name)
        gt_path = os.path.join(TRAIN_GT, name.replace('low', 'normal'))
        low_path = os.path.join(TRAIN_LOW, name)
        if not (os.path.isfile(nano) and os.path.isfile(gt_path) and os.path.isfile(low_path)):
            continue
        yl, delta, gt = branch(
            bridge, p2, args, enc, mat, head, low_path, gt_path, nano, device)
        train_cache.append(pack(yl, delta, gt))
        if len(train_cache) % 50 == 0:
            print('  train %d' % len(train_cache), flush=True)
    print('short train 2 epochs n=%d' % len(train_cache), flush=True)
    alpha, history = train_alpha(train_cache, device, epochs=2, lr=1e-3, init=0.1)
    learned = score_cached(test_cache, alpha, device)
    print('alpha_2ep  PSNR=%.4f SSIM=%.4f  a=%.5f' % (
        learned['psnr'], learned['ssim'], alpha), flush=True)

    summary = dict(
        ckpt=CKPT,
        p2_and_h3='checkpoint_00250.pt',
        llformer_ckpt=LL_CKPT,
        n_train=len(train_cache),
        n_test=len(test_cache),
        metric='RGB calc_psnr_rgb and mean-channel SSIM',
        alpha_0=base,
        alpha_0_1=fixed,
        short_train=dict(epochs=2, lr=1e-3, init=0.1, history=history, final_alpha=alpha),
        alpha_2ep=learned,
        seconds_total=time.time() - t0,
        note='Networks frozen. Fixed alpha=0.1 scored on Official Test. Scalar alpha then trained 2 epochs from 0.1 on train625 with nano v2.',
    )
    path = os.path.join(OUT, 'summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()
