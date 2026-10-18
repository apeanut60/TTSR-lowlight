#!/usr/bin/env python
"""LLFormer + P2/H3 reference-driven residual fusion on LOLv2-Real Test.

Y_final = clip(Y_L + 0.1 * (Y_N - Y_S), 0, 1)

Self protocol (v2 H2): P2 Y0 is computed with the Nano reference; H3 Self
uses that Y0 as the reference, H3 Nano uses R_nano. Same weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from imageio import imsave, imread

SOTA = '/root/projects/TTSR-lowlight-v1-data1SOTA'
LL_ROOT = '/root/projects/TTSR-lowlight'
sys.path.insert(0, LL_ROOT)
from model.LLFormerBridge import infer_rgb, load_llformer_bridge  # noqa: E402

# Drop the other repo's model package so SOTA imports resolve locally.
for key in list(sys.modules):
    if key == 'model' or key.startswith('model.'):
        del sys.modules[key]
sys.path.remove(LL_ROOT)
sys.path.insert(0, SOTA)
sys.path.insert(0, os.path.join(SOTA, 'scripts'))

from local_refine_runtime import sha256  # noqa: E402
from train_nikon_domain import h3_forward, load_h3_from, load_p2_from  # noqa: E402
from utils import calc_psnr_and_ssim  # noqa: E402
from v2_p2_runtime import tiled_p2  # noqa: E402

ALPHA = 0.1
P2_CKPT = (
    '/root/data/experiments/TTSR-lowlight-lolv2real/'
    'A_illum_only_from_huawei/model/model_00001.pt')
H3_CKPT = (
    '/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
    'H3_v3b0_s42/checkpoint_02000.pt')
LL_CKPT = (
    '/root/data/experiments/llformer_lolv2real/canonical/'
    'llformer_lolv2real_base_E110.pth')
DATA = '/root/data/datasets/lol-v2-real'
NANO_DIR = os.path.join(DATA, 'Test/nanobanana_ref_v3')
LOW_DIR = os.path.join(DATA, 'Test/Low')
GT_DIR = os.path.join(DATA, 'Test/Normal')
OUT = '/root/data/experiments/TTSR-LLFormer-Fusion'


def sha256_file(path):
    return sha256(path)


def to_01_from_m11(t):
    return (t + 1.0) * 0.5


def to_m11_from_01(t):
    return t * 2.0 - 1.0


def read_m11(path, device, size_hw=None):
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    im = im[..., :3].astype(np.float32)
    t = torch.from_numpy(im).permute(2, 0, 1)[None].to(device) / 127.5 - 1.0
    if size_hw is not None and tuple(t.shape[-2:]) != tuple(size_hw):
        t = torch.nn.functional.interpolate(
            t, size=size_hw, mode='bilinear', align_corners=False)
    return t


def rgb_metrics(pred01, gt_m11):
    """Formal trainer protocol: calc_psnr_and_ssim on [-1, 1]."""
    pred = to_m11_from_01(pred01.clamp(0, 1))
    psnr_y, ssim, mse, psnr_rgb = calc_psnr_and_ssim(pred, gt_m11)
    return float(psnr_rgb), float(ssim)


def fuse(yl, yn, ys, alpha):
    return torch.clamp(yl + alpha * (yn - ys), 0.0, 1.0)


def unit_alpha0():
    yl = torch.rand(1, 3, 8, 8)
    yn = torch.rand(1, 3, 8, 8)
    ys = torch.rand(1, 3, 8, 8)
    out = fuse(yl, yn, ys, 0.0)
    if not torch.equal(out, yl.clamp(0, 1)):
        raise SystemExit('alpha=0 did not reduce to LLFormer output')
    print('alpha=0 identity check OK')


@torch.no_grad()
def ttsr_pair(p2, args, enc, mat, head, low, nano):
    """Return YN, YS, Y0 in [0, 1]. Self ref = Y0 (H2 protocol)."""
    y0 = tiled_p2(p2, args, low, low, nano, nano)
    yn, _ = h3_forward(enc, mat, head, y0, nano)
    ys, _ = h3_forward(enc, mat, head, y0, y0)
    return to_01_from_m11(yn), to_01_from_m11(ys), to_01_from_m11(y0)


def main():
    unit_alpha0()
    os.makedirs(os.path.join(OUT, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
    log_path = os.path.join(OUT, 'logs', 'inference.log')
    log = open(log_path, 'w')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    say('loading LLFormer %s' % LL_CKPT)
    bridge, info = load_llformer_bridge(LL_CKPT, device=device, train=False)
    say('LLFormer tensors %s' % info.get('loaded', info))
    say('loading P2 %s' % P2_CKPT)
    p2, args = load_p2_from(P2_CKPT, device, joint=False)
    p2.eval()
    say('loading H3 %s' % H3_CKPT)
    enc, mat, head = load_h3_from(H3_CKPT, device, trainable=False)

    lows = sorted(f for f in os.listdir(LOW_DIR) if f.endswith('.png') and not f.startswith('.'))
    rows = []
    t_all = time.time()
    for i, name in enumerate(lows):
        t0 = time.time()
        low_path = os.path.join(LOW_DIR, name)
        gt_name = name.replace('low', 'normal')
        nano_path = os.path.join(NANO_DIR, name)
        gt_path = os.path.join(GT_DIR, gt_name)
        if not os.path.isfile(nano_path) or not os.path.isfile(gt_path):
            raise SystemExit('missing pair for %s' % name)
        low = read_m11(low_path, device)
        nano = read_m11(nano_path, device, size_hw=low.shape[-2:])
        gt = read_m11(gt_path, device, size_hw=low.shape[-2:])
        x01 = to_01_from_m11(low)
        yl = infer_rgb(bridge, x01, clamp=True)
        if yl.shape[-2:] != low.shape[-2:]:
            yl = torch.nn.functional.interpolate(
                yl, size=low.shape[-2:], mode='bilinear', align_corners=False)
        yn, ys, _y0 = ttsr_pair(p2, args, enc, mat, head, low, nano)
        if not (torch.isfinite(yn).all() and torch.isfinite(ys).all() and torch.isfinite(yl).all()):
            raise SystemExit('non-finite output at %s' % name)
        delta = yn - ys
        out = fuse(yl, yn, ys, ALPHA)
        if out.shape != yl.shape or out.min() < -1e-6 or out.max() > 1 + 1e-5:
            raise SystemExit('bad range/shape at %s' % name)
        png = (out.squeeze(0).clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        out_name = 'image_%03d.png' % (i + 1)
        imsave(os.path.join(OUT, 'outputs', out_name), png)
        psnr, ssim = rgb_metrics(out, gt)
        rec = dict(
            index=i + 1, filename=out_name, low_name=name,
            psnr_rgb=psnr, ssim_rgb=ssim,
            mean_abs_delta=float(delta.abs().mean()),
            seconds=time.time() - t0,
        )
        rows.append(rec)
        say('[%03d] %s PSNR=%.3f SSIM=%.4f |d|=%.5f' % (
            i + 1, name, psnr, ssim, rec['mean_abs_delta']))

    elapsed = time.time() - t_all
    summary = dict(
        method='LLFormer + TTSR-RDRF',
        fusion='YL + 0.1 * (YN - YS)',
        alpha=ALPHA,
        dataset='LOLv2-Real',
        split='Official Test (Test/Low + Test/Normal + Test/nanobanana_ref_v3)',
        n_images=len(rows),
        psnr_rgb=float(np.mean([r['psnr_rgb'] for r in rows])),
        ssim_rgb=float(np.mean([r['ssim_rgb'] for r in rows])),
        mean_abs_delta_ref=float(np.mean([r['mean_abs_delta'] for r in rows])),
        seconds_total=elapsed,
        seconds_per_image=elapsed / max(len(rows), 1),
        self_protocol='H2: Y0=P2(X, R_nano); YN=Y0+H3(Y0, R_nano); YS=Y0+H3(Y0, Y0)',
        value_range='all branches RGB [0,1] before fusion; metrics via calc_psnr_and_ssim',
        p2_ckpt=P2_CKPT,
        p2_sha256=sha256_file(P2_CKPT),
        p2_note='LOLv2 illum-head FT epoch 1 from v1r3-Huawei; best nano score in that run (20.768)',
        h3_ckpt=H3_CKPT,
        h3_sha256=sha256_file(H3_CKPT),
        h3_note='Huawei H3@2000; H3 was not retrained in the LOLv2 round',
        llformer_ckpt=LL_CKPT,
        llformer_sha256=sha256_file(LL_CKPT),
        p2_config='M1 texture fusion + lv2 branch_bypass (v2_p2_runtime.P2_CONFIG)',
        no_mean_align=True,
        no_gt_in_fusion=True,
    )
    with open(os.path.join(OUT, 'metrics', 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, 'metrics', 'per_image.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    say('')
    say('Method: LLFormer + TTSR-RDRF')
    say('Fusion: YL + 0.1 * (YN - YS)')
    say('Dataset: LOLv2-Real Official Test')
    say('Images: %d' % summary['n_images'])
    say('PSNR: %.4f' % summary['psnr_rgb'])
    say('SSIM: %.4f' % summary['ssim_rgb'])
    say('mean|delta_ref|: %.6f' % summary['mean_abs_delta_ref'])
    say('seconds: %.1f' % elapsed)
    log.close()


if __name__ == '__main__':
    main()
