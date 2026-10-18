#!/usr/bin/env python
"""Two-epoch joint FT of P2+H3 with fixed fusion alpha.

Y = clip(YL + 0.1 * (YN - YS), 0, 1)
Alpha is a constant. LLFormer stays frozen. LTE stays frozen.
Loss is RGB MSE on the fused crop. Official Test is scored in RGB
before training and after each epoch.
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

from dataset.data1_localrefine import read_rgb_model_tensor  # noqa: E402
from eval_nikon_export import channel_metrics, to_uint8  # noqa: E402
from infer_llformer_ttsr_fusion import (  # noqa: E402
    GT_DIR, LL_CKPT, LOW_DIR, NANO_DIR,
    fuse, infer_rgb, load_llformer_bridge, read_m11, to_01_from_m11, ttsr_pair,
)
from local_refine_runtime import read_manifest_csv, sha256  # noqa: E402
from train_nikon_domain import load_h3_from, load_p2_from  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    configure_p2, h3_forward, p2_forward,
)

ALPHA = 0.1
CKPT = (
    '/root/data/experiments/TTSR-lowlight-lolv2real/checkpoints/'
    'H3_frozenP2_mse_s42/checkpoint_00250.pt')
MAN = '/root/data/experiments/TTSR-lowlight-lolv2real/manifests_h3/manifests_train.csv'
OUT = '/root/data/experiments/TTSR-LLFormer-Fusion/joint_a01_e2'
EPOCHS = 2
CROP = 192
P2_LR = 1e-6
ENC_LR = 1e-6
MATCH_LR = 2e-6
HEAD_LR = 1e-5


def load_crop(row, rng, device):
    low = read_rgb_model_tensor(row['low_path'])
    high = read_rgb_model_tensor(row['high_path'])
    h, w = low.shape[-2:]
    if tuple(high.shape[-2:]) != (h, w):
        high = F.interpolate(high[None], size=(h, w), mode='bilinear', align_corners=False)[0]
    nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
    top = int(rng.randint(0, h - CROP + 1))
    left = int(rng.randint(0, w - CROP + 1))
    sl = (slice(None), slice(top, top + CROP), slice(left, left + CROP))
    return (
        low[sl][None].to(device),
        high[sl][None].to(device),
        nano[sl][None].to(device),
    )


def fused_crop(bridge, p2, enc, mat, head, low, nano):
    y0 = p2_forward(p2, low, nano, joint=True)
    yn, _ = h3_forward(enc, mat, head, y0, nano)
    ys, _ = h3_forward(enc, mat, head, y0, y0)
    with torch.no_grad():
        yl = infer_rgb(bridge, to_01_from_m11(low), clamp=True)
    return torch.clamp(yl + ALPHA * (to_01_from_m11(yn) - to_01_from_m11(ys)), 0.0, 1.0)


@torch.no_grad()
def eval_official(bridge, p2, args, enc, mat, head, device):
    p2.eval()
    enc.eval()
    mat.eval()
    head.eval()
    lows = sorted(f for f in os.listdir(LOW_DIR) if f.endswith('.png') and not f.startswith('.'))
    rows = []
    for name in lows:
        low = read_m11(os.path.join(LOW_DIR, name), device)
        nano = read_m11(os.path.join(NANO_DIR, name), device, size_hw=low.shape[-2:])
        gt = read_m11(os.path.join(GT_DIR, name.replace('low', 'normal')), device, size_hw=low.shape[-2:])
        yl = infer_rgb(bridge, to_01_from_m11(low), clamp=True)
        if yl.shape[-2:] != low.shape[-2:]:
            yl = F.interpolate(yl, size=low.shape[-2:], mode='bilinear', align_corners=False)
        yn, ys, _ = ttsr_pair(p2, args, enc, mat, head, low, nano)
        pred = fuse(yl, yn, ys, ALPHA)
        m = channel_metrics(to_uint8(pred * 2 - 1), to_uint8(gt))
        rows.append(m)
    keys = ['psnr_rgb', 'ssim_rgb', 'psnr_R', 'psnr_G', 'psnr_B', 'ssim_R', 'ssim_G', 'ssim_B']
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    mean['n'] = len(rows)
    configure_p2(p2, joint=True)
    for m in (enc, mat, head):
        m.train()
    return mean


def save_ckpt(path, p2, enc, mat, head, epoch, scores):
    torch.save(dict(
        p2=p2.state_dict(),
        encoder=enc.state_dict(),
        match=mat.state_dict(),
        head=head.state_dict(),
        epoch=epoch,
        alpha=ALPHA,
        scores=scores,
        p2_init=CKPT,
        h3_init=CKPT,
    ), path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.join(OUT, 'checkpoints'), exist_ok=True)
    rng = np.random.RandomState(42)
    torch.manual_seed(42)
    rows = read_manifest_csv(MAN)
    bridge, _ = load_llformer_bridge(LL_CKPT, device=device, train=False)
    for p in bridge.parameters():
        p.requires_grad_(False)
    bridge.eval()
    p2, args = load_p2_from(CKPT, device, joint=True)
    enc, mat, head = load_h3_from(CKPT, device, trainable=True)
    groups = [
        dict(params=[p for p in p2.parameters() if p.requires_grad], lr=P2_LR, name='p2'),
        dict(params=list(enc.parameters()), lr=ENC_LR, name='encoder'),
        dict(params=list(mat.parameters()), lr=MATCH_LR, name='matcher'),
        dict(params=list(head.parameters()), lr=HEAD_LR, name='head'),
    ]
    n_train = {g['name']: sum(p.numel() for p in g['params']) for g in groups}
    print('trainable', n_train, 'alpha', ALPHA, 'epochs', EPOCHS, 'n', len(rows), flush=True)
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    history = []
    t0 = time.time()
    print('--- eval epoch 0', flush=True)
    s0 = eval_official(bridge, p2, args, enc, mat, head, device)
    print('  epoch 0 PSNR_RGB=%.4f SSIM_RGB=%.4f R/G/B=%.3f/%.3f/%.3f' % (
        s0['psnr_rgb'], s0['ssim_rgb'], s0['psnr_R'], s0['psnr_G'], s0['psnr_B']), flush=True)
    history.append(dict(epoch=0, **s0))
    save_ckpt(os.path.join(OUT, 'checkpoints', 'epoch_000.pt'), p2, enc, mat, head, 0, s0)

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(rows))
        running = 0.0
        for step, idx in enumerate(order, start=1):
            low, gt, nano = load_crop(rows[int(idx)], rng, device)
            pred = fused_crop(bridge, p2, enc, mat, head, low, nano)
            loss = F.mse_loss(pred, to_01_from_m11(gt))
            if not torch.isfinite(loss):
                raise SystemExit('non-finite at epoch %d step %d' % (epoch, step))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            if step == 1 or step % 50 == 0:
                print('  epoch %d step %d/%d loss=%.5f' % (
                    epoch, step, len(rows), running / step), flush=True)
        print('--- eval epoch %d' % epoch, flush=True)
        sc = eval_official(bridge, p2, args, enc, mat, head, device)
        print('  epoch %d PSNR_RGB=%.4f SSIM_RGB=%.4f R/G/B=%.3f/%.3f/%.3f' % (
            epoch, sc['psnr_rgb'], sc['ssim_rgb'], sc['psnr_R'], sc['psnr_G'], sc['psnr_B']), flush=True)
        history.append(dict(epoch=epoch, train_mse=running / len(rows), **sc))
        save_ckpt(os.path.join(OUT, 'checkpoints', 'epoch_%03d.pt' % epoch), p2, enc, mat, head, epoch, sc)

    summary = dict(
        alpha=ALPHA,
        epochs=EPOCHS,
        crop=CROP,
        n_train=len(rows),
        trainable=n_train,
        lr=dict(p2=P2_LR, encoder=ENC_LR, matcher=MATCH_LR, head=HEAD_LR),
        init_ckpt=CKPT,
        init_sha256=sha256(CKPT),
        llformer_ckpt=LL_CKPT,
        metric='RGB calc_psnr_rgb; SSIM mean of R/G/B',
        frozen='LLFormer, LTE, fusion alpha',
        history=history,
        seconds_total=time.time() - t0,
    )
    path = os.path.join(OUT, 'summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('wrote', path)
    print('=== done %.1f min ===' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
