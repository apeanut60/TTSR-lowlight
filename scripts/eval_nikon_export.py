#!/usr/bin/env python
"""Nikon Eval20: zero-shot baselines and formal export with RGB + R/G/B metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from imageio import imsave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import read_manifest_csv, sha256, _read_eval_tensor  # noqa: E402
from train_nikon_domain import (  # noqa: E402
    DEFAULT_H3, DEFAULT_P2, EXP, h3_forward, load_h3_from, load_p2_from,
)
from utils import calc_psnr, calc_psnr_rgb, calc_ssim, calc_ssim_y, _ssim_map_mean  # noqa: E402
from v2_p2_runtime import tiled_p2  # noqa: E402

A2_HUAWEI = ('/root/data/experiments/TTSR-lowlight-v24/checkpoints/'
             'A2_jointP2_mse_s42/checkpoint_00250.pt')


def to_uint8(t):
    a = (t.detach() + 1.) * 127.5
    a = a.squeeze(0).clamp(0, 255).round().cpu().numpy().transpose(1, 2, 0)
    return a.astype(np.uint8)


def channel_metrics(pred_u8, gt_u8):
    psnr_rgb = float(calc_psnr_rgb(pred_u8, gt_u8))
    psnr_y = float(calc_psnr(pred_u8, gt_u8))
    ssim_rgb = float(calc_ssim(pred_u8, gt_u8))
    ssim_y = float(calc_ssim_y(pred_u8, gt_u8))
    psnr_c, ssim_c = {}, {}
    for i, n in enumerate(('R', 'G', 'B')):
        d = (pred_u8[:, :, i].astype(np.float64) - gt_u8[:, :, i].astype(np.float64)) / 255.0
        mse = float(np.mean(d ** 2))
        psnr_c[n] = float('inf') if mse <= 0 else -10.0 * math.log10(mse)
        ssim_c[n] = float(_ssim_map_mean(pred_u8[:, :, i], gt_u8[:, :, i]))
    mse = float(np.mean((pred_u8.astype(np.float64) - gt_u8.astype(np.float64)) ** 2))
    return dict(
        psnr_rgb=psnr_rgb, psnr_y=psnr_y, ssim_rgb=ssim_rgb, ssim_y=ssim_y,
        psnr_R=psnr_c['R'], psnr_G=psnr_c['G'], psnr_B=psnr_c['B'],
        ssim_R=ssim_c['R'], ssim_G=ssim_c['G'], ssim_B=ssim_c['B'], mse=mse,
    )


def mean_keys(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


@torch.no_grad()
def run_variant(name, p2, args, enc, mat, head, rows, device, out_dir, with_h3):
    save = os.path.join(out_dir, 'save_results')
    os.makedirs(save, exist_ok=True)
    for sub in ('GT', 'Low', 'P2'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    per = []
    for i, row in enumerate(rows):
        sid = row['sample_id']
        stem = os.path.splitext(os.path.basename(sid))[0]
        low = _read_eval_tensor(row['low_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
        gt = _read_eval_tensor(row['high_path'], device)
        y0 = tiled_p2(p2, args, low, low, nano, nano)
        if with_h3:
            y, _ = h3_forward(enc, mat, head, y0, nano)
        else:
            y = y0
        pred_u8 = to_uint8(y)
        gt_u8 = to_uint8(gt)
        y0_u8 = to_uint8(y0)
        low_u8 = to_uint8(low)
        imsave(os.path.join(save, '%s.png' % stem), pred_u8)
        imsave(os.path.join(out_dir, 'GT', '%s.png' % stem), gt_u8)
        imsave(os.path.join(out_dir, 'P2', '%s.png' % stem), y0_u8)
        imsave(os.path.join(out_dir, 'Low', '%s.png' % stem), low_u8)
        m = channel_metrics(pred_u8, gt_u8)
        m_p2 = channel_metrics(y0_u8, gt_u8)
        rec = dict(sample_id=sid, filename='%s.png' % stem, index=i, **m,
                   p2_psnr_rgb=m_p2['psnr_rgb'], p2_ssim_rgb=m_p2['ssim_rgb'])
        per.append(rec)
        print('[%s %02d] %s  PSNR=%.3f SSIM=%.4f R/G/B=%.2f/%.2f/%.2f' % (
            name, i, stem, m['psnr_rgb'], m['ssim_rgb'],
            m['psnr_R'], m['psnr_G'], m['psnr_B']), flush=True)

    keys = ['psnr_rgb', 'ssim_rgb', 'psnr_y', 'ssim_y',
            'psnr_R', 'psnr_G', 'psnr_B', 'ssim_R', 'ssim_G', 'ssim_B',
            'mse', 'p2_psnr_rgb', 'p2_ssim_rgb']
    summary = dict(variant=name, n=len(per), mean=mean_keys(per, keys), per_image=per)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'filename'] + keys)
        w.writeheader()
        for r in per:
            w.writerow({k: r[k] for k in w.fieldnames})
    s = summary['mean']
    txt = (
        'Nikon Eval20 — %s\n'
        'PSNR_RGB : %.4f\nSSIM_RGB : %.6f\n'
        'PSNR_R/G/B : %.4f / %.4f / %.4f\n'
        'SSIM_R/G/B : %.6f / %.6f / %.6f\n'
        'P2-only PSNR_RGB : %.4f  SSIM : %.6f\n'
        'save_results: %s\n' % (
            name, s['psnr_rgb'], s['ssim_rgb'],
            s['psnr_R'], s['psnr_G'], s['psnr_B'],
            s['ssim_R'], s['ssim_G'], s['ssim_B'],
            s['p2_psnr_rgb'], s['p2_ssim_rgb'], save)
    )
    open(os.path.join(out_dir, 'summary.txt'), 'w').write(txt)
    print(txt)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['p0', 'export'], required=True)
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--eval_root', default='/root/data/eval')
    ap.add_argument('--ckpt', default='',
                    help='For export: stage ckpt with p2[+h3]')
    ap.add_argument('--name', default='', help='export folder name under eval_root')
    ap.add_argument('--p2_only', action='store_true')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = read_manifest_csv(os.path.join(a.out_root, 'manifests/manifests_eval.csv'))
    assert len(rows) == 20

    if a.mode == 'p0':
        results = {}
        # P2-only
        p2, args = load_p2_from(DEFAULT_P2, device, joint=False)
        out = os.path.join(a.eval_root, 'nikon-p0-P2')
        results['P2'] = run_variant('P2', p2, args, None, None, None, rows, device, out, False)

        # H3 = Huawei P2 + H3
        enc, mat, head = load_h3_from(DEFAULT_H3, device, trainable=False)
        out = os.path.join(a.eval_root, 'nikon-p0-H3')
        results['H3'] = run_variant('H3', p2, args, enc, mat, head, rows, device, out, True)

        # A2@250 Huawei zero-shot
        ck = torch.load(A2_HUAWEI, map_location=device)
        p2a, argsa = load_p2_from(A2_HUAWEI, device, joint=False)
        enc.load_state_dict(ck['encoder']); mat.load_state_dict(ck['match'])
        head.load_state_dict(ck['head'])
        out = os.path.join(a.eval_root, 'nikon-p0-A2huawei250')
        results['A2huawei'] = run_variant(
            'A2huawei250', p2a, argsa, enc, mat, head, rows, device, out, True)

        summary = {
            'p2_init': DEFAULT_P2, 'p2_sha': sha256(DEFAULT_P2),
            'h3_init': DEFAULT_H3, 'h3_sha': sha256(DEFAULT_H3),
            'a2_init': A2_HUAWEI, 'a2_sha': sha256(A2_HUAWEI),
            'means': {k: v['mean'] for k, v in results.items()},
        }
        out_sum = os.path.join(a.out_root, 'p0_zeroshot_summary.json')
        with open(out_sum, 'w') as f:
            json.dump(summary, f, indent=2)
        print('P0 summary →', out_sum)
        for k, v in summary['means'].items():
            print('  %s  PSNR=%.4f  SSIM=%.4f  P2=%.4f' % (
                k, v['psnr_rgb'], v['ssim_rgb'], v['p2_psnr_rgb']))
        return

    # export one ckpt
    assert a.ckpt and a.name
    out = os.path.join(a.eval_root, a.name)
    p2, args = load_p2_from(a.ckpt, device, joint=False)
    with_h3 = not a.p2_only
    enc = mat = head = None
    if with_h3:
        enc, mat, head = load_h3_from(a.ckpt, device, trainable=False)
    run_variant(a.name, p2, args, enc, mat, head, rows, device, out, with_h3)


if __name__ == '__main__':
    main()
