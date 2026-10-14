#!/usr/bin/env python3
"""Formal Huawei Nano eval for all v1r6-FT C0/C1 epoch checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'eval_v1r6_query_illum',
    os.path.join(ROOT, 'scripts', 'eval_v1r6_query_illum.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_build_args = _mod._build_args
run_one = _mod.run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c0_dir', default='/root/data/experiments/retry/v1r6-ft-C0-control')
    ap.add_argument('--c1_dir', default='/root/data/experiments/retry/v1r6-ft-C1-gamma075')
    ap.add_argument('--out_dir', default='/root/data/experiments/retry/v1r6-ft-compare')
    ap.add_argument('--epochs', default='1,2,3,4,5')
    cli = ap.parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    # Patch ALPHAS usage: call run_one directly with align settings
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = _build_args(argparse.Namespace(cpu=False))

    rows = []
    # Epoch 0 = zero-shot baselines (from prior ablation if present)
    zshot = '/root/data/experiments/retry/v1r6-query-illum-align/ablation/summary.json'
    if os.path.isfile(zshot):
        with open(zshot) as f:
            slim = json.load(f)
        a0 = next(x for x in slim if x['run'] == 'A0')
        a3 = next(x for x in slim if x['run'] == 'A3')
        rows.append({
            'epoch': 0,
            'C0_psnr': a0['psnr_rgb_mean'], 'C0_ssim': a0['ssim_rgb_mean'],
            'C1_psnr': a3['psnr_rgb_mean'], 'C1_ssim': a3['ssim_rgb_mean'],
            'C1_minus_C0': a3['psnr_rgb_mean'] - a0['psnr_rgb_mean'],
            'note': 'zero-shot from v1r6 ablation',
        })

    for ep in [int(x) for x in cli.epochs.split(',') if x.strip()]:
        tag = 'model_%05d.pt' % ep
        c0 = os.path.join(cli.c0_dir, 'model', tag)
        c1 = os.path.join(cli.c1_dir, 'model', tag)
        if not os.path.isfile(c0) or not os.path.isfile(c1):
            print('skip epoch %d missing ckpt' % ep)
            continue
        print('==== epoch %d C0 ====' % ep)
        r0 = run_one('C0e%d' % ep, False, 0.0, args, c0, device,
                     os.path.join(cli.out_dir, 'C0'))
        print('==== epoch %d C1 ====' % ep)
        r1 = run_one('C1e%d' % ep, True, 0.75, args, c1, device,
                     os.path.join(cli.out_dir, 'C1'))
        rows.append({
            'epoch': ep,
            'C0_psnr': r0['psnr_rgb_mean'], 'C0_ssim': r0['ssim_rgb_mean'],
            'C1_psnr': r1['psnr_rgb_mean'], 'C1_ssim': r1['ssim_rgb_mean'],
            'C1_minus_C0': r1['psnr_rgb_mean'] - r0['psnr_rgb_mean'],
            'C0_max_cos': (r0['match'].get('max_cos_mean') or {}).get('mean'),
            'C1_max_cos': (r1['match'].get('max_cos_mean') or {}).get('mean'),
            'C0_match_l2': (r0['match'].get('match_l2_mean') or {}).get('mean'),
            'C1_match_l2': (r1['match'].get('match_l2_mean') or {}).get('mean'),
            'C0_ckpt': c0,
            'C1_ckpt': c1,
            'C0_per_image': r0['per_image'],
            'C1_per_image': r1['per_image'],
        })

    # Per-image C1-C0 at best same-epoch and at last epoch
    summary = {'table': [{k: v for k, v in r.items()
                          if k not in ('C0_per_image', 'C1_per_image')}
                         for r in rows]}
    # Same-epoch image diffs for each FT epoch
    img_diffs = {}
    for r in rows:
        if r['epoch'] == 0 or 'C0_per_image' not in r:
            continue
        diffs = []
        for a, b in zip(r['C0_per_image'], r['C1_per_image']):
            diffs.append({
                'idx': a['idx'],
                'C0': a['psnr_rgb'],
                'C1': b['psnr_rgb'],
                'delta': b['psnr_rgb'] - a['psnr_rgb'],
            })
        n_up = sum(1 for d in diffs if d['delta'] > 0.01)
        n_dn = sum(1 for d in diffs if d['delta'] < -0.01)
        n_flat = len(diffs) - n_up - n_dn
        img_diffs[str(r['epoch'])] = {
            'n_up': n_up, 'n_down': n_dn, 'n_flat': n_flat,
            'mean_delta': float(sum(d['delta'] for d in diffs) / len(diffs)),
            'per_image': diffs,
        }
    summary['image_diffs_by_epoch'] = img_diffs

    out = os.path.join(cli.out_dir, 'ft_compare_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== FT COMPARE ===')
    for r in summary['table']:
        print('e%d  C0=%.3f/%.4f  C1=%.3f/%.4f  C1-C0=%+.3f' % (
            r['epoch'], r['C0_psnr'], r['C0_ssim'],
            r['C1_psnr'], r['C1_ssim'], r['C1_minus_C0']))
    print('Wrote', out)


if __name__ == '__main__':
    main()
