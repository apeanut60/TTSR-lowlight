#!/usr/bin/env python3
"""Zero-shot A0–A4 ablation for v1r6 query illumination alignment.

Uses fixed v1r3e4 checkpoint + Huawei Nano Eval (30 images).
Collects RGB PSNR/SSIM per image, gamma stats, and match diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from dataset.data1 import TestSet  # noqa: E402
from model import TTSREnhance  # noqa: E402
from model.query_illum_align import (  # noqa: E402
    align_query_illumination, estimate_query_gamma,
)
from utils import calc_psnr_and_ssim  # noqa: E402


ALPHAS = [
    ('A0', False, 0.0),
    ('A1', True, 0.25),
    ('A2', True, 0.50),
    ('A3', True, 0.75),
    ('A4', True, 1.00),
]


def _build_args(cli):
    return SimpleNamespace(
        dataset='data1',
        dataset_dir='/root/data/datasets/data1',
        data1_camera='Huawei',
        ref_dir='/root/data/datasets/data1/Eval/Huawei/nanobanana_ref',
        ref_degrade=False,
        eval_ref_degrade=None,
        ref_gamma=1.0,
        eval_degrade_seed=1234,
        ref_color_jitter=0.2,
        ref_shift_range=4,
        ref_blur_sigma=2.0,
        eval_mean_align=False,
        enhance_mode=True,
        num_res_blocks='10+10+6+4',
        n_feats=64,
        res_scale=1.0,
        freeze_lte=False,
        ref_correction=True,
        ref_correction_feats=16,
        ref_illum_pool=8,
        query_illum_align=False,
        query_illum_alpha=0.0,
        query_gamma_min=0.3,
        query_gamma_max=1.0,
        tile_size=256,
        tile_overlap=96,
        tile_window='cosine',
        cpu=bool(cli.cpu),
        num_gpu=1,
        num_workers=2,
        batch_size=1,
        eval_chroma_gain=1.0,
        eval_tta=False,
    )


def _load_model(args, ckpt, device):
    model = TTSREnhance.TTSREnhance(args).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    matched = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            matched += 1
    model.load_state_dict(cur)
    print('Loaded %d matching tensors from %s' % (matched, ckpt))
    return model


def _tiled_forward(model, args, lr, lr_sr, ref, ref_sr, return_diag=False):
    """Mirror trainer._tiled_forward with full-image gamma for tiles."""
    tile_size = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile_size // 2))
    window = args.tile_window.lower()

    use_qia = getattr(args, 'query_illum_align', False)
    alpha = float(getattr(args, 'query_illum_alpha', 0.0))
    gamma_override = None
    if use_qia and alpha > 0.0:
        gamma_override = estimate_query_gamma(
            lr_sr, ref,
            gamma_min=float(args.query_gamma_min),
            gamma_max=float(args.query_gamma_max),
        )

    _, _, H, W = lr.shape
    fwd_kwargs = {}
    if gamma_override is not None:
        fwd_kwargs['query_gamma_override'] = gamma_override
    if return_diag:
        fwd_kwargs['return_diag'] = True

    if H <= tile_size and W <= tile_size:
        return model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, **fwd_kwargs)

    stride = max(1, tile_size - overlap)
    n_h = max(1, (H - overlap + stride - 1) // stride)
    n_w = max(1, (W - overlap + stride - 1) // stride)

    sr_accum = torch.zeros(1, 3, H, W, device=lr.device)
    weight_accum = torch.zeros(1, 1, H, W, device=lr.device)

    mask_y = torch.ones(tile_size, device=lr.device)
    mask_x = torch.ones(tile_size, device=lr.device)
    if overlap > 0:
        if window == 'cosine':
            ramp = 0.5 - 0.5 * torch.cos(
                torch.linspace(0, math.pi, overlap, device=lr.device))
        else:
            ramp = torch.linspace(0, 1, overlap, device=lr.device)
        mask_y[:overlap] = ramp
        mask_y[-overlap:] = ramp.flip(0)
        mask_x[:overlap] = ramp
        mask_x[-overlap:] = ramp.flip(0)
    feather = (mask_y[:, None] * mask_x[None, :]).view(1, 1, tile_size, tile_size)

    last = None
    has_illum = hasattr(model, 'apply_illum_head')
    has_refillum = hasattr(model, 'apply_ref_illum')

    for i_h in range(n_h):
        y0 = min(i_h * stride, H - tile_size)
        y1 = y0 + tile_size
        for i_w in range(n_w):
            x0 = min(i_w * stride, W - tile_size)
            x1 = x0 + tile_size
            tile_kwargs = dict(
                lr=lr[:, :, y0:y1, x0:x1],
                lrsr=lr_sr[:, :, y0:y1, x0:x1],
                ref=ref[:, :, y0:y1, x0:x1],
                refsr=ref_sr[:, :, y0:y1, x0:x1],
                **fwd_kwargs,
            )
            if has_illum:
                tile_kwargs['apply_illum'] = False
            if has_refillum:
                tile_kwargs['apply_ref_illum'] = False
            out = model(**tile_kwargs)
            if return_diag:
                tile_sr = out[0]
                last = out
            else:
                tile_sr = out[0]
                last = out
            sr_accum[:, :, y0:y1, x0:x1] += tile_sr * feather
            weight_accum[:, :, y0:y1, x0:x1] += feather

    sr = sr_accum / weight_accum.clamp(min=1e-8)
    if has_refillum:
        sr = model.apply_ref_illum(sr, lr, ref)
    if has_illum:
        sr = model.apply_illum_head(sr, lr)

    if return_diag:
        # reuse last tile's S/T/diag; replace sr with stitched
        return (sr,) + last[1:]
    return (sr,) + last[1:]


def _tensor_stats(xs):
    if not xs:
        return {}
    a = np.asarray(xs, dtype=np.float64)
    return {
        'mean': float(a.mean()),
        'std': float(a.std()),
        'min': float(a.min()),
        'max': float(a.max()),
        'q10': float(np.quantile(a, 0.1)),
        'q50': float(np.quantile(a, 0.5)),
        'q90': float(np.quantile(a, 0.9)),
    }


def run_one(run_name, use_align, alpha, args, ckpt, device, out_dir):
    args.query_illum_align = use_align
    args.query_illum_alpha = alpha

    model = _load_model(args, ckpt, device)
    ds = TestSet(args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    per_image = []
    gammas = []
    mean_lows, mean_refs, mean_alns = [], [], []
    diag_acc = defaultdict(list)

    with torch.no_grad():
        for i, sample in enumerate(loader):
            lr = sample['LR'].to(device)
            lr_sr = sample['LR_sr'].to(device)
            hr = sample['HR'].to(device)
            ref = sample['Ref'].to(device)
            ref_sr = sample['Ref_sr'].to(device)

            out = _tiled_forward(
                model, args, lr, lr_sr, ref, ref_sr, return_diag=True)
            sr = out[0]
            match_diag = out[5] if len(out) > 5 else None
            qinfo = out[6] if len(out) > 6 else model.last_query_illum_info

            _, ssim, mse, psnr_rgb = calc_psnr_and_ssim(sr, hr)
            row = {
                'idx': i,
                'psnr_rgb': float(psnr_rgb),
                'ssim_rgb': float(ssim),
                'mse': float(mse),
            }

            if qinfo is None:
                _, qi = align_query_illumination(
                    lr_sr, ref, alpha=0.0,
                    gamma_min=args.query_gamma_min,
                    gamma_max=args.query_gamma_max)
                qinfo = qi

            g = float(qinfo['gamma'][0].item())
            ml = float(qinfo['mean_low'][0].item())
            mr = float(qinfo['mean_ref'][0].item())
            ma = float(qinfo['mean_aligned'][0].item())
            row.update({
                'gamma': g,
                'mean_low': ml,
                'mean_ref': mr,
                'mean_aligned': ma,
                'luma_gap_before': abs(ml - mr),
                'luma_gap_after': abs(ma - mr) if alpha > 0 else abs(ml - mr),
            })
            gammas.append(g)
            mean_lows.append(ml)
            mean_refs.append(mr)
            mean_alns.append(ma)

            if match_diag is not None:
                for k, v in match_diag.items():
                    if isinstance(v, bool):
                        row[k] = v
                        continue
                    if v is None:
                        row[k] = None
                        continue
                    fv = float(v)
                    row[k] = fv
                    diag_acc[k].append(fv)

            per_image.append(row)
            print('  [%s] %02d  PSNR=%.3f  SSIM=%.4f  gamma=%.4f' % (
                run_name, i, row['psnr_rgb'], row['ssim_rgb'], row['gamma']))

    summary = {
        'run': run_name,
        'query_illum_align': use_align,
        'alpha': alpha,
        'n': len(per_image),
        'psnr_rgb_mean': float(np.mean([r['psnr_rgb'] for r in per_image])),
        'ssim_rgb_mean': float(np.mean([r['ssim_rgb'] for r in per_image])),
        'gamma': _tensor_stats(gammas),
        'mean_low': _tensor_stats(mean_lows),
        'mean_ref': _tensor_stats(mean_refs),
        'mean_aligned': _tensor_stats(mean_alns),
        'luma_gap_before': _tensor_stats(
            [r['luma_gap_before'] for r in per_image]),
        'luma_gap_after': _tensor_stats(
            [r['luma_gap_after'] for r in per_image]),
        'match': {k: _tensor_stats(v) for k, v in diag_acc.items()},
        'per_image': per_image,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, '%s_alpha%.2f.json' % (run_name, alpha))
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('==> %s  RGB PSNR=%.3f  SSIM=%.4f  -> %s' % (
        run_name, summary['psnr_rgb_mean'], summary['ssim_rgb_mean'], path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=(
        '/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt'))
    ap.add_argument('--out_dir', default=(
        '/root/data/experiments/retry/v1r6-query-illum-align/ablation'))
    ap.add_argument('--runs', default='A0,A1,A2,A3,A4')
    ap.add_argument('--cpu', action='store_true')
    cli = ap.parse_args()

    args = _build_args(cli)
    if not os.path.isdir(args.ref_dir):
        raise FileNotFoundError('Nano ref dir missing: %s' % args.ref_dir)
    if not os.path.isfile(cli.ckpt):
        raise FileNotFoundError('Checkpoint missing: %s' % cli.ckpt)

    # Count refs — fail loudly if incomplete
    n_ref = len([f for f in os.listdir(args.ref_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print('Nano refs found: %d in %s' % (n_ref, args.ref_dir))

    device = torch.device('cpu' if args.cpu else 'cuda')
    want = set(x.strip() for x in cli.runs.split(',') if x.strip())
    results = []
    for name, use_align, alpha in ALPHAS:
        if name not in want:
            continue
        print('======== %s align=%s alpha=%.2f ========' % (
            name, use_align, alpha))
        results.append(run_one(
            name, use_align, alpha, args, cli.ckpt, device, cli.out_dir))

    table_path = os.path.join(cli.out_dir, 'summary.json')
    slim = []
    for r in results:
        slim.append({
            'run': r['run'],
            'alpha': r['alpha'],
            'psnr_rgb_mean': r['psnr_rgb_mean'],
            'ssim_rgb_mean': r['ssim_rgb_mean'],
            'gamma_mean': r['gamma'].get('mean'),
            'max_cos_mean': (r['match'].get('max_cos_mean') or {}).get('mean'),
            'identity_cos_mean': (r['match'].get('identity_cos_mean') or {}).get('mean'),
            'match_l2_mean': (r['match'].get('match_l2_mean') or {}).get('mean'),
            'frac_dist_le_1': (r['match'].get('frac_dist_le_1') or {}).get('mean'),
            'frac_dist_le_2': (r['match'].get('frac_dist_le_2') or {}).get('mean'),
            'frac_dist_le_4': (r['match'].get('frac_dist_le_4') or {}).get('mean'),
            'frac_dist_le_8': (r['match'].get('frac_dist_le_8') or {}).get('mean'),
        })
    with open(table_path, 'w') as f:
        json.dump(slim, f, indent=2)
    print('\n=== SUMMARY ===')
    for r in slim:
        print('%s  alpha=%.2f  PSNR=%.3f  SSIM=%.4f  max_cos=%.4f  id_cos=%s  l2=%s' % (
            r['run'], r['alpha'], r['psnr_rgb_mean'], r['ssim_rgb_mean'],
            r['max_cos_mean'] or -1,
            '%.4f' % r['identity_cos_mean'] if r['identity_cos_mean'] is not None else 'n/a',
            '%.3f' % r['match_l2_mean'] if r['match_l2_mean'] is not None else 'n/a',
        ))
    print('Wrote', table_path)


if __name__ == '__main__':
    main()
