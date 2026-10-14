#!/usr/bin/env python3
"""Zero-shot A0 / A3 / B2-25 / B2-50 for v1r7 gamma+hist query alignment."""
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
    align_query_illumination, estimate_query_align_maps,
)
from utils import calc_psnr_and_ssim  # noqa: E402


RUNS = [
    ('A0', False, 0.0, 0.0),
    ('A3', True, 0.75, 0.0),
    ('B2-25', True, 0.75, 0.25),
    ('B2-50', True, 0.75, 0.50),
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
        query_hist_beta=0.0,
        query_gamma_min=0.3,
        query_gamma_max=1.0,
        query_hist_bins=256,
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
    tile_size = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile_size // 2))
    window = args.tile_window.lower()
    use_qia = getattr(args, 'query_illum_align', False)
    alpha = float(getattr(args, 'query_illum_alpha', 0.0))
    beta = float(getattr(args, 'query_hist_beta', 0.0))

    fwd_kwargs = {}
    if use_qia and (alpha > 0.0 or beta > 0.0):
        g, lut, _ = estimate_query_align_maps(
            lr_sr, ref, alpha=alpha,
            gamma_min=float(args.query_gamma_min),
            gamma_max=float(args.query_gamma_max),
            bins=int(args.query_hist_bins),
        )
        fwd_kwargs['query_gamma_override'] = g
        if beta > 0.0:
            fwd_kwargs['query_hist_lut_override'] = lut
    if return_diag:
        fwd_kwargs['return_diag'] = True

    _, _, H, W = lr.shape
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
            tile_sr = out[0]
            last = out
            sr_accum[:, :, y0:y1, x0:x1] += tile_sr * feather
            weight_accum[:, :, y0:y1, x0:x1] += feather
    sr = sr_accum / weight_accum.clamp(min=1e-8)
    if has_refillum:
        sr = model.apply_ref_illum(sr, lr, ref)
    if has_illum:
        sr = model.apply_illum_head(sr, lr)
    return (sr,) + last[1:]


def _stats(xs):
    if not xs:
        return {}
    a = np.asarray(xs, dtype=np.float64)
    return {
        'mean': float(a.mean()), 'std': float(a.std()),
        'min': float(a.min()), 'max': float(a.max()),
        'q10': float(np.quantile(a, 0.1)),
        'q50': float(np.quantile(a, 0.5)),
        'q90': float(np.quantile(a, 0.9)),
    }


def run_one(run_name, use_align, alpha, beta, args, ckpt, device, out_dir):
    args.query_illum_align = use_align
    args.query_illum_alpha = alpha
    args.query_hist_beta = beta
    model = _load_model(args, ckpt, device)
    loader = DataLoader(TestSet(args), batch_size=1, shuffle=False,
                        num_workers=args.num_workers)
    per_image = []
    diag_acc = defaultdict(list)
    luma_keys = ['mean_low', 'mean_gamma', 'mean_final', 'mean_ref',
                 'cdf_l1_before_hist', 'cdf_l1_after_hist', 'gamma']
    luma_acc = defaultdict(list)

    with torch.no_grad():
        for i, sample in enumerate(loader):
            lr = sample['LR'].to(device)
            lr_sr = sample['LR_sr'].to(device)
            hr = sample['HR'].to(device)
            ref = sample['Ref'].to(device)
            ref_sr = sample['Ref_sr'].to(device)
            out = _tiled_forward(model, args, lr, lr_sr, ref, ref_sr,
                                 return_diag=True)
            sr = out[0]
            match_diag = out[5] if len(out) > 5 else None
            qinfo = out[6] if len(out) > 6 else model.last_query_illum_info
            if qinfo is None:
                _, qinfo = align_query_illumination(
                    lr_sr, ref, alpha=alpha, beta=beta,
                    gamma_min=args.query_gamma_min,
                    gamma_max=args.query_gamma_max,
                    hist_bins=args.query_hist_bins)

            _, ssim, mse, psnr_rgb = calc_psnr_and_ssim(sr, hr)
            row = {
                'idx': i,
                'psnr_rgb': float(psnr_rgb),
                'ssim_rgb': float(ssim),
                'mse': float(mse),
            }
            for k in luma_keys:
                if k in qinfo and qinfo[k] is not None:
                    v = qinfo[k]
                    fv = float(v.view(-1)[0].item()) if torch.is_tensor(v) else float(v)
                    row[k] = fv
                    luma_acc[k].append(fv)
            if match_diag is not None:
                for k, v in match_diag.items():
                    if isinstance(v, bool) or v is None:
                        row[k] = v
                        continue
                    fv = float(v)
                    row[k] = fv
                    diag_acc[k].append(fv)
            per_image.append(row)
            print('  [%s] %02d  PSNR=%.3f  SSIM=%.4f  g=%.3f  cdf=%.4f->%.4f' % (
                run_name, i, row['psnr_rgb'], row['ssim_rgb'],
                row.get('gamma', -1),
                row.get('cdf_l1_before_hist', -1),
                row.get('cdf_l1_after_hist', -1)))

    summary = {
        'run': run_name,
        'query_illum_align': use_align,
        'alpha': alpha,
        'beta': beta,
        'n': len(per_image),
        'psnr_rgb_mean': float(np.mean([r['psnr_rgb'] for r in per_image])),
        'ssim_rgb_mean': float(np.mean([r['ssim_rgb'] for r in per_image])),
        'match': {k: _stats(v) for k, v in diag_acc.items()},
        'luma': {k: _stats(v) for k, v in luma_acc.items()},
        'per_image': per_image,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, '%s_a%.2f_b%.2f.json' % (run_name, alpha, beta))
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('==> %s  PSNR=%.3f  SSIM=%.4f  -> %s' % (
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
        '/root/data/experiments/retry/v1r7-gamma-hist-align/zeroshot'))
    ap.add_argument('--runs', default='A0,A3,B2-25,B2-50')
    ap.add_argument('--cpu', action='store_true')
    cli = ap.parse_args()
    args = _build_args(cli)
    if not os.path.isdir(args.ref_dir):
        raise FileNotFoundError(args.ref_dir)
    device = torch.device('cpu' if args.cpu else 'cuda')
    want = set(x.strip() for x in cli.runs.split(',') if x.strip())
    results = []
    for name, use, alpha, beta in RUNS:
        if name not in want:
            continue
        print('======== %s align=%s a=%.2f b=%.2f ========' % (
            name, use, alpha, beta))
        results.append(run_one(
            name, use, alpha, beta, args, cli.ckpt, device, cli.out_dir))

    slim = []
    for r in results:
        slim.append({
            'run': r['run'], 'alpha': r['alpha'], 'beta': r['beta'],
            'psnr_rgb_mean': r['psnr_rgb_mean'],
            'ssim_rgb_mean': r['ssim_rgb_mean'],
            'max_cos_mean': (r['match'].get('max_cos_mean') or {}).get('mean'),
            'identity_cos_mean': (r['match'].get('identity_cos_mean') or {}).get('mean'),
            'match_l2_mean': (r['match'].get('match_l2_mean') or {}).get('mean'),
            'frac_dist_le_4': (r['match'].get('frac_dist_le_4') or {}).get('mean'),
            'query_feat_norm_mean': (r['match'].get('query_feat_norm_mean') or {}).get('mean'),
            'key_feat_norm_mean': (r['match'].get('key_feat_norm_mean') or {}).get('mean'),
            'mean_low': (r['luma'].get('mean_low') or {}).get('mean'),
            'mean_gamma': (r['luma'].get('mean_gamma') or {}).get('mean'),
            'mean_final': (r['luma'].get('mean_final') or {}).get('mean'),
            'mean_ref': (r['luma'].get('mean_ref') or {}).get('mean'),
            'cdf_l1_before': (r['luma'].get('cdf_l1_before_hist') or {}).get('mean'),
            'cdf_l1_after': (r['luma'].get('cdf_l1_after_hist') or {}).get('mean'),
        })
    table_path = os.path.join(cli.out_dir, 'summary.json')
    with open(table_path, 'w') as f:
        json.dump(slim, f, indent=2)
    print('\n=== SUMMARY ===')
    for r in slim:
        print('%s  a=%.2f b=%.2f  PSNR=%.3f  SSIM=%.4f  max_cos=%.3f  l2=%.1f  cdf=%.3f->%.3f' % (
            r['run'], r['alpha'], r['beta'], r['psnr_rgb_mean'], r['ssim_rgb_mean'],
            r['max_cos_mean'] or -1, r['match_l2_mean'] or -1,
            r['cdf_l1_before'] or -1, r['cdf_l1_after'] or -1))
    print('Wrote', table_path)


if __name__ == '__main__':
    main()
