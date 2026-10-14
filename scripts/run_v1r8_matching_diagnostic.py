#!/usr/bin/env python3
"""v1r8 matching diagnostic — visual + quantitative, no training.

Compares A0 / A3 / B2-50 on frozen v1r3e4 for Huawei Nano Eval (30 images).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from dataset.data1 import TestSet, _collect_pairs  # noqa: E402
from model import TTSREnhance  # noqa: E402
from model.query_illum_align import (  # noqa: E402
    align_query_illumination, estimate_query_align_maps, _rgb_luma01,
)
from utils import calc_psnr_and_ssim  # noqa: E402

CKPT = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt')
FOCUS = [0, 4, 16, 19, 25, 26]
NEG = [0, 16, 4]
POS = [25, 26, 19]

METHODS = [
    ('A0', False, 0.0, 0.0),
    ('A3', True, 0.75, 0.0),
    ('B2-50', True, 0.75, 0.50),
]


def _args():
    return SimpleNamespace(
        dataset='data1', dataset_dir='/root/data/datasets/data1',
        data1_camera='Huawei',
        ref_dir='/root/data/datasets/data1/Eval/Huawei/nanobanana_ref',
        ref_degrade=False, eval_ref_degrade=None, ref_gamma=1.0,
        eval_degrade_seed=1234, enhance_mode=True,
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
        freeze_lte=False, ref_correction=True, ref_correction_feats=16,
        ref_illum_pool=8, query_illum_align=False, query_illum_alpha=0.0,
        query_hist_beta=0.0, query_gamma_min=0.3, query_gamma_max=1.0,
        query_hist_bins=256, tile_size=256, tile_overlap=96,
        tile_window='cosine', cpu=False, num_gpu=1, num_workers=0,
        batch_size=1, eval_chroma_gain=1.0, eval_tta=False,
    )


def _load(args, device):
    m = TTSREnhance.TTSREnhance(args).to(device).eval()
    st = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in st):
        st = {k.replace('module.', '', 1): v for k, v in st.items()}
    cur = m.state_dict()
    for k, v in st.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    m.load_state_dict(cur)
    return m


def _to_u8(t):
    x = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return x.squeeze(0).permute(1, 2, 0).numpy()


def _luma_np(rgb_u8):
    x = rgb_u8.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def _err_map(pred, gt):
    p = ((pred.detach().cpu().clamp(-1, 1) + 1) * 0.5).numpy()[0]
    g = ((gt.detach().cpu().clamp(-1, 1) + 1) * 0.5).numpy()[0]
    e = np.abs(p - g).mean(axis=0)  # HxW
    return e


def _chroma_err(pred, gt):
    """Mean |Cb|+|Cr| style error in simple opponent space."""
    p = ((pred.detach() + 1) * 0.5).clamp(0, 1)
    g = ((gt.detach() + 1) * 0.5).clamp(0, 1)
    yp = _rgb_luma01(p)
    yg = _rgb_luma01(g)
    # chroma ≈ rgb - y
    cp = p - yp
    cg = g - yg
    return float((cp - cg).abs().mean().item())


def _luma_err(pred, gt):
    p = ((pred.detach() + 1) * 0.5).clamp(0, 1)
    g = ((gt.detach() + 1) * 0.5).clamp(0, 1)
    return float((_rgb_luma01(p) - _rgb_luma01(g)).abs().mean().item())


def _resize_max(t, max_side=512):
    """Resize NCHW tensor so max(H,W)=max_side, multiples of 4."""
    _, _, h, w = t.shape
    m = max(h, w)
    if m <= max_side:
        nh, nw = h // 4 * 4, w // 4 * 4
        return t[:, :, :nh, :nw]
    scale = max_side / float(m)
    nh = max(4, int(h * scale) // 4 * 4)
    nw = max(4, int(w * scale) // 4 * 4)
    return F.interpolate(t, size=(nh, nw), mode='bilinear', align_corners=False)


@torch.no_grad()
def _tiled_sr(model, args, lr, lr_sr, ref, ref_sr):
    """Production-equivalent tiled prediction (no diag tensors retained)."""
    tile_size = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile_size // 2))
    use = getattr(args, 'query_illum_align', False)
    alpha = float(getattr(args, 'query_illum_alpha', 0.0))
    beta = float(getattr(args, 'query_hist_beta', 0.0))
    fwd = {}
    if use and (alpha > 0 or beta > 0):
        g, lut, _ = estimate_query_align_maps(
            lr_sr, ref, alpha=alpha,
            gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
            bins=args.query_hist_bins)
        fwd['query_gamma_override'] = g
        if beta > 0:
            fwd['query_hist_lut_override'] = lut
    _, _, H, W = lr.shape
    if H <= tile_size and W <= tile_size:
        sr = model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, **fwd)[0]
        return sr
    stride = max(1, tile_size - overlap)
    n_h = max(1, (H - overlap + stride - 1) // stride)
    n_w = max(1, (W - overlap + stride - 1) // stride)
    sr_accum = torch.zeros(1, 3, H, W, device=lr.device)
    weight = torch.zeros(1, 1, H, W, device=lr.device)
    my = torch.ones(tile_size, device=lr.device)
    mx = torch.ones(tile_size, device=lr.device)
    if overlap > 0:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, math.pi, overlap, device=lr.device))
        my[:overlap] = ramp
        my[-overlap:] = ramp.flip(0)
        mx[:overlap] = ramp
        mx[-overlap:] = ramp.flip(0)
    feather = (my[:, None] * mx[None, :]).view(1, 1, tile_size, tile_size)
    for ih in range(n_h):
        y0 = min(ih * stride, H - tile_size)
        for iw in range(n_w):
            x0 = min(iw * stride, W - tile_size)
            kw = dict(
                lr=lr[:, :, y0:y0 + tile_size, x0:x0 + tile_size],
                lrsr=lr_sr[:, :, y0:y0 + tile_size, x0:x0 + tile_size],
                ref=ref[:, :, y0:y0 + tile_size, x0:x0 + tile_size],
                refsr=ref_sr[:, :, y0:y0 + tile_size, x0:x0 + tile_size],
                apply_illum=False, apply_ref_illum=False, **fwd)
            tile_sr = model(**kw)[0]
            sr_accum[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += tile_sr * feather
            weight[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += feather
    sr = sr_accum / weight.clamp(min=1e-8)
    sr = model.apply_ref_illum(sr, lr, ref)
    sr = model.apply_illum_head(sr, lr)
    return sr


@torch.no_grad()
def _match_diag_scaled(model, args, lr, lr_sr, ref, ref_sr, max_side=512):
    """Match diagnostics on a fixed max-side resize (memory-safe).

    Documented scale: long side ≤512, same for all methods.
    """
    lr_d = _resize_max(lr, max_side)
    lrsr_d = _resize_max(lr_sr, max_side)
    ref_d = _resize_max(ref, max_side)
    refsr_d = _resize_max(ref_sr, max_side)
    use = getattr(args, 'query_illum_align', False)
    alpha = float(getattr(args, 'query_illum_alpha', 0.0))
    beta = float(getattr(args, 'query_hist_beta', 0.0))
    kw = dict(lr=lr_d, lrsr=lrsr_d, ref=ref_d, refsr=refsr_d, return_diag=True)
    if use and (alpha > 0 or beta > 0):
        g, lut, _ = estimate_query_align_maps(
            lrsr_d, ref_d, alpha=alpha,
            gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
            bins=args.query_hist_bins)
        kw['query_gamma_override'] = g
        if beta > 0:
            kw['query_hist_lut_override'] = lut
    out = model(**kw)
    sr_d, S, T3, T2, T1, diag, qinfo = out
    return {
        'sr_diag': sr_d, 'S': S, 'T3': T3, 'T2': T2, 'T1': T1,
        'diag': diag, 'qinfo': qinfo,
        'lr_d': lr_d, 'ref_d': ref_d, 'hr_unused': None,
    }


@torch.no_grad()
def _forward_method(model, args, lr, lr_sr, ref, ref_sr, hr):
    """Tiled full-res prediction + scaled match diagnostics."""
    sr = _tiled_sr(model, args, lr, lr_sr, ref, ref_sr)
    torch.cuda.empty_cache()
    md = _match_diag_scaled(model, args, lr, lr_sr, ref, ref_sr, max_side=512)
    torch.cuda.empty_cache()
    q_gamma, _ = align_query_illumination(
        lr_sr, ref, alpha=0.75, beta=0.0,
        gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max)
    q_b2, _ = align_query_illumination(
        lr_sr, ref, alpha=0.75, beta=0.50,
        gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
        hist_bins=args.query_hist_bins)
    gt_d = _gt_feat_dist(model, md['T3'], md['T2'], md['T1'],
                         _resize_max(hr, 512))
    # Detach heavy tensors to CPU for viz; free GPU T's
    diag = md['diag']
    S_cpu = md['S'].detach().cpu()
    result = {
        'sr': sr.detach(),
        'S': S_cpu,
        'diag': diag,
        'qinfo': md['qinfo'],
        'q_gamma': q_gamma.detach().cpu(),
        'q_b2': q_b2.detach().cpu(),
        'lr_d': md['lr_d'].detach().cpu(),
        'ref_d': md['ref_d'].detach().cpu(),
        'gt_feat': gt_d,
    }
    del md, sr
    torch.cuda.empty_cache()
    return result


def _disp_heatmap(match_arg, H, W):
    idx = torch.arange(H * W)
    by = torch.div(match_arg.view(-1), W, rounding_mode='floor')
    bx = match_arg.view(-1) % W
    iy = torch.div(idx, W, rounding_mode='floor')
    ix = idx % W
    dist = torch.sqrt((by.float() - iy.float()).pow(2) +
                      (bx.float() - ix.float()).pow(2))
    return dist.view(H, W).numpy()


def _draw_matches(ax, low_u8, ref_u8, match_arg, H, W, n_pts=12, seed=0):
    """Draw Query→Ref lines for fixed sample points."""
    rng = np.random.RandomState(seed)
    # sample on a grid for stability across methods
    ys = np.linspace(H * 0.2, H * 0.8, 3).astype(int)
    xs = np.linspace(W * 0.2, W * 0.8, 4).astype(int)
    pts = [(int(y), int(x)) for y in ys for x in xs][:n_pts]
    # side-by-side
    h, w = low_u8.shape[:2]
    canvas = np.concatenate([low_u8, ref_u8], axis=1)
    ax.imshow(canvas)
    scale_y = h / float(H)
    scale_x = w / float(W)
    for (qy, qx) in pts:
        qi = qy * W + qx
        ki = int(match_arg.view(-1)[qi].item())
        ky, kx = divmod(ki, W)
        x0 = qx * scale_x
        y0 = qy * scale_y
        x1 = w + kx * scale_x
        y1 = ky * scale_y
        ax.plot([x0, x1], [y0, y1], '-', color='lime', lw=0.6, alpha=0.85)
        ax.scatter([x0], [y0], s=8, c='yellow', zorder=3)
        ax.scatter([x1], [y1], s=8, c='cyan', zorder=3)
    ax.set_axis_off()


def _contact_sheet(out_path, fname, bundles, lr, ref, hr, err_vmax):
    """bundles: dict method -> forward result.

    Predictions/error/hist use full-res tensors. Match heatmaps/arrows use the
    memory-safe diagnostic scale (max side 512) stored in lr_d/ref_d.
    """
    fig = plt.figure(figsize=(18, 14))
    imgs = [
        ('Low', _to_u8(lr)),
        ('Nano Ref', _to_u8(ref)),
        ('GT', _to_u8(hr)),
        ('Query γ', _to_u8(bundles['A3']['q_gamma'])),
        ('Query γ+hist', _to_u8(bundles['B2-50']['q_b2'])),
    ]
    for i, (title, im) in enumerate(imgs):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    for i, m in enumerate(['A0', 'A3', 'B2-50']):
        ax = fig.add_subplot(4, 5, 6 + i)
        ax.imshow(_to_u8(bundles[m]['sr']))
        ax.set_title('%s pred' % m, fontsize=9)
        ax.set_axis_off()
    for i in range(3, 5):
        ax = fig.add_subplot(4, 5, 6 + i)
        ax.set_axis_off()
    for i, m in enumerate(['A0', 'A3', 'B2-50']):
        ax = fig.add_subplot(4, 5, 11 + i)
        e = _err_map(bundles[m]['sr'], hr)
        im = ax.imshow(e, vmin=0, vmax=err_vmax, cmap='magma')
        ax.set_title('%s |RGB err|' % m, fontsize=9)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = fig.add_subplot(4, 5, 14)
    for lab, arr, c in [
        ('Low', _luma_np(_to_u8(lr)), 'C0'),
        ('Qγ', _luma_np(_to_u8(bundles['A3']['q_gamma'])), 'C1'),
        ('QB2', _luma_np(_to_u8(bundles['B2-50']['q_b2'])), 'C2'),
        ('Ref', _luma_np(_to_u8(ref)), 'C3'),
        ('GT', _luma_np(_to_u8(hr)), 'C4'),
    ]:
        ax.hist(arr.ravel(), bins=64, range=(0, 1), histtype='step',
                density=True, label=lab, color=c, lw=1.0)
    ax.legend(fontsize=6)
    ax.set_title('Luma hist', fontsize=9)
    ax.set_xlim(0, 1)
    dmax = 0
    dmaps = {}
    for m in ['A0', 'A3', 'B2-50']:
        H, W, _, _ = bundles[m]['diag']['lv3_hw']
        d = _disp_heatmap(bundles[m]['diag']['match_arg'][0], H, W)
        dmaps[m] = d
        dmax = max(dmax, float(np.percentile(d, 95)))
    dmax = max(dmax, 1.0)
    for i, m in enumerate(['A0', 'A3', 'B2-50']):
        ax = fig.add_subplot(4, 5, 16 + i)
        im = ax.imshow(dmaps[m], vmin=0, vmax=dmax, cmap='viridis')
        ax.set_title('%s match L2@512' % m, fontsize=9)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = fig.add_subplot(4, 5, 19)
    H, W, _, _ = bundles['A0']['diag']['lv3_hw']
    _draw_matches(ax, _to_u8(bundles['A0']['lr_d']), _to_u8(bundles['A0']['ref_d']),
                  bundles['A0']['diag']['match_arg'][0], H, W, seed=0)
    ax.set_title('A0 matches@512', fontsize=9)
    ax = fig.add_subplot(4, 5, 20)
    _draw_matches(ax, _to_u8(bundles['B2-50']['lr_d']),
                  _to_u8(bundles['B2-50']['ref_d']),
                  bundles['B2-50']['diag']['match_arg'][0], H, W, seed=0)
    ax.set_title('B2 matches@512', fontsize=9)

    fig.suptitle(fname, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _gt_feat_dist(model, T3, T2, T1, hr):
    """Offline: distance of transferred T to LTE(GT). Does not affect inference."""
    with torch.no_grad():
        g1, g2, g3 = model.LTE((hr + 1.) / 2.)
    # Cosine distance on flattened channels: 1 - cos_sim mean
    def _cd(a, b):
        # resize if needed
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode='bilinear',
                              align_corners=False)
        a_n = F.normalize(a, dim=1)
        b_n = F.normalize(b, dim=1)
        cos = (a_n * b_n).sum(dim=1).mean()
        return float((1.0 - cos).item())
    return {
        'T3_gt_cosdist': _cd(T3, g3),
        'T2_gt_cosdist': _cd(T2, g2),
        'T1_gt_cosdist': _cd(T1, g1),
    }


def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(
        ROOT, 'diagnostics', 'v1r8'))
    cli = ap.parse_args()
    out = cli.out_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'contact_sheets'), exist_ok=True)
    os.makedirs(os.path.join(out, 'match_viz'), exist_ok=True)
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = _args()
    pairs = _collect_pairs(os.path.join(args.dataset_dir, 'Eval'),
                           camera_filter='Huawei')
    index_map = []
    for i, (lp, hp) in enumerate(pairs):
        index_map.append({
            'idx': i,
            'filename': os.path.basename(lp),
            'low_path': lp,
            'high_path': hp,
            'focus': i in FOCUS,
            'group': ('neg' if i in NEG else ('pos' if i in POS else 'other')),
        })
    with open(os.path.join(out, 'index_to_filename.json'), 'w') as f:
        json.dump(index_map, f, indent=2)

    model = _load(args, device)
    ds = TestSet(args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Precompute shared err vmax from A0 for focus images later
    rows = []
    match_rows = []
    focus_cache = {}

    for i, sample in enumerate(loader):
        fname = index_map[i]['filename']
        lr = sample['LR'].to(device)
        lr_sr = sample['LR_sr'].to(device)
        hr = sample['HR'].to(device)
        ref = sample['Ref'].to(device)
        ref_sr = sample['Ref_sr'].to(device)

        bundles = {}
        for name, use, alpha, beta in METHODS:
            args.query_illum_align = use
            args.query_illum_alpha = alpha
            args.query_hist_beta = beta
            bundles[name] = _forward_method(
                model, args, lr, lr_sr, ref, ref_sr, hr)

        a0_arg = bundles['A0']['diag']['match_arg'][0].view(-1)
        H, W, _, _ = bundles['A0']['diag']['lv3_hw']

        for name, use, alpha, beta in METHODS:
            b = bundles[name]
            _, ssim, mse, psnr_rgb = calc_psnr_and_ssim(b['sr'], hr)
            d = b['diag']
            marg = d['match_arg'][0].view(-1)
            changed = float((marg != a0_arg).float().mean().item())
            S = b['S']
            S_sig = torch.sigmoid(S)
            qinfo = b['qinfo'] or {}

            def _qf(k, default=None):
                if k not in qinfo or qinfo[k] is None:
                    return default
                v = qinfo[k]
                if torch.is_tensor(v):
                    return float(v.view(-1)[0].item())
                return float(v)

            row = {
                'idx': i, 'filename': fname, 'method': name,
                'alpha': alpha, 'beta': beta,
                'psnr_rgb': float(psnr_rgb), 'ssim_rgb': float(ssim),
                'mse': float(mse),
                'luma_err': _luma_err(b['sr'].to(device), hr),
                'chroma_err': _chroma_err(b['sr'].to(device), hr),
                'max_cos_mean': d['max_cos_mean'],
                'max_cos_q50': d['max_cos_q50'],
                'max_cos_q90': d['max_cos_q90'],
                'identity_cos_mean': d.get('identity_cos_mean'),
                'match_l2_mean': d.get('match_l2_mean'),
                'frac_dist_le_1': d.get('frac_dist_le_1'),
                'frac_dist_le_2': d.get('frac_dist_le_2'),
                'frac_dist_le_4': d.get('frac_dist_le_4'),
                'frac_dist_le_8': d.get('frac_dist_le_8'),
                'match_changed_frac_vs_A0': changed if name != 'A0' else 0.0,
                'S_raw_mean': float(S.mean().item()),
                'S_raw_std': float(S.std(unbiased=False).item()),
                'S_sigmoid_mean': float(S_sig.mean().item()),
                'S_sigmoid_std': float(S_sig.std(unbiased=False).item()),
                'query_feat_norm_mean': d['query_feat_norm_mean'],
                'key_feat_norm_mean': d['key_feat_norm_mean'],
                'mean_low': _qf('mean_low'),
                'mean_gamma': _qf('mean_gamma'),
                'mean_final': _qf('mean_final'),
                'mean_ref': _qf('mean_ref'),
                'cdf_l1_after': _qf('cdf_l1_after_hist'),
                'match_diag_scale': 'max_side_512',
                **b['gt_feat'],
            }
            rows.append(row)
            match_rows.append(row)

        if i in FOCUS:
            errs = [_err_map(bundles[m]['sr'], hr) for m in ['A0', 'A3', 'B2-50']]
            vmax = float(np.percentile(np.concatenate([e.ravel() for e in errs]), 99))
            vmax = max(vmax, 0.05)
            sheet = os.path.join(out, 'contact_sheets',
                                 '%02d_%s.png' % (i, os.path.splitext(fname)[0]))
            _contact_sheet(sheet, '%02d %s' % (i, fname), bundles,
                           lr.cpu(), ref.cpu(), hr.cpu(), vmax)
            for m in ['A0', 'A3', 'B2-50']:
                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                _draw_matches(ax, _to_u8(bundles[m]['lr_d']),
                              _to_u8(bundles[m]['ref_d']),
                              bundles[m]['diag']['match_arg'][0], H, W, seed=0)
                ax.set_title('%s %s matches@512' % (fname, m))
                fig.tight_layout()
                fig.savefig(os.path.join(
                    out, 'match_viz', '%02d_%s_%s.png' % (
                        i, os.path.splitext(fname)[0], m)), dpi=130)
                plt.close(fig)
            focus_cache[i] = {
                'filename': fname,
                'A0_psnr': next(r['psnr_rgb'] for r in rows
                                if r['idx'] == i and r['method'] == 'A0'),
                'B2_psnr': next(r['psnr_rgb'] for r in rows
                                if r['idx'] == i and r['method'] == 'B2-50'),
            }
        # free
        del bundles
        torch.cuda.empty_cache()
        print('[%02d] %s  A0=%.3f A3=%.3f B2=%.3f' % (
            i, fname,
            next(r['psnr_rgb'] for r in rows if r['idx'] == i and r['method'] == 'A0'),
            next(r['psnr_rgb'] for r in rows if r['idx'] == i and r['method'] == 'A3'),
            next(r['psnr_rgb'] for r in rows if r['idx'] == i and r['method'] == 'B2-50'),
        ))

    # Write CSVs
    metric_csv = os.path.join(out, 'per_image_metrics.csv')
    keys = list(rows[0].keys())
    with open(metric_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Correlation: ΔPSNR (B2-A0) vs diagnostic deltas
    a0 = {r['idx']: r for r in rows if r['method'] == 'A0'}
    b2 = {r['idx']: r for r in rows if r['method'] == 'B2-50'}
    dpsnr = [b2[i]['psnr_rgb'] - a0[i]['psnr_rgb'] for i in range(30)]
    corr_specs = [
        ('delta_id_cos', [b2[i]['identity_cos_mean'] - a0[i]['identity_cos_mean']
                          for i in range(30)]),
        ('delta_neg_l2', [-(b2[i]['match_l2_mean'] - a0[i]['match_l2_mean'])
                          for i in range(30)]),
        ('match_changed_frac', [b2[i]['match_changed_frac_vs_A0'] for i in range(30)]),
        ('delta_neg_T3_gt', [-(b2[i]['T3_gt_cosdist'] - a0[i]['T3_gt_cosdist'])
                             for i in range(30)]),
        ('delta_neg_luma_err', [-(b2[i]['luma_err'] - a0[i]['luma_err'])
                                for i in range(30)]),
        ('delta_max_cos', [b2[i]['max_cos_mean'] - a0[i]['max_cos_mean']
                           for i in range(30)]),
    ]
    corr = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (name, xs) in zip(axes, corr_specs):
        rho = _spearman(xs, dpsnr)
        corr[name] = rho
        ax.scatter(xs, dpsnr, s=18, alpha=0.8)
        ax.axhline(0, color='gray', lw=0.8)
        ax.axvline(0, color='gray', lw=0.8)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel('ΔPSNR B2−A0', fontsize=8)
        ax.set_title('Spearman ρ=%.3f' % rho, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'correlation_scatter.png'), dpi=140)
    plt.close(fig)
    with open(os.path.join(out, 'correlations.json'), 'w') as f:
        json.dump({'spearman_vs_delta_psnr_B2_minus_A0': corr,
                   'n': 30, 'note': 'exploratory'}, f, indent=2)

    # Summary table
    summary = {}
    for name, _, _, _ in METHODS:
        sub = [r for r in rows if r['method'] == name]
        summary[name] = {
            'psnr_mean': float(np.mean([r['psnr_rgb'] for r in sub])),
            'ssim_mean': float(np.mean([r['ssim_rgb'] for r in sub])),
            'max_cos_mean': float(np.mean([r['max_cos_mean'] for r in sub])),
            'id_cos_mean': float(np.mean([r['identity_cos_mean'] for r in sub])),
            'l2_mean': float(np.mean([r['match_l2_mean'] for r in sub])),
            'changed_vs_A0': float(np.mean([r['match_changed_frac_vs_A0'] for r in sub])),
            'S_sigmoid_mean': float(np.mean([r['S_sigmoid_mean'] for r in sub])),
            'T3_gt_cosdist': float(np.mean([r['T3_gt_cosdist'] for r in sub])),
            'luma_err': float(np.mean([r['luma_err'] for r in sub])),
            'chroma_err': float(np.mean([r['chroma_err'] for r in sub])),
        }
    with open(os.path.join(out, 'summary_methods.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Focus deltas
    focus_tbl = []
    for i in FOCUS:
        a = a0[i]
        b = b2[i]
        a3 = next(r for r in rows if r['idx'] == i and r['method'] == 'A3')
        focus_tbl.append({
            'idx': i, 'filename': a['filename'],
            'group': 'neg' if i in NEG else 'pos',
            'A0_psnr': a['psnr_rgb'], 'A3_psnr': a3['psnr_rgb'],
            'B2_psnr': b['psnr_rgb'],
            'B2_minus_A0': b['psnr_rgb'] - a['psnr_rgb'],
            'changed_frac': b['match_changed_frac_vs_A0'],
            'delta_l2': b['match_l2_mean'] - a['match_l2_mean'],
            'delta_id_cos': b['identity_cos_mean'] - a['identity_cos_mean'],
            'delta_T3_gt': b['T3_gt_cosdist'] - a['T3_gt_cosdist'],
            'delta_luma_err': b['luma_err'] - a['luma_err'],
            'A0_S_sig': a['S_sigmoid_mean'], 'B2_S_sig': b['S_sigmoid_mean'],
        })
    with open(os.path.join(out, 'focus_image_deltas.json'), 'w') as f:
        json.dump(focus_tbl, f, indent=2)

    print('\n=== METHOD SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print('\n=== CORRELATIONS ===')
    print(json.dumps(corr, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
