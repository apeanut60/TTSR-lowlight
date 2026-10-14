#!/usr/bin/env python3
"""v1r9 zero-shot Identity↔B2 texture fusion eval (F0/F1/M0–M4)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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

from dataset.data1 import TestSet  # noqa: E402
from model import TTSREnhance  # noqa: E402
from model.query_illum_align import (  # noqa: E402
    align_query_illumination, estimate_query_align_maps, _rgb_luma01,
)
from utils import calc_psnr_and_ssim  # noqa: E402

CKPT = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt')
FOCUS = ['2037.jpg', '2041.jpg', '2053.jpg', '2056.jpg', '2062.jpg', '2063.jpg']

# name, fusion, lambda, align, alpha, beta, s_source (fusion only)
RUNS = [
    ('F0', False, 0.0, False, 0.0, 0.0, 'a0'),
    ('F1', False, 0.0, True, 0.75, 0.50, 'a0'),
    ('M0', True, 0.00, True, 0.75, 0.50, 'a0'),
    ('M1', True, 0.25, True, 0.75, 0.50, 'a0'),
    ('M2', True, 0.50, True, 0.75, 0.50, 'a0'),
    ('M3', True, 0.75, True, 0.75, 0.50, 'a0'),
    ('M4', True, 1.00, True, 0.75, 0.50, 'a0'),
]


def _build_args():
    return SimpleNamespace(
        dataset='data1', dataset_dir='/root/data/datasets/data1',
        data1_camera='Huawei',
        ref_dir='/root/data/datasets/data1/Eval/Huawei/nanobanana_ref',
        ref_degrade=False, eval_ref_degrade=None, ref_gamma=1.0,
        eval_degrade_seed=1234, enhance_mode=True,
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
        freeze_lte=False, ref_correction=True, ref_correction_feats=16,
        ref_illum_pool=8,
        query_illum_align=False, query_illum_alpha=0.0, query_hist_beta=0.0,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=False, texture_fusion_lambda=0.0,
        texture_fusion_s_source='a0',
        tile_size=256, tile_overlap=96, tile_window='cosine',
        cpu=False, num_gpu=1, num_workers=2, batch_size=1,
        eval_chroma_gain=1.0, eval_tta=False,
    )


def _load_model(args, device):
    model = TTSREnhance.TTSREnhance(args).to(device).eval()
    state = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    matched = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            matched += 1
    model.load_state_dict(cur)
    print('Loaded %d tensors from %s' % (matched, CKPT))
    return model


def _configure(model, fusion, lam, align, alpha, beta, s_source):
    model.args.texture_fusion = bool(fusion)
    model.args.texture_fusion_lambda = float(lam)
    model.args.texture_fusion_s_source = s_source
    model.args.query_illum_align = bool(align)
    model.args.query_illum_alpha = float(alpha)
    model.args.query_hist_beta = float(beta)


@torch.no_grad()
def _tiled_sr(model, args, lr, lr_sr, ref, ref_sr):
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
        return model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, **fwd)[0]
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


def _to_u8(t):
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = ((x + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return x


def _err_map(pred, gt):
    p = ((pred[0] + 1) * 0.5).clamp(0, 1)
    g = ((gt[0] + 1) * 0.5).clamp(0, 1)
    return (p - g).abs().mean(0).cpu().numpy()


def _luma_mae(pred, gt):
    return float((_rgb_luma01(pred) - _rgb_luma01(gt)).abs().mean().item())


def _luma_bias(pred, gt):
    return float((_rgb_luma01(pred) - _rgb_luma01(gt)).mean().item())


def _chroma_mae(pred, gt):
    p = ((pred + 1) * 0.5).clamp(0, 1)
    g = ((gt + 1) * 0.5).clamp(0, 1)
    yp, yg = _rgb_luma01(p), _rgb_luma01(g)
    return float(((p - yp) - (g - yg)).abs().mean().item())


def _mse_rgb(pred, gt):
    p = ((pred + 1) * 127.5).clamp(0, 255)
    g = ((gt + 1) * 127.5).clamp(0, 255)
    return float(((p - g) ** 2).mean().item())


def _resize_max(t, max_side=512):
    _, _, h, w = t.shape
    m = max(h, w)
    if m <= max_side:
        return t
    scale = max_side / float(m)
    nh = max(4, int(round(h * scale)) // 4 * 4)
    nw = max(4, int(round(w * scale)) // 4 * 4)
    return F.interpolate(t, size=(nh, nw), mode='bilinear', align_corners=False)


def _feat_diff_stats(tb, ti):
    """Relative L2 and cosine between T_B2 and T_id."""
    out = {}
    for name, a, b in zip(('lv3', 'lv2', 'lv1'), tb, ti):
        a = a.float()
        b = b.float()
        diff = a - b
        rel = float(diff.norm() / (a.norm().clamp(min=1e-8)))
        cos = float(F.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item())
        # spatial L2 map at this level (mean over channels)
        spat = diff.pow(2).mean(1).sqrt()[0].cpu().numpy()
        out[name] = {
            'rel_l2': rel,
            'cosine': cos,
            'spat_mean': float(spat.mean()),
            'spat_p90': float(np.percentile(spat, 90)),
            'spat_map': spat,
        }
    return out


@torch.no_grad()
def _fusion_feat_diag(model, args, lr, lr_sr, ref, ref_sr, max_side=512):
    """Run one scaled forward with fusion to measure T_id vs T_B2."""
    if not getattr(args, 'texture_fusion', False):
        return None
    kw = dict(
        lr=_resize_max(lr, max_side),
        lrsr=_resize_max(lr_sr, max_side),
        ref=_resize_max(ref, max_side),
        refsr=_resize_max(ref_sr, max_side),
    )
    use = args.query_illum_align
    alpha, beta = args.query_illum_alpha, args.query_hist_beta
    if use and (alpha > 0 or beta > 0):
        g, lut, _ = estimate_query_align_maps(
            kw['lrsr'], kw['ref'], alpha=alpha,
            gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
            bins=args.query_hist_bins)
        kw['query_gamma_override'] = g
        if beta > 0:
            kw['query_hist_lut_override'] = lut
    _ = model(**kw)
    info = model.last_texture_fusion_info
    if info is None:
        return None
    return _feat_diff_stats(info['T_b2'], info['T_id'])


def _contact_focus(out_path, fname, srs, hr, run_names):
    n = len(run_names)
    fig = plt.figure(figsize=(3.2 * n, 7.5))
    errs = [_err_map(srs[r], hr) for r in run_names]
    vmax = max(0.05, float(np.percentile(np.concatenate(
        [e.ravel() for e in errs]), 99)))
    for i, r in enumerate(run_names):
        ax = fig.add_subplot(2, n, i + 1)
        ax.imshow(_to_u8(srs[r]))
        ax.set_title(r, fontsize=9)
        ax.set_axis_off()
        ax = fig.add_subplot(2, n, n + i + 1)
        im = ax.imshow(errs[i], vmin=0, vmax=vmax, cmap='magma')
        ax.set_title('|RGB| %s' % r, fontsize=8)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(fname, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(
        ROOT, 'diagnostics', 'v1r9'))
    ap.add_argument('--runs', default='F0,F1,M0,M1,M2,M3,M4')
    args_cli = ap.parse_args()
    out = args_cli.out_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'focus_compare'), exist_ok=True)
    os.makedirs(os.path.join(out, 't_diff_maps'), exist_ok=True)

    want = set(args_cli.runs.split(','))
    runs = [r for r in RUNS if r[0] in want]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = _build_args()
    model = _load_model(args, device)

    # Phase 0 geometry note
    geom = {
        'low_native': '960x720 jpg',
        'nano_native': '1195x896 png (stem-matched)',
        'eval_preprocess': 'Ref resized to Low HxW via cv2.INTER_LINEAR; '
                           'both cropped to multiples of 4; tiles share (y0,x0)',
        'identity_grid': 'valid after preprocess (H==Hr, W==Wr at RGB and lv3)',
        'caveat': 'Identity is a spatial prior on the resized Nano grid, '
                  'not oracle GT alignment; native Nano size differs.',
    }
    with open(os.path.join(out, 'phase0_geometry_audit.json'), 'w') as f:
        json.dump(geom, f, indent=2)

    ds = TestSet(args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Resolve filenames
    from dataset.data1 import _collect_pairs
    pairs = _collect_pairs(os.path.join(args.dataset_dir, 'Eval'),
                           camera_filter='Huawei')
    filenames = [os.path.basename(p[0]) for p in pairs]

    rows = []
    focus_srs = {fn: {} for fn in FOCUS}
    focus_hr = {}
    tdiff_summary = []

    for run_name, fusion, lam, align, alpha, beta, s_src in runs:
        _configure(model, fusion, lam, align, alpha, beta, s_src)
        print('=== %s fusion=%s λ=%.2f align=%s α=%.2f β=%.2f S=%s ===' % (
            run_name, fusion, lam, align, alpha, beta, s_src))
        for i, batch in enumerate(loader):
            fname = filenames[i]
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)

            sr = _tiled_sr(model, model.args, lr, lrsr, ref, refsr)
            _, ssim, mse_u, psnr_rgb = calc_psnr_and_ssim(sr, hr)
            row = {
                'idx': i, 'filename': fname, 'run': run_name,
                'psnr_rgb': float(psnr_rgb), 'ssim_rgb': float(ssim),
                'mse_rgb': float(mse_u),
                'luma_mae': _luma_mae(sr, hr),
                'luma_bias': _luma_bias(sr, hr),
                'chroma_mae': _chroma_mae(sr, hr),
                'fusion': int(fusion), 'lambda': lam,
                'align': int(align), 'alpha': alpha, 'beta': beta,
                's_source': s_src if fusion else ('B2' if align else 'A0'),
            }
            # T_B2 vs T_id on focus + all for M runs (scaled)
            if fusion and (fname in FOCUS or True):
                stats = _fusion_feat_diag(
                    model, model.args, lr, lrsr, ref, refsr, max_side=512)
                if stats is not None:
                    for lv in ('lv3', 'lv2', 'lv1'):
                        row['Tdiff_%s_rel_l2' % lv] = stats[lv]['rel_l2']
                        row['Tdiff_%s_cos' % lv] = stats[lv]['cosine']
                    if fname in FOCUS and run_name == 'M2':
                        # save spatial diff maps at mid mix
                        for lv in ('lv3', 'lv2', 'lv1'):
                            fig, ax = plt.subplots(figsize=(5, 4))
                            im = ax.imshow(stats[lv]['spat_map'], cmap='magma')
                            ax.set_title('%s %s ||T_B2-T_id|| %s' % (
                                fname, run_name, lv))
                            fig.colorbar(im, ax=ax, fraction=0.046)
                            fig.tight_layout()
                            fig.savefig(os.path.join(
                                out, 't_diff_maps',
                                '%s_%s_%s.png' % (
                                    os.path.splitext(fname)[0], run_name, lv)),
                                dpi=120)
                            plt.close(fig)
                        tdiff_summary.append({
                            'filename': fname, 'run': run_name,
                            **{k: {kk: vv for kk, vv in stats[k].items()
                                   if kk != 'spat_map'}
                               for k in stats},
                        })
            rows.append(row)
            if fname in FOCUS:
                focus_srs[fname][run_name] = sr.detach().cpu()
                focus_hr[fname] = hr.detach().cpu()
            print('[%s %02d] %s  PSNR=%.3f SSIM=%.4f' % (
                run_name, i, fname, psnr_rgb, ssim))
            torch.cuda.empty_cache()

    # CSV
    keys = sorted({k for r in rows for k in r.keys()})
    # stable column order
    preferred = ['idx', 'filename', 'run', 'psnr_rgb', 'ssim_rgb', 'mse_rgb',
                 'luma_mae', 'luma_bias', 'chroma_mae', 'fusion', 'lambda',
                 'align', 'alpha', 'beta', 's_source']
    keys = preferred + [k for k in keys if k not in preferred]
    csv_path = os.path.join(out, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    # Summaries
    summary = {}
    by_run = {}
    for r in rows:
        by_run.setdefault(r['run'], []).append(r)
    f0 = {r['filename']: r for r in by_run.get('F0', [])}
    m0 = {r['filename']: r for r in by_run.get('M0', [])}
    for name, lst in by_run.items():
        psnrs = [r['psnr_rgb'] for r in lst]
        ssims = [r['ssim_rgb'] for r in lst]
        d_f0 = [r['psnr_rgb'] - f0[r['filename']]['psnr_rgb']
                for r in lst if r['filename'] in f0]
        d_m0 = [r['psnr_rgb'] - m0[r['filename']]['psnr_rgb']
                for r in lst if r['filename'] in m0]
        summary[name] = {
            'psnr_mean': float(np.mean(psnrs)),
            'ssim_mean': float(np.mean(ssims)),
            'luma_mae_mean': float(np.mean([r['luma_mae'] for r in lst])),
            'luma_bias_mean': float(np.mean([r['luma_bias'] for r in lst])),
            'chroma_mae_mean': float(np.mean([r['chroma_mae'] for r in lst])),
            'mse_mean': float(np.mean([r['mse_rgb'] for r in lst])),
            'delta_psnr_vs_F0': float(np.mean(d_f0)) if d_f0 else None,
            'delta_psnr_vs_M0': float(np.mean(d_m0)) if d_m0 else None,
            'n_up_vs_F0': int(sum(1 for d in d_f0 if d > 0.01)),
            'n_down_vs_F0': int(sum(1 for d in d_f0 if d < -0.01)),
            'n_up_vs_M0': int(sum(1 for d in d_m0 if d > 0.01)),
            'n_down_vs_M0': int(sum(1 for d in d_m0 if d < -0.01)),
        }
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    focus_tbl = []
    for fn in FOCUS:
        entry = {'filename': fn}
        for name in [r[0] for r in runs]:
            hit = next((r for r in rows
                        if r['filename'] == fn and r['run'] == name), None)
            if hit:
                entry[name + '_psnr'] = hit['psnr_rgb']
                entry[name + '_ssim'] = hit['ssim_rgb']
                entry[name + '_luma_mae'] = hit['luma_mae']
                entry[name + '_luma_bias'] = hit['luma_bias']
        focus_tbl.append(entry)
    with open(os.path.join(out, 'focus_images.json'), 'w') as f:
        json.dump(focus_tbl, f, indent=2)
    with open(os.path.join(out, 'tdiff_summary.json'), 'w') as f:
        json.dump(tdiff_summary, f, indent=2)

    # Focus contact sheets (F0 + M*)
    sheet_runs = [r[0] for r in runs if r[0] in
                  ('F0', 'M0', 'M1', 'M2', 'M3', 'M4')]
    for fn in FOCUS:
        if not all(k in focus_srs[fn] for k in sheet_runs):
            continue
        _contact_focus(
            os.path.join(out, 'focus_compare',
                         '%s_fusion.png' % os.path.splitext(fn)[0]),
            fn, focus_srs[fn], focus_hr[fn], sheet_runs)

    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
