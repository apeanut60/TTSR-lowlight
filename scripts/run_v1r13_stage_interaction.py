#!/usr/bin/env python3
"""v1r13: lv3 Texture-Zero × lv2 Bypass interaction on frozen M1."""
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
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from dataset.data1 import TestSet, _collect_pairs  # noqa: E402
from model import TTSREnhance  # noqa: E402
from model.query_illum_align import (  # noqa: E402
    estimate_query_align_maps, _rgb_luma01,
)
from utils import calc_psnr_and_ssim  # noqa: E402

CKPT = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt')
FOCUS = ['2037.jpg', '2040.jpg', '2053.jpg', '2041.jpg',
         '2063.jpg', '2056.jpg', '2062.jpg']
A0_PSNR = 20.162
A0_CSV_CANDIDATES = [
    os.path.join(ROOT, 'diagnostics/v1r8/per_image_metrics.csv'),
    os.path.join(ROOT, 'diagnostics/v1r9/per_image_metrics.csv'),
]

RUNS = [
    ('P0', 'normal', 'normal', 'normal'),
    ('P1', 'texture_zero', 'normal', 'normal'),
    ('P2', 'normal', 'branch_bypass', 'normal'),
    ('P3', 'texture_zero', 'branch_bypass', 'normal'),
]
REGRESSION = {
    'P0': (20.1805, 0.6327),
    'P1': (20.234, None),
    'P2': (20.264, None),
}


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
        query_illum_align=True, query_illum_alpha=0.75, query_hist_beta=0.50,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=True, texture_fusion_lambda=0.25,
        texture_fusion_s_source='a0',
        gate_mode=None, gate_shift='', gate_smooth_kernel=31,
        collect_gate_stats=False, collect_residual_decomp=False,
        disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
        stage_mode_lv3=None, stage_mode_lv2=None, stage_mode_lv1=None,
        tile_size=256, tile_overlap=96, tile_window='cosine',
        cpu=False, num_gpu=1, num_workers=0, batch_size=1,
        eval_chroma_gain=1.0, eval_tta=False,
    )


def _load(args, device):
    model = TTSREnhance.TTSREnhance(args).to(device).eval()
    state = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    n = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            n += 1
    model.load_state_dict(cur)
    print('Loaded', n)
    return model


def _cfg(model, m3, m2, m1):
    model.args.stage_mode_lv3 = m3
    model.args.stage_mode_lv2 = m2
    model.args.stage_mode_lv1 = m1
    model.args.disable_T_lv3 = False
    model.args.disable_T_lv2 = False
    model.args.disable_T_lv1 = False
    model.args.texture_fusion = True
    model.args.texture_fusion_lambda = 0.25
    model.args.texture_fusion_s_source = 'a0'
    model.args.query_illum_align = True
    model.args.query_illum_alpha = 0.75
    model.args.query_hist_beta = 0.50
    model.args.gate_mode = None


def _as01(x):
    return ((x + 1) * 0.5).clamp(0, 1)


def _luma_mae(p, g):
    return float((_rgb_luma01(_as01(p)) - _rgb_luma01(_as01(g))).abs().mean())


def _luma_bias(p, g):
    return float((_rgb_luma01(_as01(p)) - _rgb_luma01(_as01(g))).mean())


def _chroma_mae(p, g):
    p1, g1 = _as01(p), _as01(g)
    return float(((p1 - _rgb_luma01(p1)) - (g1 - _rgb_luma01(g1))).abs().mean())


def _to_u8(t):
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    return ((x + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def _err(pred, gt):
    return (_as01(pred)[0] - _as01(gt)[0]).abs().mean(0).cpu().numpy()


def _luma_err_map(pred, gt):
    return (_rgb_luma01(_as01(pred))[0, 0] -
            _rgb_luma01(_as01(gt))[0, 0]).cpu().numpy()


def _bootstrap(d, n_boot=5000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        'mean': float(d.mean()), 'median': float(np.median(d)),
        'std': float(d.std(ddof=0)),
        'ci95': [float(lo), float(hi)],
        'note': 'exploratory; Eval30 reused across v1r6–v1r12',
    }


def _paired_stats(deltas, names):
    d = np.asarray(deltas, float)
    wins = int((d > 0.01).sum())
    losses = int((d < -0.01).sum())
    ties = int(len(d) - wins - losses)
    # trimmed mean 10% each tail
    k = max(1, int(0.1 * len(d)))
    ds = np.sort(d)
    trimmed = float(ds[k:-k].mean()) if len(d) > 2 * k else float(d.mean())
    # leave-one-out mean range
    loo = []
    for i in range(len(d)):
        loo.append(float(np.mean(np.delete(d, i))))
    # exclude two largest positive
    order = np.argsort(-d)
    mask = np.ones(len(d), dtype=bool)
    if len(d) >= 3:
        mask[order[:2]] = False
    excl2 = float(d[mask].mean()) if mask.any() else None
    top2 = [(names[i], float(d[i])) for i in order[:2]]
    top2_share = float(d[order[:2]].sum() / max(abs(d.sum()), 1e-8)) if len(d) else None
    return {
        'mean': float(d.mean()),
        'median': float(np.median(d)),
        'std': float(d.std(ddof=0)),
        'wins': wins, 'losses': losses, 'ties': ties,
        'bootstrap': _bootstrap(d),
        'trimmed_mean_10pct': trimmed,
        'loo_mean_min': float(min(loo)),
        'loo_mean_max': float(max(loo)),
        'mean_excluding_top2_positive': excl2,
        'top2_positive': top2,
        'top2_share_of_sum_delta': top2_share,
        'worst3': [(names[i], float(d[i]))
                   for i in np.argsort(d)[:3]],
        'best3': [(names[i], float(d[i]))
                  for i in np.argsort(-d)[:3]],
    }


def _load_a0_psnr():
    for path in A0_CSV_CANDIDATES:
        if not os.path.isfile(path):
            continue
        rows = list(csv.DictReader(open(path)))
        # v1r8/v1r9 use method column A0 or run F0
        out = {}
        for r in rows:
            meth = r.get('method') or r.get('run') or ''
            if meth in ('A0', 'F0'):
                out[r['filename']] = float(r['psnr_rgb'])
        if out:
            return out, path
    return {}, None


@torch.no_grad()
def _tiled(model, args, lr, lrsr, ref, refsr):
    tile = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile // 2))
    g, lut, _ = estimate_query_align_maps(
        lrsr, ref, alpha=args.query_illum_alpha,
        gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
        bins=args.query_hist_bins)
    fwd = dict(
        query_gamma_override=g, query_hist_lut_override=lut,
        stage_mode_lv3=args.stage_mode_lv3,
        stage_mode_lv2=args.stage_mode_lv2,
        stage_mode_lv1=args.stage_mode_lv1,
    )
    _, _, H, W = lr.shape

    def one(y0, y1, x0, x1):
        return model(
            lr=lr[:, :, y0:y1, x0:x1],
            lrsr=lrsr[:, :, y0:y1, x0:x1],
            ref=ref[:, :, y0:y1, x0:x1],
            refsr=refsr[:, :, y0:y1, x0:x1],
            apply_illum=False, apply_ref_illum=False, **fwd)[0]

    if H <= tile and W <= tile:
        sr = one(0, H, 0, W)
    else:
        stride = max(1, tile - overlap)
        nh = max(1, (H - overlap + stride - 1) // stride)
        nw = max(1, (W - overlap + stride - 1) // stride)
        acc = torch.zeros(1, 3, H, W, device=lr.device)
        wt = torch.zeros(1, 1, H, W, device=lr.device)
        my = torch.ones(tile, device=lr.device)
        mx = torch.ones(tile, device=lr.device)
        if overlap > 0:
            ramp = 0.5 - 0.5 * torch.cos(
                torch.linspace(0, math.pi, overlap, device=lr.device))
            my[:overlap] = ramp
            my[-overlap:] = ramp.flip(0)
            mx[:overlap] = ramp
            mx[-overlap:] = ramp.flip(0)
        feather = (my[:, None] * mx[None, :]).view(1, 1, tile, tile)
        for ih in range(nh):
            y0 = min(ih * stride, H - tile)
            for iw in range(nw):
                x0 = min(iw * stride, W - tile)
                t = one(y0, y0 + tile, x0, x0 + tile)
                acc[:, :, y0:y0 + tile, x0:x0 + tile] += t * feather
                wt[:, :, y0:y0 + tile, x0:x0 + tile] += feather
        sr = acc / wt.clamp(min=1e-8)
    sr = model.apply_ref_illum(sr, lr, ref)
    sr = model.apply_illum_head(sr, lr)
    return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'diagnostics', 'v1r13'))
    cli = ap.parse_args()
    out = cli.out_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'focus'), exist_ok=True)

    audit = {
        'reused_interface': 'stage_mode_lv3/lv2/lv1 from v1r12',
        'P0': 'all normal = M1 / C0',
        'P1': 'lv3 texture_zero = C1; F3+=G*R(F3,0)',
        'P2': 'lv2 branch_bypass = C4; skip conv22_head only',
        'P3': 'lv3 texture_zero + lv2 bypass; lv1 normal',
        'dependency': 'stage2 receives features after modified stage1 (live forward)',
        'gate': 'original A0 S only',
    }
    with open(os.path.join(out, 'code_audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = _build_args()
    model = _load(args, device)
    pairs = _collect_pairs(os.path.join(args.dataset_dir, 'Eval'),
                           camera_filter='Huawei')
    filenames = [os.path.basename(p[0]) for p in pairs]
    loader = DataLoader(TestSet(args), batch_size=1, shuffle=False,
                        num_workers=0)
    a0_map, a0_src = _load_a0_psnr()

    rows = []
    focus = {fn: {} for fn in FOCUS}

    for name, m3, m2, m1 in RUNS:
        _cfg(model, m3, m2, m1)
        print('=== %s %s/%s/%s ===' % (name, m3, m2, m1))
        for i, batch in enumerate(loader):
            fname = filenames[i]
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)
            sr = _tiled(model, model.args, lr, lrsr, ref, refsr)
            _, ssim, mse, psnr = calc_psnr_and_ssim(sr, hr)
            row = {
                'idx': i, 'filename': fname, 'run': name,
                'mode_lv3': m3, 'mode_lv2': m2, 'mode_lv1': m1,
                'psnr_rgb': float(psnr), 'ssim_rgb': float(ssim),
                'mse_rgb': float(mse),
                'luma_mae': _luma_mae(sr, hr),
                'luma_bias': _luma_bias(sr, hr),
                'chroma_mae': _chroma_mae(sr, hr),
            }
            if fname in a0_map:
                row['a0_psnr'] = a0_map[fname]
                row['delta_vs_a0'] = float(psnr) - a0_map[fname]
            rows.append(row)
            if fname in FOCUS:
                focus[fname].setdefault('hr', hr.cpu())
                focus[fname].setdefault('low', lr.cpu())
                focus[fname].setdefault('ref', ref.cpu())
                focus[fname][name] = sr.cpu()
            print('[%s %02d] %s %.3f' % (name, i, fname, psnr))
            torch.cuda.empty_cache()

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, 'per_image_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    by = {}
    for r in rows:
        by.setdefault(r['run'], []).append(r)
    p0 = {r['filename']: r for r in by['P0']}
    p1 = {r['filename']: r for r in by['P1']}
    p2 = {r['filename']: r for r in by['P2']}
    p3 = {r['filename']: r for r in by['P3']}
    names = [r['filename'] for r in by['P0']]

    summary = {}
    for name, lst in by.items():
        summary[name] = {
            'psnr_mean': float(np.mean([r['psnr_rgb'] for r in lst])),
            'ssim_mean': float(np.mean([r['ssim_rgb'] for r in lst])),
            'luma_mae': float(np.mean([r['luma_mae'] for r in lst])),
            'luma_bias': float(np.mean([r['luma_bias'] for r in lst])),
            'chroma_mae': float(np.mean([r['chroma_mae'] for r in lst])),
            'mean_delta_vs_a0': float(np.mean([
                r.get('delta_vs_a0', float('nan')) for r in lst
            ])),
        }

    reg = {}
    for k, (tp, ts) in REGRESSION.items():
        got = summary[k]['psnr_mean']
        reg[k] = {
            'target_psnr': tp, 'got_psnr': got,
            'psnr_ok': abs(got - tp) < 0.02,
            'target_ssim': ts, 'got_ssim': summary[k]['ssim_mean'],
            'ssim_ok': True if ts is None else abs(
                summary[k]['ssim_mean'] - ts) < 0.001,
        }

    # Paired comparisons
    def delta(a, b):
        return [a[f]['psnr_rgb'] - b[f]['psnr_rgb'] for f in names]

    comparisons = {
        'P1_minus_P0': _paired_stats(delta(p1, p0), names),
        'P2_minus_P0': _paired_stats(delta(p2, p0), names),
        'P3_minus_P2': _paired_stats(delta(p3, p2), names),
        'P3_minus_P1': _paired_stats(delta(p3, p1), names),
        'P3_minus_P0': _paired_stats(delta(p3, p0), names),
    }

    # Interaction I = P3 - P1 - P2 + P0 (per image and mean)
    I = []
    for f in names:
        I.append(p3[f]['psnr_rgb'] - p1[f]['psnr_rgb']
                 - p2[f]['psnr_rgb'] + p0[f]['psnr_rgb'])
    I = np.asarray(I, float)
    interaction = {
        'formula': 'I = PSNR(P3)-PSNR(P1)-PSNR(P2)+PSNR(P0)',
        'mean_I': float(I.mean()),
        'median_I': float(np.median(I)),
        'std_I': float(I.std(ddof=0)),
        'bootstrap_I': _bootstrap(I),
        'largest_positive': [
            (names[i], float(I[i])) for i in np.argsort(-I)[:3]],
        'largest_negative': [
            (names[i], float(I[i])) for i in np.argsort(I)[:3]],
        'per_image': {names[i]: float(I[i]) for i in range(len(names))},
        'mean_PSNR_check': {
            'P0': summary['P0']['psnr_mean'],
            'P1': summary['P1']['psnr_mean'],
            'P2': summary['P2']['psnr_mean'],
            'P3': summary['P3']['psnr_mean'],
            'I_from_means': (
                summary['P3']['psnr_mean'] - summary['P1']['psnr_mean']
                - summary['P2']['psnr_mean'] + summary['P0']['psnr_mean']),
        },
    }

    # Decision helper vs P2
    p3_p2 = comparisons['P3_minus_P2']
    decision_notes = {
        'P3_minus_P2_mean': p3_p2['mean'],
        'P3_minus_P2_median': p3_p2['median'],
        'threshold_A_mean': 0.03,
        'threshold_C': 0.02,
    }

    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump({
            'summary': summary, 'regression': reg,
            'comparisons': comparisons,
            'a0_source': a0_src, 'a0_baseline_mean': A0_PSNR,
            'decision_notes': decision_notes,
        }, f, indent=2)
    with open(os.path.join(out, 'interaction_analysis.json'), 'w') as f:
        json.dump(interaction, f, indent=2)
    with open(os.path.join(out, 'robustness_analysis.json'), 'w') as f:
        json.dump({
            'P3_minus_P2': p3_p2,
            'P3_minus_P0': comparisons['P3_minus_P0'],
            'P1_minus_P0': comparisons['P1_minus_P0'],
            'P2_minus_P0': comparisons['P2_minus_P0'],
        }, f, indent=2)

    # Focus viz
    for fn in FOCUS:
        b = focus[fn]
        if 'P0' not in b:
            continue
        names_r = [r[0] for r in RUNS]
        n = len(names_r)
        fig = plt.figure(figsize=(2.8 * max(n, 3), 9))
        hr = b['hr']
        errs = [_err(b[r], hr) for r in names_r]
        vmax = max(0.05, float(np.percentile(
            np.concatenate([e.ravel() for e in errs]), 99)))
        ax = fig.add_subplot(4, max(n, 3), 1)
        ax.imshow(_to_u8(b['low'])); ax.set_title('Low'); ax.axis('off')
        ax = fig.add_subplot(4, max(n, 3), 2)
        ax.imshow(_to_u8(b['ref'])); ax.set_title('Nano'); ax.axis('off')
        ax = fig.add_subplot(4, max(n, 3), 3)
        ax.imshow(_to_u8(hr)); ax.set_title('GT'); ax.axis('off')
        for i, r in enumerate(names_r):
            ax = fig.add_subplot(4, n, n + i + 1)
            ax.imshow(_to_u8(b[r])); ax.set_title(r); ax.axis('off')
        for i, r in enumerate(names_r):
            ax = fig.add_subplot(4, n, 2 * n + i + 1)
            im = ax.imshow(errs[i], vmin=0, vmax=vmax, cmap='magma')
            ax.set_title('|RGB| ' + r, fontsize=8); ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
        # luma residual row
        lmaps = [_luma_err_map(b[r], hr) for r in names_r]
        lv = max(0.05, float(np.percentile(np.abs(np.concatenate(
            [m.ravel() for m in lmaps])), 99)))
        for i, r in enumerate(names_r):
            ax = fig.add_subplot(4, n, 3 * n + i + 1)
            im = ax.imshow(lmaps[i], vmin=-lv, vmax=lv, cmap='coolwarm')
            ax.set_title('ΔY ' + r, fontsize=8); ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
        # caption metrics
        tbl = '  '.join(
            '%s:%.2f' % (r, next(x['psnr_rgb'] for x in rows
                                 if x['filename'] == fn and x['run'] == r))
            for r in names_r)
        fig.suptitle('%s | %s' % (fn, tbl), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'focus',
                                 '%s_interaction.png' %
                                 os.path.splitext(fn)[0]), dpi=120)
        plt.close(fig)

    print(json.dumps({
        'summary': summary, 'regression': reg,
        'P3_minus_P2': p3_p2,
        'interaction_mean': interaction['mean_I'],
    }, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
