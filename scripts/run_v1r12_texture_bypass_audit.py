#!/usr/bin/env python3
"""v1r12: Texture-Zero vs Injection-Bypass audit on frozen v1r9 M1."""
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
FOCUS = ['2037.jpg', '2053.jpg', '2041.jpg', '2063.jpg', '2056.jpg', '2062.jpg']

# run: (mode_lv3, mode_lv2, mode_lv1)
RUNS = [
    ('C0', 'normal', 'normal', 'normal'),
    ('C1', 'texture_zero', 'normal', 'normal'),
    ('C2', 'branch_bypass', 'normal', 'normal'),
    ('C3', 'normal', 'texture_zero', 'normal'),
    ('C4', 'normal', 'branch_bypass', 'normal'),
    ('C5', 'normal', 'normal', 'texture_zero'),
    ('C6', 'normal', 'normal', 'branch_bypass'),
]
REGRESSION = {
    'C0': (20.1805, 0.6327),
    'C1': (20.234, None),
    'C3': (20.212, None),
    'C5': (20.145, None),
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
    model.args.gate_mode = None  # original A0 gate only


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


def _bootstrap(deltas, n_boot=5000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(deltas, float)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        'mean': float(d.mean()), 'median': float(np.median(d)),
        'std': float(d.std(ddof=0)),
        'ci95': [float(lo), float(hi)],
        'note': 'exploratory; Eval reused across design',
    }


@torch.no_grad()
def _tiled(model, args, lr, lrsr, ref, refsr, collect_decomp=False,
           collect_stats=False):
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
        collect_residual_decomp=collect_decomp,
        collect_gate_stats=collect_stats,
    )
    _, _, H, W = lr.shape
    decomp_acc, inj_acc = [], []

    def one(y0, y1, x0, x1):
        out = model(
            lr=lr[:, :, y0:y1, x0:x1],
            lrsr=lrsr[:, :, y0:y1, x0:x1],
            ref=ref[:, :, y0:y1, x0:x1],
            refsr=refsr[:, :, y0:y1, x0:x1],
            apply_illum=False, apply_ref_illum=False, **fwd)[0]
        if collect_decomp and model.MainNet.last_residual_decomp is not None:
            decomp_acc.append({
                'stats': model.MainNet.last_residual_decomp,
                'maps': getattr(model.MainNet, 'last_residual_decomp_maps', {}),
            })
        if collect_stats and model.MainNet.last_gate_stats is not None:
            inj_acc.append(model.MainNet.last_gate_stats)
        return out

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
    return sr, decomp_acc, inj_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'diagnostics', 'v1r12'))
    ap.add_argument('--runs', default='C0,C1,C2,C3,C4,C5,C6')
    cli = ap.parse_args()
    out = cli.out_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'focus'), exist_ok=True)
    os.makedirs(os.path.join(out, 'residual_maps'), exist_ok=True)

    audit = {
        'mapping': [
            {'T': 'T_lv3', 'stage': 1, 'head': 'conv11_head',
             'T_ch': 256, 'stream': 'x11←SFE', 'add': 'x11+=G*R'},
            {'T': 'T_lv2', 'stage': 2, 'head': 'conv22_head',
             'T_ch': 128, 'stream': 'x22←SFE', 'add': 'x22+=G*R'},
            {'T': 'T_lv1', 'stage': 3, 'head': 'conv33_head',
             'T_ch': 64, 'stream': 'x33←SFE', 'add': 'x33+=G*R'},
        ],
        'residual': 'R(F,T)=conv_head(concat(F,T)); gated=R*G; F←F+gated',
        'texture_zero': 'F_zero = F + G*R(F,0)',
        'branch_bypass': 'F_bypass = F  (no conv_head call)',
        'note': 'R(F,0) ≠ 0 due to stream channels + bias in conv_head',
        'gate': 'original A0 sigmoid(upsample(S)); no v1r10 G2',
        'frozen_m1': {'alpha': 0.75, 'beta': 0.50, 'lambda': 0.25},
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

    want = set(cli.runs.split(','))
    runs = [r for r in RUNS if r[0] in want]
    rows = []
    focus = {fn: {} for fn in FOCUS}
    decomp_all = []

    for name, m3, m2, m1 in runs:
        _cfg(model, m3, m2, m1)
        print('=== %s modes=%s/%s/%s ===' % (name, m3, m2, m1))
        for i, batch in enumerate(loader):
            fname = filenames[i]
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)
            do_decomp = (name == 'C0')
            do_stats = (name == 'C0' and fname in FOCUS) or (
                fname in FOCUS and name in ('C1', 'C2'))
            sr, decomp, inj = _tiled(
                model, model.args, lr, lrsr, ref, refsr,
                collect_decomp=do_decomp, collect_stats=do_stats)
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
            if decomp:
                # average scalar decomp over tiles
                for stage_i, tag in enumerate(
                        ['stage1_Tlv3', 'stage2_Tlv2', 'stage3_Tlv1']):
                    keys = [k for k in decomp[0]['stats'][stage_i]
                            if isinstance(decomp[0]['stats'][stage_i][k],
                                          (int, float))]
                    for k in keys:
                        vals = [t['stats'][stage_i][k] for t in decomp]
                        row['%s_%s' % (tag, k)] = float(np.mean(vals))
                if fname in FOCUS:
                    # save maps from first tile
                    maps = decomp[0]['maps']
                    for st, md in maps.items():
                        for mk, mv in md.items():
                            fig, ax = plt.subplots(figsize=(4, 3))
                            im = ax.imshow(mv.numpy(), cmap='magma')
                            ax.set_title('%s %s %s' % (fname, st, mk))
                            fig.colorbar(im, ax=ax, fraction=0.046)
                            fig.tight_layout()
                            fig.savefig(os.path.join(
                                out, 'residual_maps',
                                '%s_%s_%s.png' % (
                                    os.path.splitext(fname)[0], st, mk)),
                                dpi=110)
                            plt.close(fig)
                    decomp_all.append({
                        'filename': fname,
                        'stats': [
                            {k: v for k, v in s.items()
                             if not isinstance(v, torch.Tensor)}
                            for s in decomp[0]['stats']
                        ],
                    })
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
    c0 = {r['filename']: r for r in by.get('C0', [])}
    summary = {}
    paired = {}
    for name, lst in by.items():
        d = [r['psnr_rgb'] - c0[r['filename']]['psnr_rgb']
             for r in lst if r['filename'] in c0]
        worst = sorted(
            [(r['filename'], r['psnr_rgb'] - c0[r['filename']]['psnr_rgb'])
             for r in lst if r['filename'] in c0], key=lambda x: x[1])[:3]
        best = sorted(
            [(r['filename'], r['psnr_rgb'] - c0[r['filename']]['psnr_rgb'])
             for r in lst if r['filename'] in c0], key=lambda x: -x[1])[:3]
        summary[name] = {
            'psnr_mean': float(np.mean([r['psnr_rgb'] for r in lst])),
            'ssim_mean': float(np.mean([r['ssim_rgb'] for r in lst])),
            'luma_mae': float(np.mean([r['luma_mae'] for r in lst])),
            'luma_bias': float(np.mean([r['luma_bias'] for r in lst])),
            'chroma_mae': float(np.mean([r['chroma_mae'] for r in lst])),
            'delta_vs_C0_mean': float(np.mean(d)) if d else 0.0,
            'delta_vs_C0_median': float(np.median(d)) if d else 0.0,
            'n_up': int(sum(1 for x in d if x > 0.01)),
            'n_down': int(sum(1 for x in d if x < -0.01)),
            'worst3': worst, 'best3': best,
        }
        paired[name] = _bootstrap(d) if d else None

    # C1-C2, C3-C4, C5-C6 paired
    contrasts = {}
    for a, b, label in [('C1', 'C2', 'lv3_zero_minus_bypass'),
                        ('C3', 'C4', 'lv2_zero_minus_bypass'),
                        ('C5', 'C6', 'lv1_zero_minus_bypass')]:
        if a in by and b in by:
            ba = {r['filename']: r for r in by[a]}
            bb = {r['filename']: r for r in by[b]}
            d = [ba[f]['psnr_rgb'] - bb[f]['psnr_rgb']
                 for f in ba if f in bb]
            contrasts[label] = {
                'mean_delta_psnr': float(np.mean(d)),
                'median_delta_psnr': float(np.median(d)),
                'bootstrap': _bootstrap(d),
                'interpretation': 'positive ⇒ R(F,0) helpful vs bypass',
            }

    # Regression check
    reg = {}
    for k, (tpsnr, tssim) in REGRESSION.items():
        if k not in summary:
            continue
        ok_p = abs(summary[k]['psnr_mean'] - tpsnr) < 0.02
        ok_s = True if tssim is None else abs(
            summary[k]['ssim_mean'] - tssim) < 0.001
        reg[k] = {
            'target_psnr': tpsnr, 'got_psnr': summary[k]['psnr_mean'],
            'psnr_ok': ok_p,
            'target_ssim': tssim, 'got_ssim': summary[k]['ssim_mean'],
            'ssim_ok': ok_s,
        }

    # Mean decomp from C0 rows
    decomp_mean = {}
    if 'C0' in by:
        for tag in ['stage1_Tlv3', 'stage2_Tlv2', 'stage3_Tlv1']:
            keys = [k for k in by['C0'][0] if k.startswith(tag + '_')]
            decomp_mean[tag] = {
                k[len(tag) + 1:]: float(np.mean([r[k] for r in by['C0']
                                                   if k in r and r[k] == r[k]]))
                for k in keys
            }

    payload = {
        'summary': summary,
        'paired_bootstrap_vs_C0': paired,
        'contrasts_zero_minus_bypass': contrasts,
        'regression': reg,
        'decomp_mean_C0': decomp_mean,
    }
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out, 'residual_decomp_focus.json'), 'w') as f:
        json.dump(decomp_all, f, indent=2)

    # Focus contact: C0/C1/C2 and C3-C6
    for fn in FOCUS:
        b = focus[fn]
        if 'C0' not in b:
            continue
        names = [r[0] for r in runs if r[0] in b]
        n = len(names)
        fig = plt.figure(figsize=(2.5 * max(n, 3), 7.2))
        hr = b['hr']
        errs = [_err(b[r], hr) for r in names]
        vmax = max(0.05, float(np.percentile(
            np.concatenate([e.ravel() for e in errs]), 99)))
        ax = fig.add_subplot(3, max(n, 3), 1)
        ax.imshow(_to_u8(b['low'])); ax.set_title('Low'); ax.axis('off')
        ax = fig.add_subplot(3, max(n, 3), 2)
        ax.imshow(_to_u8(b['ref'])); ax.set_title('Nano'); ax.axis('off')
        ax = fig.add_subplot(3, max(n, 3), 3)
        ax.imshow(_to_u8(hr)); ax.set_title('GT'); ax.axis('off')
        for i, r in enumerate(names):
            ax = fig.add_subplot(3, n, n + i + 1)
            ax.imshow(_to_u8(b[r])); ax.set_title(r); ax.axis('off')
        for i, r in enumerate(names):
            ax = fig.add_subplot(3, n, 2 * n + i + 1)
            im = ax.imshow(errs[i], vmin=0, vmax=vmax, cmap='magma')
            ax.set_title('|err| ' + r, fontsize=7); ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(fn)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'focus',
                                 '%s_zero_vs_bypass.png' %
                                 os.path.splitext(fn)[0]), dpi=120)
        plt.close(fig)

    print(json.dumps(payload, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
