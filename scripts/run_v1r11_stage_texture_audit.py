#!/usr/bin/env python3
"""v1r11 stage-wise texture injection audit on frozen v1r9 M1."""
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
FOCUS = ['2037.jpg', '2053.jpg', '2041.jpg', '2063.jpg']
S0_TARGET = 20.181
THRESH = 0.05  # dB to trigger S4–S9

# name: (disable_lv3, disable_lv2, disable_lv1)  — True = zero that T
RUNS_PHASE2 = [
    ('S0', False, False, False),  # all on
    ('S1', True, False, False),   # disable lv3
    ('S2', False, True, False),   # disable lv2
    ('S3', False, False, True),   # disable lv1
]
RUNS_PHASE3 = [
    ('S4', False, True, True),    # lv3 only
    ('S5', True, False, True),    # lv2 only
    ('S6', True, True, False),    # lv1 only
    ('S7', False, False, True),   # lv3+lv2  (= disable lv1) — same as S3!
    ('S8', False, True, False),   # lv3+lv1 (= disable lv2) — same as S2!
    ('S9', True, False, False),   # lv2+lv1 (= disable lv3) — same as S1!
]
# Note: S7=S3, S8=S2, S9=S1 under binary enable/disable. Phase3 "only" runs
# S4/S5/S6 are the informative new ablations; S7–S9 duplicate S3/S2/S1 and are
# reported by alias for completeness per the plan naming.


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
        collect_gate_stats=False,
        disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
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


def _cfg(model, d3, d2, d1):
    model.args.disable_T_lv3 = d3
    model.args.disable_T_lv2 = d2
    model.args.disable_T_lv1 = d1
    model.args.texture_fusion = True
    model.args.texture_fusion_lambda = 0.25
    model.args.texture_fusion_s_source = 'a0'
    model.args.query_illum_align = True
    model.args.query_illum_alpha = 0.75
    model.args.query_hist_beta = 0.50


def _as01(x):
    return ((x + 1) * 0.5).clamp(0, 1)


def _luma_mae(p, g):
    return float((_rgb_luma01(_as01(p)) - _rgb_luma01(_as01(g))).abs().mean())


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


@torch.no_grad()
def _tiled(model, args, lr, lrsr, ref, refsr, collect=False):
    tile = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile // 2))
    g, lut, _ = estimate_query_align_maps(
        lrsr, ref, alpha=args.query_illum_alpha,
        gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
        bins=args.query_hist_bins)
    fwd = dict(query_gamma_override=g, query_hist_lut_override=lut,
               disable_T_lv3=args.disable_T_lv3,
               disable_T_lv2=args.disable_T_lv2,
               disable_T_lv1=args.disable_T_lv1,
               collect_gate_stats=collect)
    _, _, H, W = lr.shape
    inj = []

    def one(y0, y1, x0, x1):
        out = model(
            lr=lr[:, :, y0:y1, x0:x1],
            lrsr=lrsr[:, :, y0:y1, x0:x1],
            ref=ref[:, :, y0:y1, x0:x1],
            refsr=refsr[:, :, y0:y1, x0:x1],
            apply_illum=False, apply_ref_illum=False, **fwd)[0]
        if collect and model.MainNet.last_gate_stats is not None:
            inj.append(model.MainNet.last_gate_stats)
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
    return sr, inj


def _run_set(model, loader, filenames, runs, device, out, focus_store, rows):
    for name, d3, d2, d1 in runs:
        _cfg(model, d3, d2, d1)
        print('=== %s disable lv3/2/1 = %s/%s/%s ===' % (name, d3, d2, d1))
        for i, batch in enumerate(loader):
            fname = filenames[i]
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)
            need = (name == 'S0') or (fname in FOCUS)
            sr, inj = _tiled(model, model.args, lr, lrsr, ref, refsr,
                             collect=need)
            _, ssim, mse, psnr = calc_psnr_and_ssim(sr, hr)
            row = {
                'idx': i, 'filename': fname, 'run': name,
                'disable_lv3': int(d3), 'disable_lv2': int(d2),
                'disable_lv1': int(d1),
                'psnr_rgb': float(psnr), 'ssim_rgb': float(ssim),
                'mse_rgb': float(mse),
                'luma_mae': _luma_mae(sr, hr),
                'chroma_mae': _chroma_mae(sr, hr),
            }
            if inj:
                tags = [s['stage'] for s in inj[0]['stages']]
                for tag in tags:
                    rels, pres, posts = [], [], []
                    for t in inj:
                        for st in t['stages']:
                            if st['stage'] == tag:
                                rels.append(st['rel_inject'])
                                pres.append(st['res_norm_pre_mean'])
                                posts.append(st['res_norm_post_mean'])
                    row[tag + '_pre'] = float(np.mean(pres))
                    row[tag + '_post'] = float(np.mean(posts))
                    row[tag + '_rel'] = float(np.mean(rels))
                if fname in FOCUS and name in ('S0', 'S1', 'S2', 'S3'):
                    sm = inj[0].get('stage_maps', {})
                    for tag, mp in sm.items():
                        fig, ax = plt.subplots(figsize=(4, 3))
                        im = ax.imshow(mp.numpy(), cmap='magma')
                        ax.set_title('%s %s |res| post' % (fname, tag))
                        fig.colorbar(im, ax=ax, fraction=0.046)
                        fig.tight_layout()
                        fig.savefig(os.path.join(
                            out, 'stage_maps',
                            '%s_%s_%s.png' % (
                                os.path.splitext(fname)[0], name, tag)),
                            dpi=110)
                        plt.close(fig)
            rows.append(row)
            if fname in FOCUS:
                focus_store[fname].setdefault('hr', hr.cpu())
                focus_store[fname].setdefault('low', lr.cpu())
                focus_store[fname].setdefault('ref', ref.cpu())
                focus_store[fname][name] = sr.cpu()
            print('[%s %02d] %s %.3f' % (name, i, fname, psnr))
            torch.cuda.empty_cache()
    return rows


def _summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r['run'], []).append(r)
    s0 = {r['filename']: r for r in by.get('S0', [])}
    summary = {}
    for name, lst in by.items():
        d = [r['psnr_rgb'] - s0[r['filename']]['psnr_rgb']
             for r in lst if r['filename'] in s0]
        summary[name] = {
            'psnr_mean': float(np.mean([r['psnr_rgb'] for r in lst])),
            'ssim_mean': float(np.mean([r['ssim_rgb'] for r in lst])),
            'luma_mae': float(np.mean([r['luma_mae'] for r in lst])),
            'chroma_mae': float(np.mean([r['chroma_mae'] for r in lst])),
            'delta_vs_S0': float(np.mean(d)) if d else 0.0,
            'n_up': int(sum(1 for x in d if x > 0.01)),
            'n_down': int(sum(1 for x in d if x < -0.01)),
        }
    return summary


def _focus_viz(out, focus_store, run_names):
    for fn in FOCUS:
        b = focus_store[fn]
        if 'S0' not in b:
            continue
        names = [r for r in run_names if r in b]
        n = len(names)
        fig = plt.figure(figsize=(2.6 * max(n, 3), 7.5))
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
            ax.set_title('|err| ' + r, fontsize=8); ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(fn)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'focus',
                                 '%s_stages.png' % os.path.splitext(fn)[0]),
                    dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'diagnostics', 'v1r11'))
    ap.add_argument('--phase2_only', action='store_true')
    cli = ap.parse_args()
    out = cli.out_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'focus'), exist_ok=True)
    os.makedirs(os.path.join(out, 'stage_maps'), exist_ok=True)

    audit = {
        'stages': [
            {
                'name': 'lv3 / stage1',
                'stream': 'x11 from SFE',
                'feature_shape': '[N, n_feats, H, W] (H,W = Low spatial)',
                'texture_native': 'T_lv3 [N, C_vgg, H/4, W/4]',
                'texture_up': 'bicubic → [N, C_vgg, H, W]',
                'residual_head': 'conv11_head: concat(x11, T_lv3_up) → n_feats',
                'gate': 'same G = sigmoid(upsample(S)) multiply residual',
                'add': 'x11 = x11 + gated_residual',
            },
            {
                'name': 'lv2 / stage2',
                'stream': 'x22 from SFE (parallel)',
                'texture_native': 'T_lv2 [N, C_vgg, H/2, W/2]',
                'residual_head': 'conv22_head',
                'gate': 'same G',
                'add': 'x22 = x22 + gated_residual',
            },
            {
                'name': 'lv1 / stage3',
                'stream': 'x33 from SFE (parallel)',
                'texture_native': 'T_lv1 [N, C_vgg, H, W]',
                'residual_head': 'conv33_head',
                'gate': 'same G',
                'add': 'x33 = x33 + gated_residual',
            },
        ],
        'intervention': 'zero T_*_up before concat/residual (disable_T_lv*)',
        'frozen_m1': {
            'alpha': 0.75, 'beta': 0.50, 'lambda': 0.25, 's': 'A0',
            'ckpt': 'v1r3e4',
        },
        'note_S7_S9': 'Under binary stage enable, S7≡S3, S8≡S2, S9≡S1; '
                      'S4/S5/S6 are single-stage-only runs.',
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

    rows = []
    focus_store = {fn: {} for fn in FOCUS}
    rows = _run_set(model, loader, filenames, RUNS_PHASE2, device, out,
                    focus_store, rows)
    summary = _summarize(rows)
    print('Phase2 summary', json.dumps(summary, indent=2))

    # Trigger Phase 3 if any single-stage disable |Δ| > 0.05
    trigger = any(
        abs(summary[k]['delta_vs_S0']) > THRESH for k in ('S1', 'S2', 'S3'))
    phase3_ran = False
    if trigger and not cli.phase2_only:
        print('Phase3 triggered (|Δ|>%.2f dB)' % THRESH)
        # Run informative S4–S6; alias S7–S9 from S3/S2/S1 without recompute
        rows = _run_set(model, loader, filenames, RUNS_PHASE3[:3], device, out,
                        focus_store, rows)
        # Aliases
        alias = {'S7': 'S3', 'S8': 'S2', 'S9': 'S1'}
        by = {}
        for r in rows:
            by.setdefault(r['run'], []).append(r)
        for new, old in alias.items():
            for r in by[old]:
                rr = dict(r)
                rr['run'] = new
                rr['alias_of'] = old
                rows.append(rr)
        phase3_ran = True
        summary = _summarize(rows)
    else:
        print('Phase3 skipped (no single-stage |Δ|>%.2f or --phase2_only)' %
              THRESH)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, 'per_image_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    # Contribution approx: S0 - S_disable_k ≈ contribution of stage k
    contrib = {}
    if 'S0' in summary:
        for k, stage in [('S1', 'lv3'), ('S2', 'lv2'), ('S3', 'lv1')]:
            if k in summary:
                # Dropping stage → Δ; positive contrib if dropping hurts (Δ<0)
                contrib[stage] = {
                    'delta_when_disabled': summary[k]['delta_vs_S0'],
                    'approx_contrib_to_S0': -summary[k]['delta_vs_S0'],
                }
    payload = {
        'summary': summary,
        'contribution': contrib,
        'phase3_ran': phase3_ran,
        'S0_vs_target': {
            'target': S0_TARGET,
            'S0': summary.get('S0', {}).get('psnr_mean'),
            'ok': abs(summary.get('S0', {}).get('psnr_mean', 0) - S0_TARGET) < 0.02,
        },
    }
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    run_names = [r[0] for r in RUNS_PHASE2]
    if phase3_ran:
        run_names += ['S4', 'S5', 'S6']
    _focus_viz(out, focus_store, run_names)
    print(json.dumps(payload, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
