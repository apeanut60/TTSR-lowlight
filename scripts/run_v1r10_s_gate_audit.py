#!/usr/bin/env python3
"""v1r10 S-gate calibration audit — G0–G4 zero-shot counterfactuals on M1 T."""
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

from dataset.data1 import TestSet, _collect_pairs  # noqa: E402
from model import TTSREnhance  # noqa: E402
from model.MainNetEnhance import MainNetEnhance  # noqa: E402
from model.query_illum_align import (  # noqa: E402
    estimate_query_align_maps, _rgb_luma01,
)
from utils import calc_psnr_and_ssim  # noqa: E402

CKPT = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt')
FOCUS = ['2037.jpg', '2041.jpg', '2053.jpg', '2056.jpg', '2062.jpg', '2063.jpg']
M1_TARGET = (20.181, 0.6327)

# name, s_source, gate_mode, gate_shift
RUNS = [
    ('G0', 'a0', None, None),
    ('G1', 'b2', None, None),
    ('G2', 'a0', 'constant_mean', None),
    ('G3', 'a0', 'smooth', None),
    ('G4a', 'a0', 'shift', (32, 0)),
    ('G4b', 'a0', 'shift', (0, 32)),
    ('G4c', 'a0', 'shift', (24, 24)),
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
        query_illum_align=True, query_illum_alpha=0.75, query_hist_beta=0.50,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=True, texture_fusion_lambda=0.25,
        texture_fusion_s_source='a0',
        gate_mode=None, gate_shift='', gate_smooth_kernel=31,
        collect_gate_stats=False,
        tile_size=256, tile_overlap=96, tile_window='cosine',
        cpu=False, num_gpu=1, num_workers=0, batch_size=1,
        eval_chroma_gain=1.0, eval_tta=False,
    )


def _load_model(args, device):
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
    print('Loaded %d tensors' % n)
    return model


def _configure(model, s_source, gate_mode, gate_shift):
    model.args.texture_fusion = True
    model.args.texture_fusion_lambda = 0.25
    model.args.texture_fusion_s_source = s_source
    model.args.query_illum_align = True
    model.args.query_illum_alpha = 0.75
    model.args.query_hist_beta = 0.50
    model.args.gate_mode = gate_mode
    model.args.gate_shift = '' if gate_shift is None else '%d,%d' % gate_shift
    model.args.gate_smooth_kernel = 31
    model.args.collect_gate_stats = False


def _as01(x):
    return ((x + 1) * 0.5).clamp(0, 1)


def _luma_mae(pred, gt):
    return float((_rgb_luma01(_as01(pred)) - _rgb_luma01(_as01(gt))).abs().mean())


def _luma_bias(pred, gt):
    return float((_rgb_luma01(_as01(pred)) - _rgb_luma01(_as01(gt))).mean())


def _chroma_mae(pred, gt):
    p, g = _as01(pred), _as01(gt)
    return float(((p - _rgb_luma01(p)) - (g - _rgb_luma01(g))).abs().mean())


def _to_u8(t):
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    return ((x + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def _err_map(pred, gt):
    p = _as01(pred)[0].mean(0).cpu().numpy()  # placeholder unused
    p = _as01(pred)[0]
    g = _as01(gt)[0]
    return (p - g).abs().mean(0).cpu().numpy()


def _s_stats(S):
    s = S.detach().float().view(-1)
    qs = torch.quantile(s, torch.tensor([0.1, 0.5, 0.9], device=s.device))
    return {
        'mean': float(s.mean()),
        'std': float(s.std(unbiased=False)),
        'q10': float(qs[0]), 'q50': float(qs[1]), 'q90': float(qs[2]),
    }


def _gate_stats(G):
    g = G.detach().float()
    flat = g.view(-1)
    qs = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=g.device))
    return {
        'mean': float(flat.mean()),
        'std': float(flat.std(unbiased=False)),
        'cv': float(flat.std(unbiased=False) / flat.mean().clamp(min=1e-8)),
        'q10': float(qs[0]), 'q50': float(qs[1]), 'q90': float(qs[2]),
        'min': float(flat.min()), 'max': float(flat.max()),
    }


@torch.no_grad()
def _tiled_sr(model, args, lr, lr_sr, ref, ref_sr, collect_stats=False,
              return_gates=False):
    tile_size = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile_size // 2))
    fwd = {}
    g, lut, _ = estimate_query_align_maps(
        lr_sr, ref, alpha=args.query_illum_alpha,
        gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
        bins=args.query_hist_bins)
    fwd['query_gamma_override'] = g
    fwd['query_hist_lut_override'] = lut

    gate_mode = args.gate_mode
    gate_shift = args.gate_shift
    if isinstance(gate_shift, str) and gate_shift.strip():
        parts = [int(p) for p in gate_shift.split(',')]
        gate_shift_t = (parts[0], parts[1])
    else:
        gate_shift_t = None

    _, _, H, W = lr.shape
    inj_acc = []
    gate_tiles = []
    gate0_tiles = []
    sa0_stats, sb2_stats = [], []
    ga0_list, gb2_list = [], []

    def _one(y0, y1, x0, x1):
        kw = dict(
            lr=lr[:, :, y0:y1, x0:x1],
            lrsr=lr_sr[:, :, y0:y1, x0:x1],
            ref=ref[:, :, y0:y1, x0:x1],
            refsr=ref_sr[:, :, y0:y1, x0:x1],
            apply_illum=False, apply_ref_illum=False,
            gate_mode=gate_mode, gate_shift=gate_shift_t,
            gate_smooth_kernel=31,
            collect_gate_stats=collect_stats,
            **fwd)
        out = model(**kw)[0]
        info = model.last_texture_fusion_info
        if info is not None:
            sa0_stats.append(_s_stats(info['S_a0']))
            sb2_stats.append(_s_stats(info['S_b2']))
            # Build native A0/B2 gates at tile resolution for Phase A
            th, tw = y1 - y0, x1 - x0
            Ga = MainNetEnhance.build_gate_from_S(info['S_a0'], (th, tw))
            Gb = MainNetEnhance.build_gate_from_S(info['S_b2'], (th, tw))
            ga0_list.append(Ga.cpu())
            gb2_list.append(Gb.cpu())
            if return_gates:
                gate_tiles.append(model.MainNet.last_gate.cpu())
                gate0_tiles.append(model.MainNet.last_gate_pre_transform.cpu())
        if collect_stats and model.MainNet.last_gate_stats is not None:
            inj_acc.append(model.MainNet.last_gate_stats)
        return out

    if H <= tile_size and W <= tile_size:
        sr = _one(0, H, 0, W)
        # Apply illum heads once
        sr = model.apply_ref_illum(sr, lr, ref)
        sr = model.apply_illum_head(sr, lr)
    else:
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
                tile = _one(y0, y0 + tile_size, x0, x0 + tile_size)
                sr_accum[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += (
                    tile * feather)
                weight[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += feather
        sr = sr_accum / weight.clamp(min=1e-8)
        sr = model.apply_ref_illum(sr, lr, ref)
        sr = model.apply_illum_head(sr, lr)

    extras = {
        'inj': inj_acc,
        'S_a0_stats': sa0_stats,
        'S_b2_stats': sb2_stats,
        'G_a0_tiles': ga0_list,
        'G_b2_tiles': gb2_list,
        'G_used_tiles': gate_tiles,
        'G0_tiles': gate0_tiles,
    }
    return sr, extras


def _mean_dict(dicts, keys):
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def _bootstrap_ci(deltas, n_boot=5000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(deltas, dtype=np.float64)
    if len(d) == 0:
        return None
    means = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(d), size=len(d))
        means.append(d[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        'mean': float(d.mean()),
        'median': float(np.median(d)),
        'std': float(d.std(ddof=0)),
        'ci95': [float(lo), float(hi)],
        'n_boot': n_boot,
        'note': 'exploratory; images reused across design',
    }


def _contact(out_path, fname, bundle, run_names):
    # bundle: run -> {sr, gates..., hr, low, ref}
    n = len(run_names)
    fig = plt.figure(figsize=(2.8 * max(n, 3), 12))
    hr = bundle['hr']
    errs = [_err_map(bundle[r]['sr'], hr) for r in run_names]
    vmax = max(0.05, float(np.percentile(
        np.concatenate([e.ravel() for e in errs]), 99)))

    ax = fig.add_subplot(4, max(n, 3), 1)
    ax.imshow(_to_u8(bundle['low'])); ax.set_title('Low'); ax.axis('off')
    ax = fig.add_subplot(4, max(n, 3), 2)
    ax.imshow(_to_u8(bundle['ref'])); ax.set_title('Nano'); ax.axis('off')
    ax = fig.add_subplot(4, max(n, 3), 3)
    ax.imshow(_to_u8(hr)); ax.set_title('GT'); ax.axis('off')

    for i, r in enumerate(run_names):
        ax = fig.add_subplot(4, n, n + i + 1)
        ax.imshow(_to_u8(bundle[r]['sr'])); ax.set_title(r); ax.axis('off')
    for i, r in enumerate(run_names):
        ax = fig.add_subplot(4, n, 2 * n + i + 1)
        im = ax.imshow(errs[i], vmin=0, vmax=vmax, cmap='magma')
        ax.set_title('|err| ' + r, fontsize=8); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Gate row
    gate_keys = [
        ('G_a0', 'A0 gate'), ('G_b2', 'B2 gate'),
        ('G_smooth', 'Smooth'), ('G_shift', 'Shift G4a'),
    ]
    for i, (k, title) in enumerate(gate_keys):
        ax = fig.add_subplot(4, 4, 12 + i + 1) if False else fig.add_subplot(
            4, max(n, 4), 3 * max(n, 4) + i + 1)
        g = bundle.get(k)
        if g is None:
            ax.axis('off'); continue
        im = ax.imshow(g, vmin=0, vmax=1, cmap='viridis')
        ax.set_title(title, fontsize=8); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'diagnostics', 'v1r10'))
    ap.add_argument('--runs', default='G0,G1,G2,G3,G4a,G4b,G4c')
    cli = ap.parse_args()
    out = cli.out_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'focus'), exist_ok=True)
    os.makedirs(os.path.join(out, 'stage_maps'), exist_ok=True)

    want = set(cli.runs.split(','))
    runs = [r for r in RUNS if r[0] in want]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = _build_args()
    model = _load_model(args, device)

    audit = {
        'gate_formula': "G = sigmoid(bicubic_upsample(S, MainNet_HW))",
        'same_G_all_stages': True,
        'injection_order': [
            'concat(stream, T_up)',
            'conv*_head → residual',
            'residual *= G',
            'stream += gated residual',
            'RB / CSFI / merge / illum',
        ],
        'smooth_kernel': 31,
        'g4_shifts': {'G4a': [32, 0], 'G4b': [0, 32], 'G4c': [24, 24]},
        'm1_frozen': {
            'alpha': 0.75, 'beta': 0.50, 'lambda': 0.25, 's_source_G0': 'a0',
        },
    }
    with open(os.path.join(out, 'code_audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    pairs = _collect_pairs(os.path.join(args.dataset_dir, 'Eval'),
                           camera_filter='Huawei')
    filenames = [os.path.basename(p[0]) for p in pairs]
    ds = TestSet(args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    rows = []
    focus_store = {fn: {'low': None, 'ref': None, 'hr': None} for fn in FOCUS}
    gate_dist_rows = []

    for run_name, s_src, gmode, gshift in runs:
        _configure(model, s_src, gmode, gshift)
        print('=== %s s=%s gate=%s shift=%s ===' % (
            run_name, s_src, gmode, gshift))
        for i, batch in enumerate(loader):
            fname = filenames[i]
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)

            need_stats = (run_name == 'G0') or (fname in FOCUS)
            sr, extras = _tiled_sr(
                model, model.args, lr, lrsr, ref, refsr,
                collect_stats=need_stats,
                return_gates=(fname in FOCUS))
            _, ssim, mse, psnr = calc_psnr_and_ssim(sr, hr)

            # Aggregate S / G stats across tiles
            sa = _mean_dict(extras['S_a0_stats'],
                            ['mean', 'std', 'q10', 'q50', 'q90'])
            sb = _mean_dict(extras['S_b2_stats'],
                            ['mean', 'std', 'q10', 'q50', 'q90'])
            # Gate A0/B2 from tiles (pre-transform)
            ga_stats, gb_stats, corr, mad = None, None, None, None
            if extras['G_a0_tiles']:
                # Use first tile as representative + mean of per-tile stats
                ga_list = [_gate_stats(t) for t in extras['G_a0_tiles']]
                gb_list = [_gate_stats(t) for t in extras['G_b2_tiles']]
                ga_stats = _mean_dict(
                    ga_list, ['mean', 'std', 'cv', 'q10', 'q50', 'q90'])
                gb_stats = _mean_dict(
                    gb_list, ['mean', 'std', 'cv', 'q10', 'q50', 'q90'])
                corrs, mads = [], []
                for ga, gb in zip(extras['G_a0_tiles'], extras['G_b2_tiles']):
                    a = ga.float().flatten()
                    b = gb.float().flatten()
                    a0 = a - a.mean(); b0 = b - b.mean()
                    denom = (a0.norm() * b0.norm()).clamp(min=1e-8)
                    corrs.append(float((a0 * b0).sum() / denom))
                    mads.append(float((a - b).abs().mean()))
                corr = float(np.mean(corrs))
                mad = float(np.mean(mads))

            inj = {}
            if extras['inj']:
                tags = [s['stage'] for s in extras['inj'][0]['stages']]
                for tag in tags:
                    rels, pres, posts = [], [], []
                    for t in extras['inj']:
                        for st in t['stages']:
                            if st['stage'] == tag:
                                rels.append(st['rel_inject'])
                                pres.append(st['res_norm_pre_mean'])
                                posts.append(st['res_norm_post_mean'])
                    inj[tag + '_rel'] = float(np.mean(rels))
                    inj[tag + '_pre'] = float(np.mean(pres))
                    inj[tag + '_post'] = float(np.mean(posts))

            row = {
                'idx': i, 'filename': fname, 'run': run_name,
                'psnr_rgb': float(psnr), 'ssim_rgb': float(ssim),
                'mse_rgb': float(mse),
                'luma_mae': _luma_mae(sr, hr),
                'luma_bias': _luma_bias(sr, hr),
                'chroma_mae': _chroma_mae(sr, hr),
                's_source': s_src,
                'gate_mode': gmode or 'default',
                'gate_shift': '' if gshift is None else '%d,%d' % gshift,
                'S_a0_mean': sa['mean'], 'S_a0_std': sa['std'],
                'S_b2_mean': sb['mean'], 'S_b2_std': sb['std'],
                'G_a0_mean': None if not ga_stats else ga_stats['mean'],
                'G_a0_std': None if not ga_stats else ga_stats['std'],
                'G_a0_cv': None if not ga_stats else ga_stats['cv'],
                'G_b2_mean': None if not gb_stats else gb_stats['mean'],
                'G_b2_std': None if not gb_stats else gb_stats['std'],
                'G_b2_cv': None if not gb_stats else gb_stats['cv'],
                'G_a0_b2_corr': corr,
                'G_a0_b2_mad': mad,
            }
            row.update(inj)
            rows.append(row)

            if extras['G_a0_tiles'] and run_name == 'G0':
                gate_dist_rows.append({
                    'filename': fname,
                    'S_a0': sa, 'S_b2': sb,
                    'G_a0': ga_stats, 'G_b2': gb_stats,
                    'corr': corr, 'mad': mad,
                    'sigmoid_note': 'G from sigmoid(upsample(S)), not sigmoid(mean S)',
                })

            if fname in FOCUS:
                focus_store[fname]['low'] = lr.cpu()
                focus_store[fname]['ref'] = ref.cpu()
                focus_store[fname]['hr'] = hr.cpu()
                focus_store[fname][run_name] = {
                    'sr': sr.cpu(),
                }
                # Store gates from first tile of G0/G1/G3/G4a
                if extras['G_a0_tiles']:
                    focus_store[fname]['G_a0'] = extras['G_a0_tiles'][0][0, 0].numpy()
                    focus_store[fname]['G_b2'] = extras['G_b2_tiles'][0][0, 0].numpy()
                if extras['G_used_tiles'] and run_name in (
                        'G0', 'G3', 'G4a', 'G2'):
                    focus_store[fname]['G_used_' + run_name] = (
                        extras['G_used_tiles'][0][0, 0].numpy())
                if extras['inj'] and run_name == 'G0':
                    # stage maps from first tile
                    sm = extras['inj'][0].get('stage_maps', {})
                    for tag, mp in sm.items():
                        fig, ax = plt.subplots(figsize=(4, 3))
                        im = ax.imshow(mp.numpy(), cmap='magma')
                        ax.set_title('%s %s |res| post-gate' % (fname, tag))
                        fig.colorbar(im, ax=ax, fraction=0.046)
                        fig.tight_layout()
                        fig.savefig(os.path.join(
                            out, 'stage_maps',
                            '%s_%s.png' % (os.path.splitext(fname)[0], tag)),
                            dpi=110)
                        plt.close(fig)

            print('[%s %02d] %s PSNR=%.3f' % (run_name, i, fname, psnr))
            torch.cuda.empty_cache()

    # CSV
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, 'per_image_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(out, 'gate_distribution_stats.json'), 'w') as f:
        json.dump(gate_dist_rows, f, indent=2)

    # Summaries + paired vs G0
    by_run = {}
    for r in rows:
        by_run.setdefault(r['run'], []).append(r)
    g0 = {r['filename']: r for r in by_run.get('G0', [])}
    summary = {}
    paired = {}
    for name, lst in by_run.items():
        psnrs = [r['psnr_rgb'] for r in lst]
        ssims = [r['ssim_rgb'] for r in lst]
        d = [r['psnr_rgb'] - g0[r['filename']]['psnr_rgb']
             for r in lst if r['filename'] in g0]
        d_sorted = sorted(d)
        worst = sorted(
            [(r['filename'], r['psnr_rgb'] - g0[r['filename']]['psnr_rgb'])
             for r in lst if r['filename'] in g0], key=lambda x: x[1])[:3]
        best = sorted(
            [(r['filename'], r['psnr_rgb'] - g0[r['filename']]['psnr_rgb'])
             for r in lst if r['filename'] in g0], key=lambda x: -x[1])[:3]
        summary[name] = {
            'psnr_mean': float(np.mean(psnrs)),
            'ssim_mean': float(np.mean(ssims)),
            'luma_mae': float(np.mean([r['luma_mae'] for r in lst])),
            'luma_bias': float(np.mean([r['luma_bias'] for r in lst])),
            'chroma_mae': float(np.mean([r['chroma_mae'] for r in lst])),
            'delta_vs_G0_mean': float(np.mean(d)) if d else 0.0,
            'delta_vs_G0_median': float(np.median(d)) if d else 0.0,
            'n_up': int(sum(1 for x in d if x > 0.01)),
            'n_down': int(sum(1 for x in d if x < -0.01)),
            'worst3': worst, 'best3': best,
        }
        paired[name] = _bootstrap_ci(d)
    # G4 aggregate
    if all(k in by_run for k in ('G4a', 'G4b', 'G4c')):
        g4_means = [summary[k]['psnr_mean'] for k in ('G4a', 'G4b', 'G4c')]
        g4_deltas = [summary[k]['delta_vs_G0_mean'] for k in ('G4a', 'G4b', 'G4c')]
        summary['G4_aggregate'] = {
            'psnr_mean_of_runs': float(np.mean(g4_means)),
            'psnr_std_across_shifts': float(np.std(g4_means)),
            'delta_mean_of_runs': float(np.mean(g4_deltas)),
            'delta_std_across_shifts': float(np.std(g4_deltas)),
            'runs': {k: summary[k]['psnr_mean'] for k in ('G4a', 'G4b', 'G4c')},
        }

    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'paired_bootstrap': paired,
                   'G0_vs_m1_target': {
                       'target_psnr': M1_TARGET[0],
                       'target_ssim': M1_TARGET[1],
                       'G0_psnr': summary.get('G0', {}).get('psnr_mean'),
                       'G0_ssim': summary.get('G0', {}).get('ssim_mean'),
                       'psnr_ok': abs(summary.get('G0', {}).get('psnr_mean', 0)
                                     - M1_TARGET[0]) < 0.02,
                   }}, f, indent=2)

    # Focus contact sheets
    sheet_runs = [r[0] for r in runs]
    for fn in FOCUS:
        b = focus_store[fn]
        if b.get('G0') is None:
            continue
        # attach gate maps
        if 'G_used_G3' in b:
            b['G_smooth'] = b['G_used_G3']
        if 'G_used_G4a' in b:
            b['G_shift'] = b['G_used_G4a']
        # rebuild run dicts for contact
        pack = {'low': b['low'], 'ref': b['ref'], 'hr': b['hr'],
                'G_a0': b.get('G_a0'), 'G_b2': b.get('G_b2'),
                'G_smooth': b.get('G_smooth'), 'G_shift': b.get('G_shift')}
        for r in sheet_runs:
            if r in b and isinstance(b[r], dict):
                pack[r] = b[r]
        try:
            _contact(os.path.join(out, 'focus',
                                  '%s_gates.png' % os.path.splitext(fn)[0]),
                     fn, pack, [r for r in sheet_runs if r in pack])
        except Exception as e:
            print('focus viz fail', fn, e)

    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
