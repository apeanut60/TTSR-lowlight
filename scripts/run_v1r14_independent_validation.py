#!/usr/bin/env python3
"""v1r14: independent / cross-domain validation of A0 vs M1 vs P2."""
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

# Historical Eval30 regression targets (NOT part of independent aggregate)
REG_HUAWEI = {
    'A0': (20.162, 0.6324),
    'M1': (20.1805, 0.6327),
    'P2': (20.264, 0.6341),
}


def _build_args(camera, ref_dir):
    return SimpleNamespace(
        dataset='data1', dataset_dir='/root/data/datasets/data1',
        data1_camera=camera, ref_dir=ref_dir,
        ref_degrade=False, eval_ref_degrade=None, ref_gamma=1.0,
        eval_degrade_seed=1234, enhance_mode=True,
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
        freeze_lte=False, ref_correction=True, ref_correction_feats=16,
        ref_illum_pool=8,
        query_illum_align=False, query_illum_alpha=0.0, query_hist_beta=0.0,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=False, texture_fusion_lambda=0.0,
        texture_fusion_s_source='a0',
        gate_mode=None, gate_shift='', gate_smooth_kernel=31,
        collect_gate_stats=False, collect_residual_decomp=False,
        disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
        stage_mode_lv3='normal', stage_mode_lv2='normal',
        stage_mode_lv1='normal',
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


def _cfg(model, name):
    """A0 / M1 / P2 exactly as frozen in the plan."""
    model.args.gate_mode = None
    model.args.disable_T_lv3 = False
    model.args.disable_T_lv2 = False
    model.args.disable_T_lv1 = False
    model.args.stage_mode_lv3 = 'normal'
    model.args.stage_mode_lv1 = 'normal'
    if name == 'A0':
        model.args.query_illum_align = False
        model.args.query_illum_alpha = 0.0
        model.args.query_hist_beta = 0.0
        model.args.texture_fusion = False
        model.args.texture_fusion_lambda = 0.0
        model.args.stage_mode_lv2 = 'normal'
    elif name == 'M1':
        model.args.query_illum_align = True
        model.args.query_illum_alpha = 0.75
        model.args.query_hist_beta = 0.50
        model.args.texture_fusion = True
        model.args.texture_fusion_lambda = 0.25
        model.args.texture_fusion_s_source = 'a0'
        model.args.stage_mode_lv2 = 'normal'
    elif name == 'P2':
        model.args.query_illum_align = True
        model.args.query_illum_alpha = 0.75
        model.args.query_hist_beta = 0.50
        model.args.texture_fusion = True
        model.args.texture_fusion_lambda = 0.25
        model.args.texture_fusion_s_source = 'a0'
        model.args.stage_mode_lv2 = 'branch_bypass'
    else:
        raise ValueError(name)


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


def _bootstrap(d, n_boot=5000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    if len(d) == 0:
        return None
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        'mean': float(d.mean()), 'median': float(np.median(d)),
        'std': float(d.std(ddof=0)),
        'ci95': [float(lo), float(hi)],
        'note': 'exploratory; small n / cross-domain clustering possible',
    }


def _paired(names, a, b):
    d = np.asarray([a[f] - b[f] for f in names], float)
    wins = int((d > 0.01).sum()); losses = int((d < -0.01).sum())
    ties = len(d) - wins - losses
    k = max(1, int(0.1 * len(d))) if len(d) >= 10 else 0
    ds = np.sort(d)
    trimmed = float(ds[k:-k].mean()) if k and len(d) > 2 * k else float(d.mean())
    order = np.argsort(-d)
    mask = np.ones(len(d), bool)
    if len(d) >= 5:
        mask[order[:2]] = False
    excl2 = float(d[mask].mean()) if mask.any() else None
    # Leave-one-out mean sensitivity
    loo = []
    if len(d) >= 3:
        for i in range(len(d)):
            loo.append(float(np.delete(d, i).mean()))
    return {
        'mean': float(d.mean()), 'median': float(np.median(d)),
        'std': float(d.std(ddof=0)),
        'wins': wins, 'losses': losses, 'ties': ties,
        'trimmed_mean_10pct': trimmed,
        'mean_excluding_top2_positive': excl2,
        'bootstrap': _bootstrap(d),
        'best3': [(names[i], float(d[i])) for i in order[:3]],
        'worst3': [(names[i], float(d[i])) for i in np.argsort(d)[:3]],
        'top2_positive': [(names[i], float(d[i])) for i in order[:2]],
        'loo_mean_min': float(min(loo)) if loo else None,
        'loo_mean_max': float(max(loo)) if loo else None,
        'loo_mean_range': float(max(loo) - min(loo)) if loo else None,
    }


@torch.no_grad()
def _tiled(model, args, lr, lrsr, ref, refsr):
    tile = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile // 2))
    fwd = {}
    if args.query_illum_align and (
            args.query_illum_alpha > 0 or args.query_hist_beta > 0):
        g, lut, _ = estimate_query_align_maps(
            lrsr, ref, alpha=args.query_illum_alpha,
            gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
            bins=args.query_hist_bins)
        fwd['query_gamma_override'] = g
        if args.query_hist_beta > 0:
            fwd['query_hist_lut_override'] = lut
    fwd.update(dict(
        stage_mode_lv3=args.stage_mode_lv3,
        stage_mode_lv2=args.stage_mode_lv2,
        stage_mode_lv1=args.stage_mode_lv1,
    ))
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


def _eval_split(model, camera, ref_dir, tag, out_dir, device, save_preds=False):
    args = _build_args(camera, ref_dir)
    # rebind model.args fields used by _cfg / tiled
    for k, v in vars(args).items():
        setattr(model.args, k, v)
    pairs = _collect_pairs(os.path.join(args.dataset_dir, 'Eval'),
                           camera_filter=camera)
    filenames = [os.path.basename(p[0]) for p in pairs]
    loader = DataLoader(TestSet(args), batch_size=1, shuffle=False,
                        num_workers=0)
    rows = []
    # Cache tensors for later focus viz (full set; Nikon n=20 is small)
    cache = {}
    pred_dir = os.path.join(out_dir, 'predictions', tag)
    if save_preds:
        os.makedirs(pred_dir, exist_ok=True)

    for run in ('A0', 'M1', 'P2'):
        _cfg(model, run)
        print('=== %s / %s ===' % (tag, run))
        for i, batch in enumerate(loader):
            fname = filenames[i]
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)
            sr = _tiled(model, model.args, lr, lrsr, ref, refsr)
            _, ssim, mse, psnr = calc_psnr_and_ssim(sr, hr)
            rows.append({
                'split': tag, 'idx': i, 'filename': fname, 'run': run,
                'psnr_rgb': float(psnr), 'ssim_rgb': float(ssim),
                'mse_rgb': float(mse),
                'luma_mae': _luma_mae(sr, hr),
                'luma_bias': _luma_bias(sr, hr),
                'chroma_mae': _chroma_mae(sr, hr),
            })
            if save_preds:
                from PIL import Image
                Image.fromarray(_to_u8(sr)).save(
                    os.path.join(pred_dir, '%s_%s.png' % (
                        os.path.splitext(fname)[0], run)))
            if save_preds:
                cache.setdefault(fname, {
                    'low': lr.cpu(), 'ref': ref.cpu(), 'hr': hr.cpu()})
                cache[fname][run] = sr.cpu()
            print('[%s %s %02d] %s %.3f' % (tag, run, i, fname, psnr))
            torch.cuda.empty_cache()
    return rows, cache, filenames


def _summarize(rows, tag):
    by = {}
    for r in rows:
        if r['split'] != tag:
            continue
        by.setdefault(r['run'], {})[r['filename']] = r
    names = sorted(by['A0'].keys())
    summary = {}
    for run in ('A0', 'M1', 'P2'):
        lst = [by[run][f] for f in names]
        summary[run] = {
            'psnr_mean': float(np.mean([r['psnr_rgb'] for r in lst])),
            'ssim_mean': float(np.mean([r['ssim_rgb'] for r in lst])),
            'luma_mae': float(np.mean([r['luma_mae'] for r in lst])),
            'luma_bias': float(np.mean([r['luma_bias'] for r in lst])),
            'chroma_mae': float(np.mean([r['chroma_mae'] for r in lst])),
            'n': len(lst),
        }
    psnr = {run: {f: by[run][f]['psnr_rgb'] for f in names}
            for run in ('A0', 'M1', 'P2')}
    ssim = {run: {f: by[run][f]['ssim_rgb'] for f in names}
            for run in ('A0', 'M1', 'P2')}
    paired = {
        'M1_minus_A0_psnr': _paired(names, psnr['M1'], psnr['A0']),
        'P2_minus_M1_psnr': _paired(names, psnr['P2'], psnr['M1']),
        'P2_minus_A0_psnr': _paired(names, psnr['P2'], psnr['A0']),
        'M1_minus_A0_ssim': _paired(names, ssim['M1'], ssim['A0']),
        'P2_minus_M1_ssim': _paired(names, ssim['P2'], ssim['M1']),
        'P2_minus_A0_ssim': _paired(names, ssim['P2'], ssim['A0']),
    }
    return summary, paired, names


def _focus_viz(out, focus, tag):
    d = os.path.join(out, 'focus', tag)
    os.makedirs(d, exist_ok=True)
    for fn, b in focus.items():
        if not all(k in b for k in ('A0', 'M1', 'P2')):
            continue
        names = ['A0', 'M1', 'P2']
        fig = plt.figure(figsize=(9, 7))
        hr = b['hr']
        errs = [_err(b[r], hr) for r in names]
        vmax = max(0.05, float(np.percentile(
            np.concatenate([e.ravel() for e in errs]), 99)))
        ax = fig.add_subplot(3, 3, 1)
        ax.imshow(_to_u8(b['low'])); ax.set_title('Low'); ax.axis('off')
        ax = fig.add_subplot(3, 3, 2)
        ax.imshow(_to_u8(b['ref'])); ax.set_title('Nano'); ax.axis('off')
        ax = fig.add_subplot(3, 3, 3)
        ax.imshow(_to_u8(hr)); ax.set_title('GT'); ax.axis('off')
        for i, r in enumerate(names):
            ax = fig.add_subplot(3, 3, 4 + i)
            ax.imshow(_to_u8(b[r])); ax.set_title(r); ax.axis('off')
        for i, r in enumerate(names):
            ax = fig.add_subplot(3, 3, 7 + i)
            im = ax.imshow(errs[i], vmin=0, vmax=vmax, cmap='magma')
            ax.set_title('|err| ' + r); ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle('%s / %s' % (tag, fn))
        fig.tight_layout()
        fig.savefig(os.path.join(d, '%s.png' % os.path.splitext(fn)[0]),
                    dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'diagnostics', 'v1r14'))
    ap.add_argument('--skip_regression', action='store_true')
    ap.add_argument('--skip_nikon', action='store_true')
    cli = ap.parse_args()
    out = cli.out_dir
    os.makedirs(out, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dummy args for model init
    model = _load(_build_args(
        'Huawei',
        '/root/data/datasets/data1/Eval/Huawei/nanobanana_ref'), device)

    all_rows = []
    results = {
        'eligibility': json.load(open(os.path.join(
            out, 'eligibility_decision.json')))
        if os.path.isfile(os.path.join(out, 'eligibility_decision.json'))
        else {},
        'validation_label': 'Outcome D — Nikon Eval20 CROSS-DOMAIN only; '
                            'no clean independent Huawei Eval holdout',
    }

    # --- Historical Eval30 regression (not in independent aggregate) ---
    if not cli.skip_regression:
        rows_h, focus_h, _ = _eval_split(
            model, 'Huawei',
            '/root/data/datasets/data1/Eval/Huawei/nanobanana_ref',
            'HuaweiEval30_regression_ONLY', out, device, save_preds=False)
        # mark
        for r in rows_h:
            r['role'] = 'historical_regression_only'
        all_rows.extend(rows_h)
        sum_h, _, _ = _summarize(rows_h, 'HuaweiEval30_regression_ONLY')
        reg = {}
        for k, (tp, ts) in REG_HUAWEI.items():
            got = sum_h[k]['psnr_mean']
            reg[k] = {
                'target_psnr': tp, 'got_psnr': got,
                'psnr_ok': abs(got - tp) < 0.03,
                'target_ssim': ts, 'got_ssim': sum_h[k]['ssim_mean'],
                'ssim_ok': abs(sum_h[k]['ssim_mean'] - ts) < 0.002,
            }
        results['huawei_eval30_regression'] = {
            'summary': sum_h, 'checks': reg,
            'note': 'NOT included in independent validation aggregates',
        }
        print('Eval30 regression', json.dumps(reg, indent=2))
        if not all(v['psnr_ok'] and v['ssim_ok'] for v in reg.values()):
            print('WARNING: Eval30 regression mismatch — investigate before '
                  'interpreting Nikon results')

    # --- Independent / cross-domain: Nikon Eval20 ---
    if not cli.skip_nikon:
        rows_n, focus_n, names_n = _eval_split(
            model, 'Nikon',
            '/root/data/datasets/data1/Eval/Nikon/nanobanana_ref_Nikon',
            'NikonEval20_cross_domain', out, device, save_preds=True)
        for r in rows_n:
            r['role'] = 'independent_cross_domain'
        all_rows.extend(rows_n)
        sum_n, paired_n, names_n = _summarize(
            rows_n, 'NikonEval20_cross_domain')
        # Focus: extremes of P2−A0 and P2−M1 (best/worst 2 each, unique)
        focus_names = set()
        for key in ('P2_minus_A0_psnr', 'P2_minus_M1_psnr'):
            for fn, _ in paired_n[key]['best3'][:2]:
                focus_names.add(fn)
            for fn, _ in paired_n[key]['worst3'][:2]:
                focus_names.add(fn)
        focus_sel = {fn: focus_n[fn] for fn in focus_names if fn in focus_n}
        _focus_viz(out, focus_sel, 'NikonEval20_cross_domain')
        results['nikon_eval20'] = {
            'label': 'CROSS-DOMAIN (not same-domain Huawei generalization)',
            'summary': sum_n,
            'paired': paired_n,
            'n': 20,
            'focus_images': sorted(focus_names),
            'caveat': (
                'Stems 3001–3020 also exist under Training/Nikon with '
                'different full-file SHA; treat as cross-device transfer, '
                'not proof of scene uniqueness vs training pool.'),
        }
        with open(os.path.join(out, 'paired_analysis.json'), 'w') as f:
            json.dump(paired_n, f, indent=2)

    keys = []
    for r in all_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, 'per_image_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)

    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    main()
