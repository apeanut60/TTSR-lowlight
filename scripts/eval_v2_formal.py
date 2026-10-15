#!/usr/bin/env python
"""Formal Huawei Eval30 evaluation + ablations for v2 H1/H2/H3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, evaluate_conditions, evaluate_v3b0, load_cache,
    metrics, mismatch_permutation, read_manifest_csv, sha256,
    _read_eval_tensor,
)
from model.LocalRefine import LocalSoftMatch, Refiner, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v2'
FOCUS = ['2037.jpg', '2040.jpg', '2053.jpg', '2041.jpg',
         '2063.jpg', '2056.jpg', '2062.jpg']


def _paired(names, a, b):
    d = np.asarray([a[n] - b[n] for n in names], float)
    wins = int((d > 0.01).sum()); losses = int((d < -0.01).sum())
    return dict(
        mean=float(d.mean()), median=float(np.median(d)),
        std=float(d.std(ddof=0)),
        wins=wins, losses=losses, ties=len(d) - wins - losses,
        best5=[(names[i], float(d[i])) for i in np.argsort(-d)[:5]],
        worst5=[(names[i], float(d[i])) for i in np.argsort(d)[:5]],
    )


def _freq_diag(delta, sr, hr):
    """Low/high frequency energy of residual and rough PSNR contribution."""
    # delta, sr, hr: [1,3,H,W]
    d = delta[0].detach().float().cpu()
    # simple blur residual as low-freq
    k = torch.ones(1, 1, 5, 5) / 25.0
    low = torch.nn.functional.conv2d(
        d.mean(0, keepdim=True)[None], k, padding=2)[0, 0]
    high = d.mean(0) - low
    e_low = float((low ** 2).mean())
    e_high = float((high ** 2).mean())
    p_full, _, _ = metrics(sr, hr)
    p_nolow, _, _ = metrics(sr - low[None, None].to(sr.device), hr)  # rough
    return dict(e_low=e_low, e_high=e_high,
                ratio_low=e_low / (e_low + e_high + 1e-12),
                psnr_full=p_full)


def load_refiner(ckpt, device):
    m = Refiner().to(device)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd['refiner'] if 'refiner' in sd else sd)
    m.eval()
    m.match.chunk = EVAL_QUERY_CHUNK
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_root', default=EXP)
    ap.add_argument('--h1_ckpt', default='')
    ap.add_argument('--h2_ckpt', default='')
    ap.add_argument('--h3_ckpt', default='')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = read_manifest_csv(os.path.join(a.exp_root, 'manifests',
                                          'manifests_eval.csv'))
    cache = os.path.join(a.exp_root, 'cache_p2_eval')
    out_dir = os.path.join(a.exp_root, 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    vis = os.path.join(a.exp_root, 'visualizations')
    os.makedirs(vis, exist_ok=True)

    def latest(run):
        d = os.path.join(a.exp_root, 'checkpoints', run)
        cands = sorted(f for f in os.listdir(d)
                       if f.startswith('checkpoint_') and f.endswith('.pt'))
        return os.path.join(d, cands[-1]) if cands else ''

    h1 = a.h1_ckpt or latest('H1_nano_s42')
    h2 = a.h2_ckpt or latest('H2_self_s42')
    h3 = a.h3_ckpt or latest('H3_v3b0_s42')

    results = {}
    per_rows = []

    # H0 from cache
    h0 = {}
    for r in rows:
        y0 = torch.from_numpy(np.ascontiguousarray(load_cache(cache, r)))[None].to(device)
        hr = _read_eval_tensor(r['high_path'], device)
        p, s, m = metrics(y0, hr)
        h0[r['sample_id']] = dict(psnr=p, ssim=s)
        per_rows.append(dict(sample_id=r['sample_id'], run='H0_P2',
                             psnr_rgb=p, ssim=s))
    results['H0'] = dict(
        psnr_mean=float(np.mean([v['psnr'] for v in h0.values()])),
        ssim_mean=float(np.mean([v['ssim'] for v in h0.values()])),
    )
    print('H0', results['H0'])

    # H1 ablations
    if h1 and os.path.isfile(h1):
        ref = load_refiner(h1, device)
        abl = evaluate_conditions(
            ref, rows, cache,
            [('H1_nano', 'nano'), ('H1_self_input', 'self'),
             ('H1_mismatch', 'mismatch'), ('H0_bypass', 'bypass')],
            device, verbose=True)
        for label in abl:
            ps = [abl[label][r['sample_id']]['psnr_rgb'] for r in rows]
            ss = [abl[label][r['sample_id']]['ssim_y'] for r in rows]
            results[label] = dict(psnr_mean=float(np.mean(ps)),
                                  ssim_mean=float(np.mean(ss)),
                                  ckpt=h1, ckpt_sha256=sha256(h1))
            for r in rows:
                d = abl[label][r['sample_id']]
                per_rows.append(dict(sample_id=r['sample_id'], run=label,
                                     psnr_rgb=d['psnr_rgb'], ssim=d['ssim_y']))
        names = [r['sample_id'] for r in rows]
        p_h0 = {n: h0[n]['psnr'] for n in names}
        p_h1 = {n: abl['H1_nano'][n]['psnr_rgb'] for n in names}
        results['H1_minus_H0'] = _paired(names, p_h1, p_h0)
        results['H1_nano_minus_self'] = _paired(
            names, p_h1, {n: abl['H1_self_input'][n]['psnr_rgb'] for n in names})
        results['H1_nano_minus_mismatch'] = _paired(
            names, p_h1, {n: abl['H1_mismatch'][n]['psnr_rgb'] for n in names})

        # frequency diag on focus set
        freq = {}
        for r in rows:
            bn = os.path.basename(r['sample_id'])
            if bn not in FOCUS:
                continue
            y0 = torch.from_numpy(np.ascontiguousarray(load_cache(cache, r)))[None].to(device)
            hr = _read_eval_tensor(r['high_path'], device)
            nano = _read_eval_tensor(r['nano_path'], device, size=y0.shape[-2:])
            with torch.no_grad():
                sr, aux = ref(y0, nano)
            freq[r['sample_id']] = _freq_diag(aux['delta'] * aux['gate'], sr, hr)
        results['H1_freq_focus'] = freq

    if h2 and os.path.isfile(h2):
        ref2 = load_refiner(h2, device)
        abl2 = evaluate_conditions(
            ref2, rows, cache, [('H2_self', 'self')], device, verbose=True)
        ps = [abl2['H2_self'][r['sample_id']]['psnr_rgb'] for r in rows]
        ss = [abl2['H2_self'][r['sample_id']]['ssim_y'] for r in rows]
        results['H2'] = dict(psnr_mean=float(np.mean(ps)),
                             ssim_mean=float(np.mean(ss)),
                             ckpt=h2, ckpt_sha256=sha256(h2))
        for r in rows:
            d = abl2['H2_self'][r['sample_id']]
            per_rows.append(dict(sample_id=r['sample_id'], run='H2_self',
                                 psnr_rgb=d['psnr_rgb'], ssim=d['ssim_y']))
        if 'H1_nano' in results:
            names = [r['sample_id'] for r in rows]
            results['H1_minus_H2'] = _paired(
                names,
                {n: abl['H1_nano'][n]['psnr_rgb'] for n in names},
                {n: abl2['H2_self'][n]['psnr_rgb'] for n in names})
            results['H2_minus_H0'] = _paired(
                names,
                {n: abl2['H2_self'][n]['psnr_rgb'] for n in names},
                {n: h0[n]['psnr'] for n in names})

    if h3 and os.path.isfile(h3):
        ck = torch.load(h3, map_location=device)
        enc = SharedEncoder(32).to(device).eval()
        mat = LocalSoftMatch(32, 4, 0.1, chunk=EVAL_QUERY_CHUNK).to(device).eval()
        head = V3B0ResidualFusion(96).to(device).eval()
        enc.load_state_dict(ck['encoder'])
        mat.load_state_dict(ck['match'])
        head.load_state_dict(ck['head'])
        abl3 = evaluate_v3b0(head, enc, mat, rows, cache, device)
        ps = [abl3['H3'][r['sample_id']]['psnr_rgb'] for r in rows]
        ss = [abl3['H3'][r['sample_id']]['ssim_y'] for r in rows]
        results['H3'] = dict(psnr_mean=float(np.mean(ps)),
                             ssim_mean=float(np.mean(ss)),
                             ckpt=h3, ckpt_sha256=sha256(h3))
        for r in rows:
            d = abl3['H3'][r['sample_id']]
            per_rows.append(dict(sample_id=r['sample_id'], run='H3',
                                 psnr_rgb=d['psnr_rgb'], ssim=d['ssim_y']))
        names = [r['sample_id'] for r in rows]
        results['H3_minus_H0'] = _paired(
            names,
            {n: abl3['H3'][n]['psnr_rgb'] for n in names},
            {n: h0[n]['psnr'] for n in names})
        if 'H1_nano' in results:
            results['H3_minus_H1'] = _paired(
                names,
                {n: abl3['H3'][n]['psnr_rgb'] for n in names},
                {n: abl['H1_nano'][n]['psnr_rgb'] for n in names})

    with open(os.path.join(out_dir, 'formal_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, 'per_image_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'run', 'psnr_rgb', 'ssim'])
        w.writeheader(); w.writerows(per_rows)
    print(json.dumps(results, indent=2)[:4000])
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
