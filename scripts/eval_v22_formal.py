#!/usr/bin/env python
"""v2.2 formal Eval30: P2 / V0 H3 / V1 / V2 Full+Centered + reference ablations."""
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
    EVAL_QUERY_CHUNK, load_cache, metrics, mismatch_permutation,
    read_manifest_csv, sha256, _read_eval_tensor,
)
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from model.v22_dc_loss import mean_hw, split_dc  # noqa: E402

H3_CKPT = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
           'H3_v3b0_s42/checkpoint_02000.pt')
V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
EXP = '/root/data/experiments/TTSR-lowlight-v22'
FOCUS_BAD = ['2053', '2059', '2055', '2058', '2042']
FOCUS_GOOD = ['2052', '2057', '2044', '2061', '2056']


def load_bundle(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    enc = SharedEncoder(32).to(device).eval()
    mat = LocalSoftMatch(32, 4, 0.1, chunk=EVAL_QUERY_CHUNK).to(device).eval()
    head = V3B0ResidualFusion(96).to(device).eval()
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    return enc, mat, head, ck


@torch.no_grad()
def pred_d(enc, mat, head, y0, r_eff):
    f0 = enc((y0 + 1.) * 0.5)
    fr = enc((r_eff + 1.) * 0.5)
    return head(f0, mat(f0, fr), y0.shape[-2:])


@torch.no_grad()
def pred_d_zerot(enc, head, y0):
    f0 = enc((y0 + 1.) * 0.5)
    return head(f0, torch.zeros_like(f0), y0.shape[-2:])


def summarize_deltas(full_psnr, p2_psnr, names):
    d = np.asarray([full_psnr[n] - p2_psnr[n] for n in names], float)
    return dict(
        mean=float(d.mean()), median=float(np.median(d)),
        p10=float(np.percentile(d, 10)),
        worst=float(d.min()),
        n_lt_m1=int((d < -1).sum()),
        wins=int((d > 0.01).sum()), losses=int((d < -0.01).sum()),
        worst5=[(names[i], float(d[i])) for i in np.argsort(d)[:5]],
        best5=[(names[i], float(d[i])) for i in np.argsort(-d)[:5]],
    )


@torch.no_grad()
def eval_run(enc, mat, head, rows, cache, device, do_ablation=False):
    mismatch = mismatch_permutation(rows)
    names = [r['sample_id'] for r in rows]
    store = {k: {} for k in (
        'P2', 'Full', 'Centered',
        'Full_Self', 'Full_Mismatch', 'Full_ZeroT')}
    dc_stats = []
    for i, row in enumerate(rows):
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(cache, row)))[None].to(device)
        gt = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        d = pred_d(enc, mat, head, y0, nano)
        m, ds = split_dc(d)
        variants = {
            'P2': y0,
            'Full': y0 + d,
            'Centered': y0 + ds,
        }
        if do_ablation:
            variants['Full_Self'] = y0 + pred_d(enc, mat, head, y0, y0)
            mis = _read_eval_tensor(rows[mismatch[i]]['nano_path'], device,
                                    size=y0.shape[-2:])
            variants['Full_Mismatch'] = y0 + pred_d(enc, mat, head, y0, mis)
            variants['Full_ZeroT'] = y0 + pred_d_zerot(enc, head, y0)
        for lab, pred in variants.items():
            p, s, mse = metrics(pred, gt)
            store[lab][row['sample_id']] = dict(psnr=p, ssim=s)
        mv = m[0, :, 0, 0].cpu().numpy()
        e0 = mean_hw(y0 - gt)[0, :, 0, 0].cpu().numpy()
        sum_m = float((m.expand_as(d) ** 2).sum())
        sum_s = float((ds ** 2).sum())
        dc_stats.append(dict(
            sample_id=row['sample_id'],
            d_mean_R=float(mv[0]), d_mean_G=float(mv[1]), d_mean_B=float(mv[2]),
            m_norm=float(np.linalg.norm(mv)),
            e_mean_frac=sum_m / (sum_m + sum_s + 1e-12),
            signed_dc_err=float(np.dot(e0, mv)),
        ))
    p2 = {n: store['P2'][n]['psnr'] for n in names}
    full = {n: store['Full'][n]['psnr'] for n in names}
    cent = {n: store['Centered'][n]['psnr'] for n in names}
    summary = dict(
        P2_psnr=float(np.mean(list(p2.values()))),
        P2_ssim=float(np.mean([store['P2'][n]['ssim'] for n in names])),
        Full_psnr=float(np.mean(list(full.values()))),
        Full_ssim=float(np.mean([store['Full'][n]['ssim'] for n in names])),
        Centered_psnr=float(np.mean(list(cent.values()))),
        Centered_ssim=float(np.mean([store['Centered'][n]['ssim'] for n in names])),
        full_vs_p2=summarize_deltas(full, p2, names),
        centered_vs_p2=summarize_deltas(cent, p2, names),
        mean_e_mean_frac=float(np.mean([s['e_mean_frac'] for s in dc_stats])),
        mean_m_norm=float(np.mean([s['m_norm'] for s in dc_stats])),
    )
    if do_ablation:
        for lab in ('Full_Self', 'Full_Mismatch', 'Full_ZeroT'):
            summary[lab + '_psnr'] = float(np.mean(
                [store[lab][n]['psnr'] for n in names]))
    return summary, store, dc_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--v1_step', type=int, default=1000)
    ap.add_argument('--v2_step', type=int, default=1000)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    cache = os.path.join(V2_EXP, 'cache_p2_eval')
    out = os.path.join(a.out_root, 'metrics')
    os.makedirs(out, exist_ok=True)

    runs = {
        'V0': H3_CKPT,
        'V1': os.path.join(a.out_root, 'checkpoints/V1_full_mse_s42',
                           'checkpoint_%05d.pt' % a.v1_step),
        'V2': os.path.join(a.out_root, 'checkpoints/V2_full_dc2_s42',
                           'checkpoint_%05d.pt' % a.v2_step),
    }
    # also eval best intermediate if histories exist
    all_results = {}
    for name, ckpt in runs.items():
        if not os.path.isfile(ckpt):
            print('skip missing', name, ckpt)
            continue
        print('===', name, ckpt)
        enc, mat, head, ck = load_bundle(ckpt, device)
        summary, store, dc = eval_run(
            enc, mat, head, rows, cache, device, do_ablation=True)
        summary['ckpt'] = ckpt
        summary['ckpt_sha256'] = sha256(ckpt)
        all_results[name] = summary
        # per-image csv fragment
        with open(os.path.join(out, '%s_per_image.csv' % name), 'w', newline='') as f:
            fields = ['sample_id', 'psnr_P2', 'psnr_Full', 'psnr_Centered',
                      'd_mean_R', 'd_mean_G', 'd_mean_B', 'e_mean_frac']
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            dc_map = {d['sample_id']: d for d in dc}
            for r in rows:
                sid = r['sample_id']
                w.writerow(dict(
                    sample_id=sid,
                    psnr_P2=store['P2'][sid]['psnr'],
                    psnr_Full=store['Full'][sid]['psnr'],
                    psnr_Centered=store['Centered'][sid]['psnr'],
                    d_mean_R=dc_map[sid]['d_mean_R'],
                    d_mean_G=dc_map[sid]['d_mean_G'],
                    d_mean_B=dc_map[sid]['d_mean_B'],
                    e_mean_frac=dc_map[sid]['e_mean_frac'],
                ))
        print(json.dumps({k: summary[k] for k in summary if 'psnr' in k or k.startswith('Full_')}, indent=2))

    # curves from training histories
    curves = {}
    for arm, dirname in [('V1', 'V1_full_mse_s42'), ('V2', 'V2_full_dc2_s42')]:
        hp = os.path.join(a.out_root, 'checkpoints', dirname, 'eval_history.json')
        if os.path.isfile(hp):
            curves[arm] = json.load(open(hp))
    all_results['curves'] = curves
    all_results['note'] = 'Eval30 is a reused development set; not independent holdout.'

    with open(os.path.join(out, 'formal_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()
