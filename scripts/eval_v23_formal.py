#!/usr/bin/env python
"""v2.3 formal Eval30 for all H0/H1/H2 checkpoints."""
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
    EVAL_QUERY_CHUNK, load_cache, metrics, read_manifest_csv, sha256,
    _read_eval_tensor,
)
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402

H3_CKPT = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
           'H3_v3b0_s42/checkpoint_02000.pt')
V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
EXP = '/root/data/experiments/TTSR-lowlight-v23'
H3_PSNR, H3_SSIM = 20.466, 0.6331
P2_PSNR, P2_SSIM = 20.265, 0.6340
V22_V1_PSNR, V22_V1_SSIM = 20.479, 0.6334


def load_from_ckpt(path, device):
    ck = torch.load(path, map_location=device)
    enc = SharedEncoder(32).to(device).eval()
    mat = LocalSoftMatch(32, 4, 0.1, chunk=EVAL_QUERY_CHUNK).to(device).eval()
    head = V3B0ResidualFusion(96).to(device).eval()
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    return enc, mat, head, ck


@torch.no_grad()
def eval_one(enc, mat, head, rows, cache, device):
    per = {}
    for row in rows:
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(cache, row)))[None].to(device)
        gt = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        f0 = enc((y0 + 1.) * 0.5)
        fr = enc((nano + 1.) * 0.5)
        y = y0 + head(f0, mat(f0, fr), y0.shape[-2:])
        p, s, m = metrics(y, gt)
        p2, s2, m2 = metrics(y0, gt)
        per[row['sample_id']] = dict(
            psnr=p, ssim=s, mse=m,
            p2_psnr=p2, p2_ssim=s2)
    deltas = [v['psnr'] - v['p2_psnr'] for v in per.values()]
    summary = dict(
        psnr=float(np.mean([v['psnr'] for v in per.values()])),
        ssim=float(np.mean([v['ssim'] for v in per.values()])),
        median_delta=float(np.median(deltas)),
        p10_delta=float(np.percentile(deltas, 10)),
        worst_delta=float(np.min(deltas)),
        n_lt_m1=int(sum(1 for d in deltas if d < -1)),
        win=int(sum(1 for d in deltas if d > 0)),
        loss_n=int(sum(1 for d in deltas if d < 0)),
    )
    return summary, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() else 'cpu'
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    cache = os.path.join(V2_EXP, 'cache_p2_eval')

    arms = [
        ('H0', 'H0_head_only_s42'),
        ('H1', 'H1_matcher_head_s42'),
        ('H2', 'H2_encoder_matcher_head_s42'),
    ]
    steps = [0, 250, 500, 750, 1000]
    rows_out = []
    all_per = {}

    # baselines
    rows_out.append(dict(model='P2', step='', psnr=P2_PSNR, ssim=P2_SSIM,
                          d_h3='', d_h0=''))
    rows_out.append(dict(model='H3@2000', step='', psnr=H3_PSNR, ssim=H3_SSIM,
                          d_h3=0.0, d_h0=''))
    rows_out.append(dict(model='v2.2 V1@1000', step=1000, psnr=V22_V1_PSNR,
                          ssim=V22_V1_SSIM, d_h3=0.013, d_h0=''))

    h0_by_step = {}
    for arm, run in arms:
        run_dir = os.path.join(a.out_root, 'checkpoints', run)
        for st in steps:
            ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % st)
            if not os.path.isfile(ckpt):
                print('missing', ckpt)
                continue
            # reuse training-time eval if present
            evj = os.path.join(run_dir, 'eval_step_%05d.json' % st)
            if os.path.isfile(evj):
                with open(evj) as f:
                    data = json.load(f)
                summary, per = data['summary'], data['per_image']
            else:
                enc, mat, head, _ = load_from_ckpt(ckpt, device)
                summary, per = eval_one(enc, mat, head, ev, cache, device)
                del enc, mat, head
                torch.cuda.empty_cache()
            all_per['%s@%d' % (arm, st)] = per
            if arm == 'H0':
                h0_by_step[st] = summary['psnr']
            d_h3 = summary['psnr'] - H3_PSNR
            d_h0 = (summary['psnr'] - h0_by_step[st]) if st in h0_by_step else ''
            rows_out.append(dict(
                model=arm, step=st, psnr=summary['psnr'], ssim=summary['ssim'],
                d_h3=d_h3, d_h0=d_h0,
                median_delta=summary.get('median_delta'),
                p10_delta=summary.get('p10_delta'),
                worst_delta=summary.get('worst_delta'),
                n_lt_m1=summary.get('n_lt_m1'),
                win=summary.get('win'), loss_n=summary.get('loss_n'),
                ckpt_sha=sha256(ckpt),
            ))
            print('%s@%d  PSNR=%.4f SSIM=%.4f ΔH3=%+.4f' % (
                arm, st, summary['psnr'], summary['ssim'], d_h3))

    # recompute d_h0 after H0 known
    for r in rows_out:
        if r['model'] in ('H1', 'H2') and r['step'] in h0_by_step:
            r['d_h0'] = r['psnr'] - h0_by_step[r['step']]

    out_dir = os.path.join(a.out_root, 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'eval30_summary.csv')
    fields = sorted({k for r in rows_out for k in r.keys()})
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    with open(os.path.join(out_dir, 'eval30_all.json'), 'w') as f:
        json.dump(dict(rows=rows_out, per_image=all_per), f)
    print('wrote', csv_path)


if __name__ == '__main__':
    main()
