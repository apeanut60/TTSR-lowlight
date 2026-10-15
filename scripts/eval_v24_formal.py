#!/usr/bin/env python
"""v2.4 formal Eval30 over A0–A3 checkpoints (loads per-arm P2)."""
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
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import read_manifest_csv, sha256  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    EXP, V2_EXP, configure_p2, eval_full, load_h3, load_p2,
)

H3_PSNR, H3_SSIM = 20.466, 0.6331
P2_PSNR, P2_SSIM = 20.265, 0.6340
V23_H0 = 20.479

ARMS = [
    ('A0', 'A0_frozenP2_mse_s42'),
    ('A1', 'A1_frozenP2_l1ssim_s42'),
    ('A2', 'A2_jointP2_mse_s42'),
    ('A3', 'A3_jointP2_l1ssim_s42'),
]
STEPS = [0, 250, 500, 1000, 1500, 2000]


def load_arm_ckpt(path, device, joint):
    ck = torch.load(path, map_location=device)
    p2, args = load_p2(device, joint=False)  # eval mode
    # restore weights then freeze for eval
    p2.load_state_dict(ck['p2'], strict=True)
    configure_p2(p2, joint=False)
    enc, mat, head, _ = load_h3(device)
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    enc.eval(); mat.eval(); head.eval()
    return p2, args, enc, mat, head, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() else 'cpu'
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))

    rows_out = [
        dict(model='P2', step='', psnr=P2_PSNR, ssim=P2_SSIM, d_h3='', d_a0=''),
        dict(model='H3@2000', step='', psnr=H3_PSNR, ssim=H3_SSIM, d_h3=0.0, d_a0=''),
        dict(model='v2.3 H0@1000', step=1000, psnr=V23_H0, ssim=0.6334, d_h3=0.013, d_a0=''),
    ]
    a0_by_step = {}
    all_per = {}

    for arm, run in ARMS:
        run_dir = os.path.join(a.out_root, 'checkpoints', run)
        for st in STEPS:
            ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % st)
            if not os.path.isfile(ckpt):
                print('missing', ckpt); continue
            evj = os.path.join(run_dir, 'eval_step_%05d.json' % st)
            if os.path.isfile(evj) and not a.recompute:
                with open(evj) as f:
                    data = json.load(f)
                summary, per = data['summary'], data['per_image']
            else:
                p2, args, enc, mat, head, _ = load_arm_ckpt(
                    ckpt, device, joint=(arm in ('A2', 'A3')))
                summary, per = eval_full(p2, args, enc, mat, head, ev, device)
                with open(evj, 'w') as f:
                    json.dump(dict(step=st, summary=summary, per_image=per), f)
                del p2, enc, mat, head
                torch.cuda.empty_cache()
            all_per['%s@%d' % (arm, st)] = per
            if arm == 'A0':
                a0_by_step[st] = summary['psnr']
            d_h3 = summary['psnr'] - H3_PSNR
            d_a0 = (summary['psnr'] - a0_by_step[st]) if st in a0_by_step else ''
            rows_out.append(dict(
                model=arm, step=st, psnr=summary['psnr'], ssim=summary['ssim'],
                p2_only=summary.get('p2_only_psnr'),
                d_h3=d_h3, d_a0=d_a0,
                median_delta=summary.get('median_delta'),
                p10_delta=summary.get('p10_delta'),
                worst_delta=summary.get('worst_delta'),
                n_lt_m1=summary.get('n_lt_m1'),
                win=summary.get('win'), loss_n=summary.get('loss_n'),
                ckpt_sha=sha256(ckpt),
            ))
            print('%s@%d  PSNR=%.4f SSIM=%.4f P2only=%.4f ΔH3=%+.4f' % (
                arm, st, summary['psnr'], summary['ssim'],
                summary.get('p2_only_psnr', float('nan')), d_h3))

    for r in rows_out:
        if r['model'] in ('A1', 'A2', 'A3') and r['step'] in a0_by_step:
            r['d_a0'] = r['psnr'] - a0_by_step[r['step']]

    out_dir = os.path.join(a.out_root, 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    fields = sorted({k for r in rows_out for k in r.keys()})
    csv_path = os.path.join(out_dir, 'eval30_summary.csv')
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
