#!/usr/bin/env python
"""v2.5 formal table from W0/W1/W2 eval_history (+ optional recompute)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

EXP = '/root/data/experiments/TTSR-lowlight-v25'
ARMS = [
    ('W0', 'W0_l1_ms1_s42', 1.0),
    ('W1', 'W1_l1_ms03_s42', 0.3),
    ('W2', 'W2_l1_ms01_s42', 0.1),
]
STEPS = [0, 250, 500, 1000, 1500, 2000]
H3_PSNR, H3_SSIM = 20.466, 0.6331


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    a = ap.parse_args()
    rows = [
        dict(model='P2', step='', eval_psnr=20.265, eval_ssim=0.6340,
             train_psnr='', train_ssim='', lambda_ms='', d_h3='', d_w0=''),
        dict(model='H3@2000', step='', eval_psnr=H3_PSNR, eval_ssim=H3_SSIM,
             train_psnr='', train_ssim='', lambda_ms='', d_h3=0.0, d_w0=''),
        dict(model='v2.4 A2@250', step=250, eval_psnr=20.494, eval_ssim=0.6355,
             train_psnr='', train_ssim='', lambda_ms='MSE', d_h3=0.028, d_w0=''),
        dict(model='v2.4 A3@250', step=250, eval_psnr=20.472, eval_ssim=0.6364,
             train_psnr='', train_ssim='', lambda_ms=1.0, d_h3=0.006, d_w0=''),
    ]
    w0_by = {}
    all_hist = {}
    for arm, run, lam in ARMS:
        hist_path = os.path.join(a.out_root, 'checkpoints', run, 'eval_history.json')
        if not os.path.isfile(hist_path):
            print('missing', hist_path); continue
        hist = json.load(open(hist_path))
        all_hist[arm] = hist
        for h in hist:
            st = h['step']
            ev = h['eval30']
            tr = h.get('train80_crop') or {}
            if arm == 'W0':
                w0_by[st] = ev['psnr']
            rows.append(dict(
                model=arm, step=st, lambda_ms=lam,
                eval_psnr=ev['psnr'], eval_ssim=ev['ssim'],
                eval_l1=ev.get('l1'), eval_ms=ev.get('ms_ssim'),
                eval_p2=ev.get('p2_psnr'),
                train_psnr=tr.get('psnr'), train_ssim=tr.get('ssim'),
                train_l1=tr.get('l1'), train_ms=tr.get('ms_ssim'),
                train_p2=tr.get('p2_psnr'),
                train80_full_psnr=(h.get('train80_full') or {}).get('psnr'),
                gap_psnr=(tr.get('psnr') - ev['psnr']) if tr.get('psnr') is not None else '',
                d_h3=ev['psnr'] - H3_PSNR,
                d_w0='',
                n_lt_m1=ev.get('n_lt_m1'),
            ))
            print('%s@%d λ=%.1f Eval=%.4f/%.4f TrainCrop=%.4f/%.4f gap=%s' % (
                arm, st, lam, ev['psnr'], ev['ssim'],
                tr.get('psnr', float('nan')), tr.get('ssim', float('nan')),
                '%.3f' % (tr['psnr'] - ev['psnr']) if tr.get('psnr') is not None else 'n/a'))

    for r in rows:
        if r['model'] in ('W1', 'W2') and r['step'] in w0_by:
            r['d_w0'] = r['eval_psnr'] - w0_by[r['step']]
        elif r['model'] == 'W0' and r['step'] in w0_by:
            r['d_w0'] = 0.0

    out = os.path.join(a.out_root, 'metrics')
    os.makedirs(out, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(out, 'eval30_train_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(out, 'curves.json'), 'w') as f:
        json.dump(all_hist, f, indent=2)
    print('wrote', os.path.join(out, 'eval30_train_summary.csv'))


if __name__ == '__main__':
    main()
