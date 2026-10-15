#!/usr/bin/env python
"""v2.6 formal Train/Eval tables for C1/C2."""
from __future__ import annotations

import argparse
import csv
import json
import os

EXP = '/root/data/experiments/TTSR-lowlight-v26'
ARMS = [
    ('C1', 'C1_random_p2lr1e6_s42', 1e-6),
    ('C2', 'C2_random_p2lr2e7_s42', 2e-7),
]
H3_PSNR = 20.466


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    a = ap.parse_args()
    rows = [
        dict(model='P2', step='', eval_psnr=20.265, eval_ssim=0.6340, p2_lr=''),
        dict(model='H3@2000', step='', eval_psnr=H3_PSNR, eval_ssim=0.6331, p2_lr=''),
        dict(model='v2.4 A2@250', step=250, eval_psnr=20.494, eval_ssim=0.6355, p2_lr='MSE'),
        dict(model='v2.5 W0@2000', step=2000, eval_psnr=20.381, eval_ssim=0.6373, p2_lr=1e-6),
    ]
    c1_by = {}
    all_hist = {}
    for arm, run, lr in ARMS:
        hist_path = os.path.join(a.out_root, 'checkpoints', run, 'eval_history.json')
        if not os.path.isfile(hist_path):
            print('missing', hist_path); continue
        hist = json.load(open(hist_path))
        all_hist[arm] = hist
        for h in hist:
            st = h['step']
            ev = h['eval30']; tr = h.get('train80_crop') or {}
            tf = h.get('train80_full') or {}
            if arm == 'C1':
                c1_by[st] = ev['psnr']
            rows.append(dict(
                model=arm, step=st, p2_lr=lr,
                eval_psnr=ev['psnr'], eval_ssim=ev['ssim'],
                eval_p2=ev.get('p2_psnr'),
                train_crop_psnr=tr.get('psnr'), train_crop_ssim=tr.get('ssim'),
                train_full_psnr=tf.get('psnr') if tf else '',
                gap=tr.get('psnr', 0) - ev['psnr'] if tr.get('psnr') is not None else '',
                d_h3=ev['psnr'] - H3_PSNR,
                d_c1='', n_lt_m1=ev.get('n_lt_m1'),
            ))
            print('%s@%d Eval=%.4f/%.4f TrainCrop=%.4f P2=%.4f' % (
                arm, st, ev['psnr'], ev['ssim'],
                tr.get('psnr', float('nan')), ev.get('p2_psnr', float('nan'))))
    for r in rows:
        if r['model'] == 'C2' and r['step'] in c1_by:
            r['d_c1'] = r['eval_psnr'] - c1_by[r['step']]
        elif r['model'] == 'C1' and r['step'] in c1_by:
            r['d_c1'] = 0.0
    out = os.path.join(a.out_root, 'metrics')
    os.makedirs(out, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(out, 'eval30_train_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(all_hist, open(os.path.join(out, 'curves.json'), 'w'), indent=2)
    print('wrote', os.path.join(out, 'eval30_train_summary.csv'))


if __name__ == '__main__':
    main()
