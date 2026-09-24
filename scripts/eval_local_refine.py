#!/usr/bin/env python
"""V2 five-condition evaluation (plan section H1).

    N0                 cached Y0 untouched (baseline)
    V2-Self            self weights, second input = Y0
    V2-Nano-correct    nano weights, corresponding reference
    V2-Nano-bypass     nano weights, refiner bypassed -> must equal N0 exactly
    V2-Nano-mismatch   nano weights, fixed within-camera reference derangement

Uses the same `evaluate_conditions` the training-time validation calls, so the
two entry points cannot drift apart.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import (build_manifest, evaluate_conditions,    # noqa: E402
                                  mismatch_permutation, read_manifest_csv,
                                  sha256)
from model.LocalRefine import Refiner, count_params                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root',
                    default='/root/data/experiments/retinex_v2_localref')
    ap.add_argument('--dataset_dir', default='/root/data/datasets/data1')
    ap.add_argument('--manifest_dir',
                    default='/root/data/datasets/data1/.nanobanana_sample_manifest')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    a = ap.parse_args(_CLI)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = read_manifest_csv(os.path.join(a.out_root, 'manifests_eval.csv'))
    eval_cache = os.path.join(a.out_root, 'cache_n0_eval')

    def load(arm):
        p = os.path.join(a.out_root, 'V2_%s_s%d' % (arm.capitalize(), a.seed),
                         'checkpoint_%05d.pt' % a.step)
        if not os.path.isfile(p):
            raise SystemExit('missing checkpoint %s' % p)
        m = Refiner().to(device)
        m.load_state_dict(torch.load(p, map_location=device)['refiner'])
        print('%s <- %s (%d params)' % (arm, p, count_params(m)))
        return m

    self_m, nano_m = load('self'), load('nano')
    out = {}
    out.update(evaluate_conditions(self_m, rows, eval_cache,
                                   [('N0', 'bypass'), ('V2-Self', 'self')],
                                   device, verbose=False))
    out.update(evaluate_conditions(nano_m, rows, eval_cache,
                                   [('V2-Nano-correct', 'nano'),
                                    ('V2-Nano-bypass', 'bypass'),
                                    ('V2-Nano-mismatch', 'mismatch')],
                                   device, verbose=False))
    labels = ['N0', 'V2-Self', 'V2-Nano-correct', 'V2-Nano-bypass',
              'V2-Nano-mismatch']
    cams = sorted({r['camera'] for r in rows})

    def mean(lbl, cam=None, key='psnr_rgb'):
        v = [d[key] for d in out[lbl].values()
             if cam is None or d['camera'] == cam]
        return float(np.mean(v))

    print()
    print('══ V2 step %d ══' % a.step)
    print('  %-12s %s' % ('camera', ' '.join('%16s' % l for l in labels)))
    for cam in cams + ['ALL', '0.6HW+0.4NK']:
        row = []
        for l in labels:
            if cam == 'ALL':
                row.append(mean(l))
            elif cam == '0.6HW+0.4NK':
                row.append(0.6 * mean(l, 'Huawei') + 0.4 * mean(l, 'Nikon'))
            else:
                row.append(mean(l, cam))
        print('  %-12s %s' % (cam, ' '.join('%16.4f' % v for v in row)))
    print()
    proj = {l: 0.6 * mean(l, 'Huawei') + 0.4 * mean(l, 'Nikon') for l in labels}
    print('  Δscheme   = V2-Nano − N0   = %+.4f dB' % (proj['V2-Nano-correct'] - proj['N0']))
    print('  Δexternal = V2-Nano − Self = %+.4f dB' % (proj['V2-Nano-correct'] - proj['V2-Self']))
    print('  V2-Self   − N0             = %+.4f dB' % (proj['V2-Self'] - proj['N0']))
    print('  bypass    − N0             = %+.4f dB  (应精确为 0)'
          % (proj['V2-Nano-bypass'] - proj['N0']))
    print('  correct   − mismatch       = %+.4f dB'
          % (proj['V2-Nano-correct'] - proj['V2-Nano-mismatch']))
    for l in ['V2-Self', 'V2-Nano-correct']:
        d = np.array([out[l][k]['psnr_rgb'] - out['N0'][k]['psnr_rgb'] for k in out[l]])
        print('  per-image %-16s vs N0: mean %+.4f median %+.4f better %d/%d worst %+.3f'
              % (l, d.mean(), np.median(d), int((d > 0).sum()), len(d), d.min()))
    # ssim summary
    print()
    print('  SSIM(rgb) 0.6HW+0.4NK: %s' % ' '.join(
        '%s=%.4f' % (l, 0.6 * mean(l, 'Huawei', 'ssim_rgb')
                     + 0.4 * mean(l, 'Nikon', 'ssim_rgb')) for l in labels))

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        # Long format per plan section I: one row per (sample, condition) so the
        # report can carry the actual reference used, not just a dataset class
        # name (the §14.4 lesson: an empty ref_dir silently fell back to HR).
        by_sid = {r['sample_id']: r for r in rows}
        perm = {r['sample_id']: rows[j]['sample_id']
                for r, j in zip(rows, mismatch_permutation(rows))}
        init_name = ('refiner_init.pt' if a.seed == 42
                     else 'refiner_init_s%d.pt' % a.seed)
        base_sha = sha256(os.path.join(a.out_root, init_name))
        ck_sha = {}
        for arm in ('self', 'nano'):
            p = os.path.join(a.out_root, 'V2_%s_s%d' % (arm.capitalize(), a.seed),
                             'checkpoint_%05d.pt' % a.step)
            ck_sha[arm] = sha256(p)

        def ref_of(lbl, sid):
            if lbl == 'N0' or lbl == 'V2-Nano-bypass':
                return ('bypass(no_refiner)', '')
            if lbl == 'V2-Self':
                return ('self_Y0_cache', '')
            if lbl == 'V2-Nano-correct':
                return ('nano_correct', by_sid[sid]['nano_path'])
            return ('nano_mismatch', by_sid[perm[sid]]['nano_path'])

        def ck_of(lbl):
            if lbl == 'N0' or lbl == 'V2-Self':
                return 'V2_Self_s%d/checkpoint_%05d.pt' % (a.seed, a.step)
            return 'V2_Nano_s%d/checkpoint_%05d.pt' % (a.seed, a.step)

        with open(a.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'camera', 'step', 'condition', 'ref_source',
                        'ref_path', 'base_sha', 'checkpoint_sha', 'psnr_rgb',
                        'ssim_rgb', 'delta_vs_n0', 'delta_vs_self'])
            for sid in sorted(out['N0']):
                n0 = out['N0'][sid]['psnr_rgb']
                slf = out['V2-Self'][sid]['psnr_rgb']
                for l in labels:
                    rs, rp = ref_of(l, sid)
                    w.writerow([sid, out['N0'][sid]['camera'], a.step, l, rs, rp,
                                base_sha, ck_sha['nano' if 'Nano' in l else 'self'],
                                '%.6f' % out[l][sid]['psnr_rgb'],
                                '%.6f' % out[l][sid]['ssim_rgb'],
                                '%+.6f' % (out[l][sid]['psnr_rgb'] - n0),
                                '%+.6f' % (out[l][sid]['psnr_rgb'] - slf)])
        print('\nper-image -> %s' % a.out)


if __name__ == '__main__':
    main()
