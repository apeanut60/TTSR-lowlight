#!/usr/bin/env python
"""V3-A final comparison + oracle-gate diagnostic (plan §9 / §10).

Conditions: Base-R, R1-correct, R2-correct, R2-bypass, R2-mismatch,
R2-corrupt, and the diagnostic R2-oracle-gate (the model's own delta_low with
the *predicted* gate replaced by the GT-derived q_star).

The oracle comparison says which half is the bottleneck:
    oracle >> R2   -> the verifier is the bottleneck
    oracle ~= R2   -> the proposal is the bottleneck
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest   # noqa: E402
from local_refine_runtime import metrics                          # noqa: E402
from model.V3ARefiner import V3ARefiner                           # noqa: E402
from option import parser as option_parser                        # noqa: E402
from scripts.train_v3a_lolv2real import build_args                # noqa: E402
from v3a_runtime import mismatch_permutation, splice_corrupt, usefulness  # noqa: E402


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    m = V3ARefiner().to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


@torch.no_grad()
def run(root, data_dir, variant, step, device):
    args = build_args(root, data_dir, variant)
    pairs = pairs_from_manifest(os.path.join(root, 'manifests', 'test.csv'))
    cache = os.path.join(root, 'cache_y0', 'test')
    ds = TrainSet(args, crop_size=0, pairs=pairs, y0_cache=cache, split='Test')
    tau = json.load(open(os.path.join(root, 'tau.json')))['tau']
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(pairs))

    r1, _ = load_model(os.path.join(root, 'R1_naive_s42', 'checkpoint_%05d.pt' % step), device)
    r2, _ = load_model(os.path.join(root, 'R2_hallucination_aware_s42',
                                    'checkpoint_%05d.pt' % step), device)

    out = {}
    for i in range(len(pairs)):
        _n, _lr, hr, ref, y0, _m = ds._load(i)
        from dataset.lolv2real_v3a import read_model_image
        mis = read_model_image(ds.ref_map[os.path.basename(pairs[perm[i]][1])],
                               size=hr.shape[-2:])
        cor = splice_corrupt(ref, mis, np.random.default_rng(1234 + i))
        y0d, hrd = y0[None].to(device), hr[None].to(device)
        sid = os.path.basename(pairs[i][1])
        rec = dict(sample_id=sid)
        rec['Base'] = metrics(y0d, hrd)
        for nm, mdl, r in (('R1-correct', r1, ref), ('R2-correct', r2, ref),
                           ('R2-mismatch', r2, mis), ('R2-corrupt', r2, cor)):
            o, _aux = mdl(y0d, r[None].to(device))
            rec[nm] = metrics(o, hrd)
        o, _ = r2(y0d, None)
        rec['R2-bypass'] = metrics(o, hrd)
        # oracle gate: keep the model's delta, swap in the true q_star
        _o, aux = r2(y0d, ref[None].to(device))
        q_star, _d = usefulness(y0d, ref[None].to(device), hrd, tau)
        import torch.nn.functional as F
        q = F.interpolate(q_star, size=y0d.shape[-2:], mode='bilinear',
                          align_corners=False)
        rec['R2-oracle-gate'] = metrics(y0d + q * aux['delta'], hrd)
        rec['_q_mean'] = float(aux['gate'].mean())
        rec['_qstar_mean'] = float(q_star.mean())
        out[sid] = rec
        if (i + 1) % 25 == 0:
            print('  %d/%d' % (i + 1, len(pairs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    res = run(a.root, a.data_dir, a.variant, a.step, a.device)
    conds = ['Base', 'R1-correct', 'R2-correct', 'R2-bypass', 'R2-mismatch',
             'R2-corrupt', 'R2-oracle-gate']
    sids = sorted(res)

    def col(c, k=0):
        return np.array([res[s][c][k] for s in sids])

    print()
    print('══ V3-A  LOL-v2-real Test (n=%d, step=%d) ══' % (len(sids), a.step))
    print('  %-16s %10s %10s' % ('condition', 'PSNR(rgb)', 'SSIM(Y)'))
    for c in conds:
        print('  %-16s %10.4f %10.4f' % (c, col(c, 0).mean(), col(c, 1).mean()))
    base = col('Base', 0)
    print()
    for c in conds[1:]:
        d = col(c, 0) - base
        print('  %-16s d_mean %+.4f  median %+.4f  better %2d/%d  worst %+.3f  <-1dB %d'
              % (c, d.mean(), np.median(d), int((d > 0).sum()), len(d), d.min(),
                 int((d < -1.0).sum())))
    d_verify = col('R2-correct', 0) - col('R1-correct', 0)
    print()
    print('  Delta_scheme  R2 - Base      = %+.4f' % (col('R2-correct', 0) - base).mean())
    print('  Delta_verify  R2 - R1        = %+.4f' % d_verify.mean())
    print('  Delta_safety  R2-correct - R2-mismatch  = %+.4f'
          % (col('R2-correct', 0) - col('R2-mismatch', 0)).mean())
    print('  Delta_corrupt R2-correct - R2-corrupt   = %+.4f'
          % (col('R2-correct', 0) - col('R2-corrupt', 0)).mean())
    print('  oracle - R2   = %+.4f  (large => verifier is the bottleneck)'
          % (col('R2-oracle-gate', 0) - col('R2-correct', 0)).mean())
    q = np.array([res[s]['_q_mean'] for s in sids])
    qs = np.array([res[s]['_qstar_mean'] for s in sids])
    print('  corr(predicted q, q_star) = %.3f' % float(np.corrcoef(q, qs)[0, 1]))

    out = os.path.join(a.root, 'eval')
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, 'per_image_v3a_s42.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id'] + [x for c in conds
                                    for x in (c + '_psnr', c + '_ssim')]
                   + ['q_mean', 'q_star_mean'])
        for s in sids:
            w.writerow([s] + [v for c in conds for v in res[s][c][:2]]
                       + ['%.6f' % res[s]['_q_mean'], '%.6f' % res[s]['_qstar_mean']])
    json.dump(dict(step=a.step, n=len(sids),
                   psnr={c: float(col(c, 0).mean()) for c in conds},
                   ssim_y={c: float(col(c, 1).mean()) for c in conds},
                   delta={c: float((col(c, 0) - base).mean()) for c in conds[1:]},
                   corr_q_qstar=float(np.corrcoef(q, qs)[0, 1])),
              open(os.path.join(out, 'summary_v3a.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    print('\n-> %s' % csv_path)


if __name__ == '__main__':
    main()
