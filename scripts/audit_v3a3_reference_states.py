#!/usr/bin/env python
"""V3-A.3 §7: corruption audit on dev64 with the frozen R1 proposal.

Measures, for every reference state, whether the frozen proposal is helped or
hurt, together with the state's own ActionOptimal headroom. Only states that are
actually harmful (R1-state < Base) may enter verifier training.

Note: dev64 is held out for the *verifiers* but in-sample for the frozen
proposal (which trained on all 639), so harm estimates here are conservative.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest, read_model_image  # noqa: E402
from local_refine_runtime import metrics                          # noqa: E402
from model.V3A1Refiner import V3A1Refiner                         # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a2_runtime import action_optimal_gate                      # noqa: E402
from v3a_runtime import (contrast_compress, exposure_gain,        # noqa: E402
                         mismatch_permutation, splice_corrupt)

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--out_root', default='/root/data/experiments/v3a3_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--noise_sigma', type=float, default=0.20)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device
    os.makedirs(os.path.join(a.out_root, 'corruption_audit'), exist_ok=True)

    split = json.load(open(os.path.join(a.out_root, 'splits', 'split.json'),
                           encoding='utf-8'))
    dev_ids = set(split['dev'])
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    rows = [r for r in pairs_from_manifest(
        os.path.join(a.root, 'manifests', 'refiner_train.csv'))
        if r[0] in dev_ids]
    rows.sort(key=lambda r: r[0])
    if len(rows) != 64:
        raise SystemExit('expected 64 dev samples, got %d' % len(rows))
    ds = TrainSet(ns, crop_size=0, pairs=rows,
                  y0_cache=os.path.join(a.root, a.cache_name, 'refiner_train'),
                  split='Train')
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(rows))

    m = V3A1Refiner().to(dev)
    m.load_state_dict(torch.load(os.path.join(a.root, R1_CK), map_location=dev)['model'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    states = ['correct', 'true_dark_g0.5', 'true_dark_g0.7', 'true_bright_g1.3',
              'contrast_compress_0.5', 'gaussian_noise_s0.20', 'mismatch', 'corrupt']
    acc = {s: [] for s in ['Base'] + states}
    qopt = {s: [] for s in states}
    head = {s: [] for s in states}
    with torch.no_grad():
        for i, (name, low, high) in enumerate(rows):
            _n, _lr, hr, ref, y0, _mm = ds._load(i)
            y0d, hrd = y0[None].to(dev), hr[None].to(dev)
            mis = read_model_image(ds.ref_map[rows[perm[i]][0]], size=hr.shape[-2:])
            ngen = torch.Generator(device=dev).manual_seed(1234 + i)
            refs = {
                'correct': ref,
                'true_dark_g0.5': exposure_gain(ref, 0.5),
                'true_dark_g0.7': exposure_gain(ref, 0.7),
                'true_bright_g1.3': exposure_gain(ref, 1.3),
                'contrast_compress_0.5': contrast_compress(ref, 0.5),
                'gaussian_noise_s0.20': (ref.to(dev) + a.noise_sigma * torch.randn(
                    ref.shape, generator=ngen, device=dev)).clamp(-1, 1).cpu(),
                'mismatch': mis,
                'corrupt': splice_corrupt(ref, mis, np.random.default_rng(1234 + i)),
            }
            acc['Base'].append(metrics(y0d, hrd)[0])
            for s in states:
                sr, aux = m.proposal(y0d, refs[s][None].to(dev))
                D = aux['gate'] * aux['delta']
                acc[s].append(metrics(sr, hrd)[0])
                q_opt, _e = action_optimal_gate(y0d, hrd, D)
                qf = F.interpolate(q_opt, size=y0d.shape[-2:], mode='bilinear',
                                   align_corners=False)
                head[s].append(metrics(y0d + qf * D, hrd)[0] - acc[s][-1])
                qopt[s].append(float(q_opt.mean()))

    b = float(np.mean(acc['Base']))
    out = dict(n=len(rows), base=b, states={})
    print('══ corruption audit on dev64 (frozen R1) ══')
    print('  Base %.4f' % b)
    print('  %-24s %9s %9s %9s %6s %6s %9s %9s'
          % ('state', 'PSNR', 'd_mean', 'd_median', 'better', '<-1dB',
             'q_opt', 'headroom'))
    for s in states:
        v = np.array(acc[s]); d = v - b
        out['states'][s] = dict(
            psnr=float(v.mean()), d_mean=float(d.mean()),
            d_median=float(np.median(d)), better=int((d > 0).sum()), n=len(d),
            worst=float(d.min()), below_minus1=int((d < -1).sum()),
            q_opt=float(np.mean(qopt[s])), headroom=float(np.mean(head[s])),
            harmful=bool(d.mean() < 0))
        print('  %-24s %9.4f %+9.4f %+9.4f %6d %6d %9.3f %+9.4f  %s'
              % (s, v.mean(), d.mean(), np.median(d), (d > 0).sum(),
                 (d < -1).sum(), np.mean(qopt[s]), np.mean(head[s]),
                 'HARMFUL' if d.mean() < 0 else ''))
    harmful = [s for s in states if out['states'][s]['harmful']]
    out['harmful_states'] = harmful
    json.dump(out, open(os.path.join(a.out_root, 'corruption_audit',
                                     'audit.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    print('\n  harmful states: %s' % (harmful or 'NONE'))
    print('  -> %s' % os.path.join(a.out_root, 'corruption_audit', 'audit.json'))


if __name__ == '__main__':
    main()
