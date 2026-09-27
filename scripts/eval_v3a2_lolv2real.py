#!/usr/bin/env python
"""V3-A.2 formal comparison (plan §14) with the action-optimal diagnostic."""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest   # noqa: E402
from local_refine_runtime import metrics                          # noqa: E402
from model.V3A1Refiner import V3A1Refiner                         # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a2_runtime import action_optimal_gate                      # noqa: E402
from v3a_runtime import (mismatch_permutation, rescale_reference, # noqa: E402
                         splice_corrupt)

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--out_root', default='/root/data/experiments/v3a2_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--noise_sigma', type=float, default=0.20)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    pairs = pairs_from_manifest(os.path.join(a.root, 'manifests', 'test.csv'))
    ds = TrainSet(ns, crop_size=0, pairs=pairs,
                  y0_cache=os.path.join(a.root, a.cache_name, 'test'), split='Test')
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(pairs))
    eps_energy = json.load(open(os.path.join(a.out_root, 'action_upper',
                                             'energy.json')))['eps_energy']

    def load(path):
        m = V3A1Refiner().to(dev)
        m.load_state_dict(torch.load(path, map_location=dev)['model'])
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    r1 = load(os.path.join(a.root, R1_CK))
    v2 = load(os.path.join(a.out_root, 'verifier_s42',
                           'checkpoint_%05d.pt' % a.step))

    conds = ['Base',
             'R1-correct', 'R1-dark', 'R1-noise-gaussian', 'R1-noise-uniform',
             'V3A2-correct', 'V3A2-dark', 'V3A2-noise-gaussian',
             'V3A2-noise-uniform', 'V3A2-mismatch', 'V3A2-corrupt',
             'V3A2-q0', 'V3A2-q1', 'V3A2-ActionOptimalGate']
    res, extra = {}, {}
    for i in range(len(pairs)):
        _n, lr, hr, ref, y0, _m = ds._load(i)
        mis = __import__('dataset.lolv2real_v3a', fromlist=['x']).read_model_image(
            ds.ref_map[os.path.basename(pairs[perm[i]][1])], size=hr.shape[-2:])
        y0d, hrd, lowd = y0[None].to(dev), hr[None].to(dev), lr[None].to(dev)
        rd = ref[None].to(dev)
        refd = ref.to(dev)
        ngen = torch.Generator(device=dev).manual_seed(1234 + i)
        noise_g = (refd + a.noise_sigma * torch.randn(refd.shape, generator=ngen,
                                                      device=dev)).clamp(-1, 1).cpu()
        noise_u = (refd + 0.5 * (torch.rand(refd.shape, generator=ngen,
                                            device=dev) * 2 - 1)).clamp(-1, 1).cpu()
        dark = rescale_reference(ref, 0.5)
        cor = splice_corrupt(ref, mis, np.random.default_rng(1234 + i))
        sid = os.path.basename(pairs[i][1])
        r = dict(Base=metrics(y0d, hrd))
        with torch.no_grad():
            refs = {'correct': ref, 'dark': dark, 'noise-gaussian': noise_g,
                    'noise-uniform': noise_u, 'mismatch': mis, 'corrupt': cor}
            # R1 has no verifier: force q_v = 1 so it is exactly the proposal
            for tag in ('correct', 'dark', 'noise-gaussian', 'noise-uniform'):
                o, _ = r1(y0d, refs[tag][None].to(dev), low=lowd, force_qv=1.0)
                r['R1-%s' % tag] = metrics(o, hrd)
            for tag, rr in refs.items():
                o, aux = v2(y0d, rr[None].to(dev), low=lowd)
                r['V3A2-%s' % tag] = metrics(o, hrd)
                if tag == 'correct':
                    D, gate_v2, delta = aux['gate_final'], aux['gate_v2'], aux['delta']
                    q_star_ref = D
                    qv_c = float(aux['q_v'].mean())
            o, _ = v2(y0d, rd, low=lowd, force_qv=0.0)
            r['V3A2-q0'] = metrics(o, hrd)
            o, _ = v2(y0d, rd, low=lowd, force_qv=1.0)
            r['V3A2-q1'] = metrics(o, hrd)
            # action-optimal diagnostic on the FROZEN proposal's correction
            _sr, auxp = v2.proposal(y0d, rd)
            Dp = auxp['gate'] * auxp['delta']
            q_opt, _e = action_optimal_gate(y0d, hrd, Dp)
            qf = F.interpolate(q_opt, size=y0d.shape[-2:], mode='bilinear',
                               align_corners=False)
            r['V3A2-ActionOptimalGate'] = metrics(y0d + qf * Dp, hrd)
        res[sid] = r
        extra[sid] = dict(q_v_correct=qv_c, q_opt_mean=float(q_opt.mean()))
        if (i + 1) % 25 == 0:
            print('  %d/%d' % (i + 1, len(pairs)))

    sids = sorted(res)
    col = lambda c: np.array([res[s][c][0] for s in sids])
    b = col('Base').mean()
    print()
    print('══ V3-A.2  LOL-v2-real Test Nano-v2 (n=%d, step=%d) ══' % (len(sids), a.step))
    print('  %-28s %10s %10s' % ('condition', 'PSNR(rgb)', 'delta'))
    for c in conds:
        print('  %-28s %10.4f %+10.4f' % (c, col(c).mean(), col(c).mean() - b))
    print()
    d = lambda c: col(c) - col('Base')
    print('  V3A2-correct - R1-correct   = %+.4f  (plan: >= -0.03)'
          % (col('V3A2-correct') - col('R1-correct')).mean())
    print('  V3A2-dark    - R1-dark      = %+.4f  (plan: >= +0.20)'
          % (col('V3A2-dark') - col('R1-dark')).mean())
    print('  V3A2-noise-g - R1-noise-g   = %+.4f'
          % (col('V3A2-noise-gaussian') - col('R1-noise-gaussian')).mean())
    print('  V3A2-q0 - Base (must be 0)  = %+.4f' % (col('V3A2-q0') - col('Base')).mean())
    print('  V3A2-q1 - R1-correct        = %+.4f'
          % (col('V3A2-q1') - col('R1-correct')).mean())
    print('  ActionOptimalGate - R1-correct = %+.4f'
          % (col('V3A2-ActionOptimalGate') - col('R1-correct')).mean())
    qv = np.array([extra[s]['q_v_correct'] for s in sids])
    print('  q_v(correct) mean %.3f' % qv.mean())
    print()
    for c in ('R1-correct', 'V3A2-correct', 'V3A2-dark', 'R1-dark'):
        dd = d(c)
        print('  %-14s median %+7.4f  better %3d/%d  worst %+7.3f  <-1dB %d'
              % (c, np.median(dd), int((dd > 0).sum()), len(dd), dd.min(),
                 int((dd < -1).sum())))

    out = os.path.join(a.out_root, 'eval')
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, 'per_image_v3a2_s42.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id'] + [c + '_psnr' for c in conds]
                   + ['q_v_correct', 'q_opt_mean'])
        for s in sids:
            w.writerow([s] + [res[s][c][0] for c in conds]
                       + ['%.6f' % extra[s]['q_v_correct'],
                          '%.6f' % extra[s]['q_opt_mean']])
    json.dump(dict(step=a.step, n=len(sids), eps_energy=eps_energy,
                   psnr={c: float(col(c).mean()) for c in conds},
                   delta={c: float(d(c).mean()) for c in conds},
                   q_v_correct=float(qv.mean())),
              open(os.path.join(out, 'summary_v3a2.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    print('\n-> %s' % p)


if __name__ == '__main__':
    main()
