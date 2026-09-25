#!/usr/bin/env python
"""V3-A.1 formal comparison: 11 conditions (plan §14) + usefulness-gate diagnostic.

  Base, R1-correct, R1-mismatch, R1-corrupt,
  R2-correct, R2-mismatch, R2-corrupt, R2-bypass,
  R2-qv1, R2-qv0, R2-GT-usefulness-gate

R1 has no verifier, so its conditions run the proposal path directly; R2's run
the full model with the true low input. The GT-usefulness gate is a diagnostic
(the plan forbids calling it a "perfect oracle").
"""

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

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest, read_model_image  # noqa: E402
from local_refine_runtime import metrics                            # noqa: E402
from model.V3A1Refiner import V3A1Refiner                           # noqa: E402
from option import parser as option_parser                          # noqa: E402
from v3a1_runtime import usefulness_at                              # noqa: E402
from v3a_runtime import (mismatch_permutation, rescale_reference,   # noqa: E402
                         splice_corrupt)

R1_DIR = 'R1_v2stable_naive_s42'
R2_DIR = 'R2_v2stable_verifier_s42'


def load(path, device):
    ck = torch.load(path, map_location=device)
    m = V3A1Refiner().to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    device = a.device

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    pairs = pairs_from_manifest(os.path.join(a.root, 'manifests', 'test.csv'))
    cache = os.path.join(a.root, a.cache_name, 'test')
    ds = TrainSet(ns, crop_size=0, pairs=pairs, y0_cache=cache, split='Test')
    tau = json.load(open(os.path.join(a.root, 'tau_%s.json' % a.cache_name),
                         encoding='utf-8'))['tau']
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(pairs))

    r1 = load(os.path.join(a.root, R1_DIR, 'checkpoint_%05d.pt' % a.step), device)
    r2 = load(os.path.join(a.root, R2_DIR, 'checkpoint_%05d.pt' % a.step), device)

    conds = ['Base', 'R1-correct', 'R1-mismatch', 'R1-corrupt',
             'R2-correct', 'R2-mismatch', 'R2-corrupt', 'R2-bypass',
             'R2-qv1', 'R2-qv0', 'R2-GT-usefulness-gate',
             # extra: the constructions that are ACTUALLY harmful on LOL
             # (mismatch/corrupt are neutral-to-helpful here), plus one the
             # verifier was never trained on, as a generalisation probe
             'R1-dark', 'R2-dark', 'R2-noise']
    res, extra = {}, {}
    for i in range(len(pairs)):
        _n, lr, hr, ref, y0, _m = ds._load(i)
        mis = read_model_image(ds.ref_map[os.path.basename(pairs[perm[i]][1])],
                               size=hr.shape[-2:])
        cor = splice(ref, mis, i)
        dark = rescale_reference(ref, 0.5)
        noise = torch.rand(ref.shape, generator=torch.Generator().manual_seed(i)) * 2 - 1
        y0d, hrd = y0[None].to(device), hr[None].to(device)
        lowd = lr[None].to(device)
        sid = os.path.basename(pairs[i][1])
        r = {}
        r['Base'] = metrics(y0d, hrd)
        with torch.no_grad():
            for tag, refi in (('correct', ref), ('mismatch', mis), ('corrupt', cor)):
                o, _ = r1.proposal(y0d, refi[None].to(device))
                r['R1-%s' % tag] = metrics(o, hrd)
            o, _ = r1.proposal(y0d, dark[None].to(device))
            r['R1-dark'] = metrics(o, hrd)
            o, _ = r2(y0d, None)
            r['R2-bypass'] = metrics(o, hrd)
            qs = {}
            for tag, refi in (('correct', ref), ('mismatch', mis), ('corrupt', cor)):
                o, aux = r2(y0d, refi[None].to(device), low=lowd)
                r['R2-%s' % tag] = metrics(o, hrd)
                qs[tag] = aux
            o, _ = r2(y0d, dark[None].to(device), low=lowd)
            r['R2-dark'] = metrics(o, hrd)
            o, _ = r2(y0d, noise.to(device)[None], low=lowd)
            r['R2-noise'] = metrics(o, hrd)
            o1, _ = r2(y0d, ref[None].to(device), low=lowd, force_qv=1.0)
            r['R2-qv1'] = metrics(o1, hrd)
            o0, _ = r2(y0d, ref[None].to(device), low=lowd, force_qv=0.0)
            r['R2-qv0'] = metrics(o0, hrd)
            q_star, _d = usefulness_at(y0d, ref[None].to(device), hrd, tau)
            qf = F.interpolate(q_star, size=y0d.shape[-2:], mode='bilinear',
                               align_corners=False)
            r['R2-GT-usefulness-gate'] = metrics(
                y0d + qf * qs['correct']['delta'], hrd)
        res[sid] = r
        extra[sid] = dict(
            q_v_correct=float(qs['correct']['q_v'].mean()),
            q_v_mismatch=float(qs['mismatch']['q_v'].mean()),
            gate_v2=float(qs['correct']['gate_v2'].mean()),
            q_star=float(q_star.mean()),
            corr_mean=float((qs['correct']['gate_final'] *
                             qs['correct']['delta']).abs().mean()))
        if (i + 1) % 25 == 0:
            print('  %d/%d' % (i + 1, len(pairs)))

    sids = sorted(res)
    col = lambda c, k=0: np.array([res[s][c][k] for s in sids])
    print()
    print('══ V3-A.1  LOL-v2-real Test Nano-v2 (n=%d, step=%d) ══' % (len(sids), a.step))
    print('  %-24s %10s %10s' % ('condition', 'PSNR(rgb)', 'SSIM(Y)'))
    for c in conds:
        print('  %-24s %10.4f %10.4f' % (c, col(c, 0).mean(), col(c, 1).mean()))
    base = col('Base', 0)
    print()
    for c in conds[1:]:
        d = col(c, 0) - base
        print('  %-24s %+8.4f  median %+7.4f  better %3d/%d  worst %+7.3f  <-1dB %d'
              % (c, d.mean(), np.median(d), int((d > 0).sum()), len(d),
                 d.min(), int((d < -1.0).sum())))

    def g(c):
        return col(c, 0).mean() - base.mean()

    print()
    print('  Delta_scheme  R2-correct - Base      = %+.4f' % g('R2-correct'))
    print('  Delta_verify  R2-correct - R1-correct= %+.4f'
          % (col('R2-correct', 0) - col('R1-correct', 0)).mean())
    print('  harm_R1 = R1-mismatch - Base         = %+.4f' % g('R1-mismatch'))
    print('  harm_R2 = R2-mismatch - Base         = %+.4f' % g('R2-mismatch'))
    print('  safety gain (harm_R2 - harm_R1)      = %+.4f'
          % (g('R2-mismatch') - g('R1-mismatch')))
    print('  R2-corrupt - R1-corrupt              = %+.4f'
          % (col('R2-corrupt', 0) - col('R1-corrupt', 0)).mean())
    print('  R2-qv0 - Base (must be 0)            = %+.4f' % g('R2-qv0'))
    print('  R2-qv1 - R1-correct (proposal only)  = %+.4f'
          % (col('R2-qv1', 0) - col('R1-correct', 0)).mean())
    print('  GT-usefulness-gate - R2-correct      = %+.4f'
          % (col('R2-GT-usefulness-gate', 0) - col('R2-correct', 0)).mean())
    qc = np.array([extra[s]['q_v_correct'] for s in sids])
    qm = np.array([extra[s]['q_v_mismatch'] for s in sids])
    qs_ = np.array([extra[s]['q_star'] for s in sids])
    print('  q_v: correct %.3f, mismatch %.3f  |  corr(q_v, q_star) = %.3f'
          % (qc.mean(), qm.mean(), float(np.corrcoef(qc, qs_)[0, 1])))

    out = os.path.join(a.root, 'eval')
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, 'per_image_v3a1_s42.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id'] + [x for c in conds for x in (c + '_psnr', c + '_ssim_y')]
                   + list(next(iter(extra.values())).keys()))
        for s in sids:
            w.writerow([s] + [v for c in conds for v in res[s][c][:2]]
                       + ['%.6f' % extra[s][k] for k in extra[s]])
    json.dump(dict(step=a.step, n=len(sids), tau=tau, cache=a.cache_name,
                   psnr={c: float(col(c, 0).mean()) for c in conds},
                   ssim_y={c: float(col(c, 1).mean()) for c in conds},
                   delta={c: g(c) for c in conds[1:]},
                   q_v=dict(correct=float(qc.mean()), mismatch=float(qm.mean()),
                            corr_with_qstar=float(np.corrcoef(qc, qs_)[0, 1]))),
              open(os.path.join(out, 'summary_v3a1.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    print('\n-> %s' % p)


def splice(ref, donor, i):
    from v3a_runtime import splice_corrupt
    return splice_corrupt(ref, donor, np.random.default_rng(1234 + i))


if __name__ == '__main__':
    main()
