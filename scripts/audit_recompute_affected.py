#!/usr/bin/env python
"""Retrospective audit: recompute every reported number that passed through a
buggy code path.

Two bugs touched diagnostics:
  * the GT-usefulness gate was applied without the V2 gate in V3-A
    (``Y0 + q_star * delta``) and with it twice in V3-A.2
    (``Y0 + g_v2 * q_star * g_v2 * delta``). The correct form is
    ``Y0 + g_v2 * q_star * delta``.
  * the V3-A "oracle gate" had the same omission.

Main results (N0 / Self / Nano in sections 17-22) do not touch these paths and
are not recomputed here.
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

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest   # noqa: E402
from local_refine_runtime import metrics                          # noqa: E402
from model.V3ARefiner import V3ARefiner                           # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a_runtime import mismatch_permutation                      # noqa: E402

V3A = '/root/data/experiments/v3a_lolv2real'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/data/experiments/audit_recompute.json')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device

    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = 'nanobanana_ref_v2'
    pairs = pairs_from_manifest(os.path.join(V3A, 'manifests', 'test.csv'))
    ds = TrainSet(ns, crop_size=0, pairs=pairs,
                  y0_cache=os.path.join(V3A, 'cache_y0', 'test'), split='Test')
    tau = json.load(open(os.path.join(V3A, 'tau.json')))['tau']

    m = V3ARefiner().to(dev)
    m.load_state_dict(torch.load(os.path.join(
        V3A, 'R2_hallucination_aware_s42', 'checkpoint_03000.pt'),
        map_location=dev)['model'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    from v3a_runtime import usefulness

    acc = {k: [] for k in ('Base', 'R2-correct', 'R2-qv1(=full correction)',
                           'delta only (no gate)',
                           'gate*delta (no q_star)',
                           'gate*0.25*delta (constant attenuation)',
                           'R2-GT-gate(old, no g_v2)',
                           'R2-GT-gate(correct)',
                           'R2-GT-gate(g_v2 twice)')}
    qstat = []
    for i in range(len(pairs)):
        _n, lr, hr, ref, y0, _m = ds._load(i)
        y0d, hrd, rd = y0[None].to(dev), hr[None].to(dev), ref[None].to(dev)
        with torch.no_grad():
            sr, aux = m(y0d, rd)
            acc['Base'].append(metrics(y0d, hrd)[0])
            acc['R2-correct'].append(metrics(sr, hrd)[0])
            q_star, _d = usefulness(y0d, rd, hrd, tau)      # H/8
            qf = F.interpolate(q_star, size=y0d.shape[-2:], mode='bilinear',
                               align_corners=False)
            g, d = aux['gate'], aux['delta']
            qstat.append((float(q_star.mean()), float(g.mean()), float(d.abs().mean())))
            acc['R2-qv1(=full correction)'].append(metrics(y0d + g * d, hrd)[0])
            acc['delta only (no gate)'].append(metrics(y0d + d, hrd)[0])
            acc['gate*delta (no q_star)'].append(metrics(y0d + g * d, hrd)[0])
            acc['gate*0.25*delta (constant attenuation)'].append(
                metrics(y0d + 0.25 * g * d, hrd)[0])
            acc['R2-GT-gate(old, no g_v2)'].append(metrics(y0d + qf * d, hrd)[0])
            acc['R2-GT-gate(correct)'].append(metrics(y0d + g * qf * d, hrd)[0])
            acc['R2-GT-gate(g_v2 twice)'].append(metrics(y0d + g * qf * g * d, hrd)[0])
        if (i + 1) % 25 == 0:
            print('  %d/%d' % (i + 1, len(pairs)))

    b = float(np.mean(acc['Base']))
    out = dict(n=len(pairs), source='v3a_lolv2real/R2_hallucination_aware_s42/step3000')
    print('══ V3-A diagnostic recomputation (n=%d) ══' % len(pairs))
    for k, v in acc.items():
        out[k] = float(np.mean(v))
        print('  %-28s %.4f  (%+.4f)' % (k, out[k], out[k] - b))
    qm, gm, dm = np.mean(qstat, axis=0)
    out['q_star_mean'] = float(qm)
    out['gate_mean'] = float(gm)
    out['delta_absmean'] = float(dm)
    print('  q_star mean %.3f | g_v2 mean %.3f | |delta| mean %.4f' % (qm, gm, dm))
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('-> %s' % a.out)


if __name__ == '__main__':
    main()
