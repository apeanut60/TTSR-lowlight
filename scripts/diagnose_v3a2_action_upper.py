#!/usr/bin/env python
"""V3-A.2 §13 Go/No-Go: is there headroom for a gate at all?

Uses the frozen R1 proposal plus a GT-derived action-optimal gate:

    Y_opt = Y0 + up(q_opt) * (g_v2 * delta)

If that is not clearly better than the plain R1-correct output, no verifier can
help and the verifier route should stop.

Also reports the corrected old-style GT usefulness gate (which must now include
g_v2; the earlier +0.5821 figure omitted it and is deprecated).
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
from model.V3A1Refiner import V3A1Refiner                         # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a1_runtime import usefulness_at                            # noqa: E402
from v3a2_runtime import action_optimal_gate, energy_threshold    # noqa: E402

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def load_proposal(root, device):
    m = V3A1Refiner().to(device)
    m.load_state_dict(torch.load(os.path.join(root, R1_CK), map_location=device)['model'])
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
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out_root', default='/root/data/experiments/v3a2_lolv2real')
    a = ap.parse_args(_CLI)
    os.makedirs(os.path.join(a.out_root, 'action_upper'), exist_ok=True)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    m = load_proposal(a.root, a.device)

    # --- energy threshold from the TRAIN split (plan §5) -------------------
    tp = pairs_from_manifest(os.path.join(a.root, 'manifests', 'refiner_train.csv'))
    tds = TrainSet(ns, crop_size=0, pairs=tp,
                   y0_cache=os.path.join(a.root, a.cache_name, 'refiner_train'),
                   split='Train')
    energies = []
    with torch.no_grad():
        for i in range(len(tp)):
            _n, _lr, hr, ref, y0, _mm = tds._load(i)
            _sr, aux = m.proposal(y0[None].to(a.device), ref[None].to(a.device))
            _q, e = action_optimal_gate(y0[None].to(a.device), hr[None].to(a.device),
                                        (aux['gate'] * aux['delta']))
            energies.append(e.cpu())
            if (i + 1) % 150 == 0:
                print('  energy %d/%d' % (i + 1, len(tp)))
    eps_energy = energy_threshold(energies)
    json.dump(dict(eps_energy=eps_energy, method='p10 of per-pixel proposal energy',
                   split='refiner_train', n=len(tp),
                   manifest_sha256=_sha(os.path.join(a.root, 'manifests',
                                                     'refiner_train.csv'))),
              open(os.path.join(a.out_root, 'action_upper', 'energy.json'), 'w',
                   encoding='utf-8'), indent=2, sort_keys=True)
    print('eps_energy = %.6e (p10 over %d train images)' % (eps_energy, len(tp)))

    # --- test evaluation ---------------------------------------------------
    ep = pairs_from_manifest(os.path.join(a.root, 'manifests', 'test.csv'))
    eds = TrainSet(ns, crop_size=0, pairs=ep,
                   y0_cache=os.path.join(a.root, a.cache_name, 'test'), split='Test')
    tau = json.load(open(os.path.join(a.root, 'tau_%s.json' % a.cache_name)))[ 'tau']
    acc = {k: [] for k in ('Base', 'R1-correct', 'R1-ActionOptimalGate',
                           'R1-GT-usefulness-gate(fixed)')}
    qstats = dict(q_opt=[], energy_above=[], q_v_init=[])
    with torch.no_grad():
        for i in range(len(ep)):
            _n, _lr, hr, ref, y0, _mm = eds._load(i)
            y0d, hrd = y0[None].to(a.device), hr[None].to(a.device)
            rd = ref[None].to(a.device)
            sr, aux = m.proposal(y0d, rd)
            D = aux['gate'] * aux['delta']
            acc['Base'].append(metrics(y0d, hrd)[0])
            acc['R1-correct'].append(metrics(sr, hrd)[0])
            q_opt, e = action_optimal_gate(y0d, hrd, D)
            qf = F.interpolate(q_opt, size=y0d.shape[-2:], mode='bilinear',
                               align_corners=False)
            acc['R1-ActionOptimalGate'].append(metrics(y0d + qf * D, hrd)[0])
            qs, _ = usefulness_at(y0d, rd, hrd, tau)
            qsf = F.interpolate(qs, size=y0d.shape[-2:], mode='bilinear',
                                align_corners=False)
            # ``D`` already contains g_v2, so the old-style usefulness gate must
            # NOT multiply by the proposal gate again (doing so scored g_v2^2
            # and produced the deprecated +0.0592 figure).
            acc['R1-GT-usefulness-gate(fixed)'].append(
                metrics(y0d + qsf * D, hrd)[0])
            qstats['q_opt'].append(float(q_opt.mean()))
            qstats['energy_above'].append(float((e > eps_energy).float().mean()))
            if (i + 1) % 25 == 0:
                print('  test %d/%d' % (i + 1, len(ep)))

    b = float(np.mean(acc['Base']))
    print()
    print('══ V3-A.2 action-optimal upper diagnostic (n=%d) ══' % len(ep))
    for k in acc:
        m_ = float(np.mean(acc[k]))
        print('  %-32s %.4f  (%+.4f)' % (k, m_, m_ - b))
    head = float(np.mean(acc['R1-ActionOptimalGate'])) - float(np.mean(acc['R1-correct']))
    print()
    print('  headroom (ActionOptimalGate - R1-correct) = %+.4f dB' % head)
    print('  q_opt mean %.3f | fraction of pixels above eps_energy %.3f'
          % (float(np.mean(qstats['q_opt'])), float(np.mean(qstats['energy_above']))))
    verdict = 'GO' if head >= 0.15 else ('NO-GO' if head <= 0 else 'MARGINAL')
    print('  §13 threshold: headroom >= +0.15 dB  ->  **%s**' % verdict)
    json.dump(dict(n=len(ep), eps_energy=eps_energy,
                   psnr={k: float(np.mean(v)) for k, v in acc.items()},
                   headroom=head, verdict=verdict,
                   q_opt_mean=float(np.mean(qstats['q_opt'])),
                   energy_above_frac=float(np.mean(qstats['energy_above']))),
              open(os.path.join(a.out_root, 'action_upper', 'summary.json'), 'w',
                   encoding='utf-8'), indent=2, sort_keys=True)


def _sha(p):
    from local_refine_runtime import sha256
    return sha256(p)


if __name__ == '__main__':
    main()
