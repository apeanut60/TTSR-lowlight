#!/usr/bin/env python
"""V3-A.3 §12-§13: dev64 evaluation of C0 vs C1.

Reports, per locked corruption state: R1-S, C0-S, C1-S, the state's own
ActionOptimal headroom, and the capture ratio -- computed strictly within the
same state (never across conditions).
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
from model.V3A3Verifier import V3A3Refiner                        # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a2_runtime import action_optimal_gate                      # noqa: E402
from v3a_runtime import (contrast_compress, exposure_gain,        # noqa: E402
                         mismatch_permutation, splice_corrupt)

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def load_proposal_into(m, src_root, dev):
    sd = torch.load(os.path.join(src_root, R1_CK), map_location=dev)['model']
    prop = {k[len('proposal.'):]: v for k, v in sd.items() if k.startswith('proposal.')}
    m.proposal.load_state_dict(prop, strict=True)
    if float(m.proposal.c_out.weight.abs().max()) == 0.0:
        raise SystemExit('proposal not loaded (c_out still zero)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--root', default='/root/data/experiments/v3a3_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--noise_sigma', type=float, default=0.20)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device

    split = json.load(open(os.path.join(a.root, 'splits', 'split.json'),
                           encoding='utf-8'))
    dev_ids = set(split['dev'])
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    rows = sorted([p for p in pairs_from_manifest(
        os.path.join(a.src_root, 'manifests', 'refiner_train.csv'))
        if p[0] in dev_ids], key=lambda p: p[0])
    ds = TrainSet(ns, crop_size=0, pairs=rows,
                  y0_cache=os.path.join(a.src_root, a.cache_name, 'refiner_train'),
                  split='Train')
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(rows))
    eps_e = json.load(open(os.path.join(a.root, 'action_upper',
                                        'energy.json')))['eps_energy']

    def load(path, use_action, full=True):
        m = V3A3Refiner(use_action=use_action).to(dev)
        if full:
            # C0/C1 checkpoints are V3A3Refiner dicts (verifier head0 matches)
            m.load_state_dict(torch.load(path, map_location=dev)['model'])
        # R1's checkpoint is a V3A1Refiner dict (160-ch verifier); only its
        # proposal is needed because R1 is evaluated with force_qv=1.
        load_proposal_into(m, a.src_root, dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    r1 = load(os.path.join(a.src_root, R1_CK), True, full=False)
    c0 = load(os.path.join(a.root, 'C0_v3a2_control_s42',
                           'checkpoint_%05d.pt' % a.step), False)
    c1 = load(os.path.join(a.root, 'C1_v3a3_action_s42',
                           'checkpoint_%05d.pt' % a.step), True)

    states = ['correct', 'true_dark_g0.5', 'true_dark_g0.7', 'true_bright_g1.3',
              'contrast_compress_0.5', 'gaussian_noise_s0.20', 'mismatch', 'corrupt']
    acc = {k: [] for k in ['Base'] + ['%s|%s' % (arm, s) for arm in
                                      ('R1', 'C0', 'C1', 'AO') for s in states]}
    qv = {arm: {s: [] for s in states} for arm in ('C0', 'C1')}
    qo = {s: [] for s in states}
    with torch.no_grad():
        for i in range(len(rows)):
            _n, lr, hr, ref, y0, _mm = ds._load(i)
            y0d, hrd, lowd = y0[None].to(dev), hr[None].to(dev), lr[None].to(dev)
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
            for s, r in refs.items():
                rd = r[None].to(dev)
                o, _ = r1(y0d, rd, low=lowd, force_qv=1.0)
                acc['R1|%s' % s].append(metrics(o, hrd)[0])
                for arm, mdl in (('C0', c0), ('C1', c1)):
                    o, aux = mdl(y0d, rd, low=lowd)
                    acc['%s|%s' % (arm, s)].append(metrics(o, hrd)[0])
                    qv[arm][s].append(float(aux['q_v'].mean()))
                _sr, auxp = c1.proposal(y0d, rd)
                D = auxp['gate'] * auxp['delta']
                q_opt, _e = action_optimal_gate(y0d, hrd, D)
                qf = F.interpolate(q_opt, size=y0d.shape[-2:], mode='bilinear',
                                   align_corners=False)
                acc['AO|%s' % s].append(metrics(y0d + qf * D, hrd)[0])
                qo[s].append(float(q_opt.mean()))

    b = float(np.mean(acc['Base']))
    print('══ V3-A.3 dev64 (n=%d, step=%d)  Base %.4f ══' % (len(rows), a.step, b))
    print('  %-24s %9s %9s %9s %9s | %8s %8s'
          % ('state', 'R1', 'C0', 'C1', 'AO', 'C1-C0', 'capture'))
    out = dict(n=len(rows), base=b, eps_energy=eps_e, states={})
    for s in states:
        r1v = float(np.mean(acc['R1|%s' % s]))
        c0v = float(np.mean(acc['C0|%s' % s]))
        c1v = float(np.mean(acc['C1|%s' % s]))
        aov = float(np.mean(acc['AO|%s' % s]))
        head = aov - r1v
        cap = (c1v - r1v) / head if abs(head) > 1e-9 else float('nan')
        out['states'][s] = dict(r1=r1v, c0=c0v, c1=c1v, ao=aov, headroom=head,
                                capture=cap, c1_minus_c0=c1v - c0v,
                                q_opt=float(np.mean(qo[s])),
                                q_v_c0=float(np.mean(qv['C0'][s])),
                                q_v_c1=float(np.mean(qv['C1'][s])))
        print('  %-24s %9.4f %9.4f %9.4f %9.4f | %+8.4f %8.2f  (head %+.4f)'
              % (s, r1v, c0v, c1v, aov, c1v - c0v, cap, head))

    print()
    for arm in ('C0', 'C1'):
        c = float(np.mean(acc['%s|correct' % arm]))
        print('  %s-correct %.4f  (vs R1-correct %+.4f)'
              % (arm, c, c - float(np.mean(acc['R1|correct']))))
    print('  C1-correct - C0-correct = %+.4f'
          % (float(np.mean(acc['C1|correct'])) - float(np.mean(acc['C0|correct']))))
    print('  q_v correct: C0 %.3f  C1 %.3f' % (out['states']['correct']['q_v_c0'],
                                               out['states']['correct']['q_v_c1']))
    # last logged training row of each arm: the q_opt-learning evidence
    for arm, d in (('C0', 'C0_v3a2_control_s42'), ('C1', 'C1_v3a3_action_s42')):
        p = os.path.join(a.root, d, 'train.jsonl')
        if os.path.isfile(p):
            last = json.loads(open(p, encoding='utf-8').readlines()[-1])
            out['train_last_%s' % arm] = dict(
                corr_qv_qopt=last.get('correct_corr_qv_qopt'),
                masked_mae=last.get('correct_gate_masked_mae'),
                gap_pred=last.get('gap_pred'), gap_gt=last.get('gap_gt'))

    os.makedirs(os.path.join(a.root, 'dev_eval'), exist_ok=True)
    p = os.path.join(a.root, 'dev_eval', 'summary_dev.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('\n-> %s' % p)


if __name__ == '__main__':
    main()
