#!/usr/bin/env python
"""V3-A.4 §15: dev64 evaluation of C0 / C1-raw / C2-norm.

Every metric here is computed ON dev64 (the earlier round read train-batch
statistics out of train.jsonl and presented them next to dev numbers).
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

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest   # noqa: E402
from local_refine_runtime import metrics                          # noqa: E402
from model.V3A4Verifier import V3A4Refiner                        # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a2_runtime import action_optimal_gate                      # noqa: E402
from v3a4_runtime import (load_r1_proposal_strict, masked_mae,    # noqa: E402
                          masked_rmse, masked_smooth_l1, pixel_corr)
from v3a_runtime import exposure_gain                             # noqa: E402

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
TAGS = {'none': 'C0_clean_s42', 'raw': 'C1_raw_clean_s42',
        'norm': 'C2_norm_clean_s42'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--root', default='/root/data/experiments/v3a4_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device

    split = json.load(open(os.path.join(a.root, 'splits', 'split.json'),
                           encoding='utf-8'))
    dset = set(split['dev'])
    rms = json.load(open(os.path.join(a.root, 'action_stats',
                                      'action_norm.json')))['rms_D']
    eps_e = json.load(open(os.path.join(a.root, 'action_stats',
                                        'energy.json')))['eps_energy']
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    rows = sorted([p for p in pairs_from_manifest(
        os.path.join(a.src_root, 'manifests', 'refiner_train.csv'))
        if p[0] in dset], key=lambda p: p[0])
    ds = TrainSet(ns, crop_size=0, pairs=rows,
                  y0_cache=os.path.join(a.src_root, a.cache_name,
                                        'refiner_train'), split='Train')
    mmap = json.load(open(os.path.join(a.root, 'mappings',
                                       'mismatch_dev_64.json'), encoding='utf-8'))

    def build(mode, path=None):
        m = V3A4Refiner(mode, action_rms=rms).to(dev).eval()
        if path:
            m.load_state_dict(torch.load(path, map_location=dev)['model'])
        load_r1_proposal_strict(m, os.path.join(a.src_root, R1_CK), dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    arms = {}
    for mode, tag in TAGS.items():
        p = os.path.join(a.root, tag, 'checkpoint_%05d.pt' % a.step)
        if os.path.isfile(p):
            arms[mode] = build(mode, p)
    if 'none' not in arms:
        arms['none'] = build('none')
    r1 = build('none')

    states = ['correct', 'true_dark_g0.5', 'mismatch']
    acc = {}
    gate = {}
    perimg = []
    for i, (name, low, high) in enumerate(rows):
        _n, lr, hr, ref, y0, _mm = ds._load(i)
        y0d, hrd, lowd = y0[None].to(dev), hr[None].to(dev), lr[None].to(dev)
        donor = ds.ref_map[mmap[os.path.basename(low)]]
        from dataset.lolv2real_v3a import read_model_image
        mis = read_model_image(donor, size=hr.shape[-2:])
        refs = {'correct': ref, 'true_dark_g0.5': exposure_gain(ref, 0.5),
                'mismatch': mis}
        n0 = metrics(y0d, hrd)[0]
        acc.setdefault('Base', {})[name] = n0
        rec = dict(sample_id=os.path.basename(low), base_psnr=n0)
        for s, r in refs.items():
            rd = r[None].to(dev)
            with torch.no_grad():
                o, _ = r1(y0d, rd, low=lowd, force_qv=1.0)
                acc.setdefault('R1', {}).setdefault(s, []).append(metrics(o, hrd)[0])
                # ActionOptimal reference for this state
                _sr, auxp = r1.proposal(y0d, rd)
                D = auxp['gate'] * auxp['delta']
                q_opt, e = action_optimal_gate(y0d, hrd, D)
                qf = F.interpolate(q_opt, size=y0d.shape[-2:], mode='bilinear',
                                   align_corners=False)
                acc.setdefault('AO', {}).setdefault(s, []).append(
                    metrics(y0d + qf * D, hrd)[0])
                mask = (e > eps_e).float()
                for mode, m in arms.items():
                    o, aux = m(y0d, rd, low=lowd)
                    acc.setdefault(mode, {}).setdefault(s, []).append(
                        metrics(o, hrd)[0])
                    if mode != 'none':
                        g = gate.setdefault(mode, {}).setdefault(s, dict(
                            q=[[], []], corr=[], mae=[], rmse=[], sl1=[],
                            qopt=[[], []]))
                        g['corr'].append(pixel_corr(aux['q_v4'], q_opt, mask))
                        g['mae'].append(masked_mae(aux['q_v4'], q_opt, mask))
                        g['rmse'].append(masked_rmse(aux['q_v4'], q_opt, mask))
                        g['sl1'].append(float(masked_smooth_l1(aux['q_v4'], q_opt, mask)))
                        g['q'][0].append(float(aux['q_v4'].mean()))
                        g['q'][1].append(float(aux['q_v4'].mean()))
                        g['qopt'][0].append(float(q_opt.mean()))
                        g['qopt'][1].append(float(q_opt.mean()))
                        # per-image spatial correlation
                        cc = []
                        for b in range(q_opt.shape[0]):
                            cc.append(pixel_corr(aux['q_v4'][b:b+1], q_opt[b:b+1],
                                                 mask[b:b+1]))
                        g.setdefault('spatial_corr', []).extend(cc)
                    rec['%s_%s' % (mode, s)] = metrics(o, hrd)[0]
                o, _ = r1(y0d, rd, low=lowd, force_qv=0.0)
                rec['R1bypass_%s' % s] = metrics(o, hrd)[0]
        perimg.append(rec)
        if (i + 1) % 16 == 0:
            print('  %d/%d' % (i + 1, len(rows)))

    def mean(arm, s):
        v = np.asarray(acc[arm][s], dtype=float)
        return float(v.mean())

    b = float(np.mean(list(acc['Base'].values())))
    out = dict(n=len(rows), base=b, eps_energy=eps_e, rms_D=rms,
               psnr={}, delta_vs_base={}, delta_vs_r1={}, headroom={}, capture={})
    print()
    print('══ V3-A.4 dev64 (n=%d, step=%d)  Base %.4f ══' % (len(rows), a.step, b))
    print('  %-22s %9s %9s %9s %9s %9s | %8s %8s'
          % ('state', 'R1', 'C0', 'C1-raw', 'C2-norm', 'AO',
             'C2-R1', 'capture'))
    for s in states:
        r1v = mean('R1', s)
        c0v = mean('none', s)
        c1v = mean('raw', s) if 'raw' in arms else float('nan')
        c2v = mean('norm', s) if 'norm' in arms else float('nan')
        aov = mean('AO', s)
        head = aov - r1v
        cap = (c2v - r1v) / head if abs(head) > 1e-9 else float('nan')
        out['psnr'][s] = dict(R1=r1v, C0=c0v, C1_raw=c1v, C2_norm=c2v, AO=aov)
        out['headroom'][s] = head
        out['capture'][s] = cap
        print('  %-22s %9.4f %9.4f %9.4f %9.4f %9.4f | %+8.4f %8.3f  (head %+.4f)'
              % (s, r1v, c0v, c1v, c2v, aov, c2v - r1v, cap, head))
    r1c = mean('R1', 'correct')
    out['correct'] = dict(
        R1=r1c, C0=mean('none', 'correct'),
        C1_raw=mean('raw', 'correct') if 'raw' in arms else None,
        C2_norm=mean('norm', 'correct') if 'norm' in arms else None)
    print()
    print('  correct preservation: C0 %+.4f / C1 %+.4f / C2 %+.4f  (vs R1, limit -0.03)'
          % (out['correct']['C0'] - r1c,
             (out['correct']['C1_raw'] or float('nan')) - r1c,
             (out['correct']['C2_norm'] or float('nan')) - r1c))
    if gate:
        print()
        print('  %-8s %-22s %8s %8s %8s %8s' % ('arm', 'state', 'corr', 'MAE', 'RMSE', 'SmoothL1'))
        for mode in ('raw', 'norm'):
            for s in states:
                if mode not in gate:
                    continue
                g = gate[mode][s]
                print('  %-8s %-22s %8.3f %8.3f %8.3f %8.3f'
                      % (mode, s, float(np.nanmean(g['corr'])),
                         float(np.mean(g['mae'])), float(np.mean(g['rmse'])),
                         float(np.mean(g['sl1']))))
        out['gate'] = {mode: {s: dict(
            corr=float(np.nanmean(gate[mode][s]['corr'])),
            mae=float(np.mean(gate[mode][s]['mae'])),
            rmse=float(np.mean(gate[mode][s]['rmse'])),
            sl1=float(np.mean(gate[mode][s]['sl1'])),
            spatial_corr_mean=float(np.nanmean(gate[mode][s].get('spatial_corr', [np.nan]))),
            spatial_corr_median=float(np.nanmedian(gate[mode][s].get('spatial_corr', [np.nan]))),
            spatial_corr_p25=float(np.nanpercentile(gate[mode][s].get('spatial_corr', [np.nan]), 25)),
            spatial_corr_p75=float(np.nanpercentile(gate[mode][s].get('spatial_corr', [np.nan]), 75)),
        ) for s in states} for mode in gate}

    os.makedirs(os.path.join(a.root, 'dev_eval'), exist_ok=True)
    jp = os.path.join(a.root, 'dev_eval', 'summary_dev.json')
    json.dump(out, open(jp, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    cp = os.path.join(a.root, 'dev_eval', 'per_image_dev.csv')
    with open(cp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(perimg[0].keys()))
        w.writeheader()
        w.writerows(perimg)
    print('\n-> %s\n-> %s' % (jp, cp))


if __name__ == '__main__':
    main()
