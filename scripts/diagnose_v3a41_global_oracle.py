#!/usr/bin/env python
"""V3-A.4.1 Diagnostic B (§10-§12): does a global scalar gate suffice?

Compares, per state:
    Base, R1 (q=1), Global-AO (one scalar q per image), Spatial-AO (H/4 gate)

    H_global  = PSNR(Global-AO)  - PSNR(R1)
    H_spatial = PSNR(Spatial-AO) - PSNR(R1)
    capture_global = H_global / H_spatial        (only when H_spatial > 0)

Read-only; the dev64 correct Spatial-AO must reproduce the V3-A.4 number.
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
from local_refine_runtime import metrics                          # noqa: E402
from model.V3A4Verifier import V3A4Refiner                        # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a41_runtime import global_action_optimal_gate, qopt_components  # noqa: E402
from v3a4_runtime import load_r1_proposal_strict                  # noqa: E402
from v3a_runtime import exposure_gain                             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
STATES = ('correct', 'true_dark_g0.5', 'mismatch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--root', default='/root/data/experiments/v3a41_target_audit')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--splits', default='dev,train')
    ap.add_argument('--coarse', default='1,2,4',
                    help='extra coarse gate resolutions (g x g blocks) for §12 B3')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device
    os.makedirs(os.path.join(a.root, 'oracle'), exist_ok=True)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    model = V3A4Refiner('none').to(dev).eval()
    load_r1_proposal_strict(model, os.path.join(a.src_root, R1_CK), dev)
    for p in model.parameters():
        p.requires_grad_(False)

    split = json.load(open(os.path.join(a.v4_root, 'splits', 'split.json'),
                           encoding='utf-8'))
    all_pairs = {p[0]: p for p in pairs_from_manifest(
        os.path.join(a.src_root, 'manifests', 'refiner_train.csv'))}
    coarse = [int(x) for x in a.coarse.split(',') if x.strip()]

    out, per_image = {}, []
    for tag in [s.strip() for s in a.splits.split(',') if s.strip()]:
        rows = [all_pairs[i] for i in split['train' if tag == 'train' else 'dev']]
        ds = TrainSet(ns, crop_size=0, pairs=rows,
                      y0_cache=os.path.join(a.src_root, a.cache_name,
                                            'refiner_train'), split='Train')
        mmap = json.load(open(os.path.join(
            a.v4_root, 'mappings',
            'mismatch_%s.json' % ('train_575' if tag == 'train' else 'dev_64')),
            encoding='utf-8'))
        acc = {c: [] for c in ('Base', 'R1')}
        for s in STATES:
            for c in ('Global-AO', 'Spatial-AO'):
                acc['%s|%s' % (c, s)] = []
            for g in coarse:
                acc['Coarse%d-AO|%s' % (g, s)] = []
        for i, (name, low, high) in enumerate(rows):
            _n, lr, hr, ref, y0, _m = ds._load(i)
            donor = ds.ref_map[mmap[name]]
            mis = read_model_image(donor, size=hr.shape[-2:])
            refs = {'correct': ref, 'true_dark_g0.5': exposure_gain(ref, 0.5),
                    'mismatch': mis}
            y0d, hrd = y0[None].to(dev), hr[None].to(dev)
            rec = dict(sample_id=name, split=tag,
                       Base=metrics(y0d, hrd)[0])
            with torch.no_grad():
                for s in STATES:
                    rd = refs[s][None].to(dev)
                    sr, aux = model.proposal(y0d, rd)
                    D = aux['gate'] * aux['delta']
                    rec['R1|%s' % s] = metrics(sr, hrd)[0]
                    # global scalar
                    qg, _N, _Z = global_action_optimal_gate(y0d, hrd, D)
                    rec['Global-AO|%s' % s] = metrics(y0d + qg * D, hrd)[0]
                    # spatial H/4
                    c = qopt_components(y0d, hrd, D)
                    qf = F.interpolate(c['q_opt'], size=y0d.shape[-2:],
                                       mode='bilinear', align_corners=False)
                    rec['Spatial-AO|%s' % s] = metrics(y0d + qf * D, hrd)[0]
                    # coarse block gates
                    h, w = y0d.shape[-2:]
                    for g in coarse:
                        qq = F.adaptive_avg_pool2d(c['q_opt'], (g, g))
                        qq = F.interpolate(qq, size=(h, w), mode='bilinear',
                                           align_corners=False)
                        rec['Coarse%d-AO|%s' % (g, s)] = metrics(y0d + qq * D,
                                                                 hrd)[0]
                acc['Base'].append(rec['Base'])
            for s in STATES:
                acc['R1'].append(rec['R1|%s' % s])
                for c in ('Global-AO', 'Spatial-AO'):
                    acc['%s|%s' % (c, s)].append(rec['%s|%s' % (c, s)])
                for g in coarse:
                    acc['Coarse%d-AO|%s' % (g, s)].append(rec['Coarse%d-AO|%s' % (g, s)])
            per_image.append(rec)
            if (i + 1) % 100 == 0:
                print('  %s %d/%d' % (tag, i + 1, len(rows)))

        res = {}
        base = float(np.mean(acc['Base']))
        for s in STATES:
            r1 = float(np.mean(acc['R1|%s' % s]))
            glob = float(np.mean(acc['Global-AO|%s' % s]))
            spat = float(np.mean(acc['Spatial-AO|%s' % s]))
            hg, hs = glob - r1, spat - r1
            entry = dict(base=base, R1=r1, Global_AO=glob, Spatial_AO=spat,
                         H_global=hg, H_spatial=hs,
                         capture_global=(hg / hs) if abs(hs) > 1e-9 else None)
            for g in coarse:
                v = float(np.mean(acc['Coarse%d-AO|%s' % (g, s)]))
                entry['Coarse%d_AO' % g] = v
                entry['capture_coarse%d' % g] = (v - r1) / hs if abs(hs) > 1e-9 else None
            # per-image stats vs R1
            d = np.array(acc['Global-AO|%s' % s]) - np.array(acc['R1|%s' % s])
            ds_ = np.array(acc['Spatial-AO|%s' % s]) - np.array(acc['R1|%s' % s])
            entry['global_vs_r1'] = dict(mean=float(d.mean()), median=float(np.median(d)),
                                         better=int((d > 0).sum()), n=len(d),
                                         worst=float(d.min()),
                                         below_minus1=int((d < -1).sum()))
            entry['spatial_vs_r1'] = dict(mean=float(ds_.mean()),
                                          median=float(np.median(ds_)),
                                          better=int((ds_ > 0).sum()), n=len(ds_),
                                          worst=float(ds_.min()),
                                          below_minus1=int((ds_ < -1).sum()))
            res[s] = entry
        out[tag] = res
        print('%s done' % tag)

    # §15: dev64 correct Spatial-AO must reproduce the V3-A.4 headroom
    ref_h = None
    p = os.path.join(a.v4_root, 'dev_eval', 'summary_dev.json')
    if os.path.isfile(p):
        ref_h = json.load(open(p))['headroom']['correct']
        got = out['dev']['correct']['H_spatial']
        if abs(got - ref_h) > 0.01:
            raise SystemExit('dev correct Spatial-AO headroom %.4f != V3-A.4 %.4f'
                             % (got, ref_h))
        print('reproduced V3-A.4 dev correct Spatial-AO headroom %.4f' % got)

    json.dump(out, open(os.path.join(a.root, 'oracle', 'global_vs_spatial.json'),
                        'w', encoding='utf-8'), indent=2, sort_keys=True)
    with open(os.path.join(a.root, 'oracle', 'per_image.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    print()
    print('══ D2: Gate Resolution Necessity（§19 表 2） ══')
    hdr = '  %-6s %-18s %9s %10s %11s %9s' % ('split', 'state', 'R1', 'Global-AO',
                                              'Spatial-AO', 'capture')
    print(hdr)
    for tag in out:
        for s in STATES:
            e = out[tag][s]
            cap = e['capture_global']
            line = '  %-6s %-18s %9.4f %10.4f %11.4f %9s' % (
                tag, s, e['R1'], e['Global_AO'], e['Spatial_AO'],
                ('%.3f' % cap) if cap is not None else 'n/a')
            print(line)
    print()
    print('  headroom:  H_global vs H_spatial')
    for tag in out:
        for s in STATES:
            e = out[tag][s]
            print('    %-6s %-18s %+9.4f %+11.4f' % (tag, s, e['H_global'],
                                                     e['H_spatial']))


if __name__ == '__main__':
    main()
