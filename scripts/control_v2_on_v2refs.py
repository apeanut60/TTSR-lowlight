#!/usr/bin/env python
"""Locked provenance for the V2-refiner-on-Nano-v2 control (V3-A.1 plan §17).

The +0.3736 dB figure was originally produced ad hoc with no saved artifacts.
This script regenerates it with the full record the plan requires: checkpoint
and base SHAs, the exact reference directory, the manifest, per-image CSV and a
summary JSON. Same base, same 100 test images, same metric as everything else.
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
from local_refine_runtime import metrics, sha256                  # noqa: E402
from model.LocalRefine import Refiner                             # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a_runtime import mismatch_permutation                      # noqa: E402

BASE = '/root/data/experiments/retinex_v11/N0_fixed_s42/model/model_00040.pt'
V21 = '/root/data/experiments/retinex_v21_verified'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out_dir', default='/root/data/experiments/v3a1_lolv2real/control_v2_nano_v2')
    a = ap.parse_args(_CLI)
    os.makedirs(a.out_dir, exist_ok=True)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    pairs = pairs_from_manifest(os.path.join(a.root, 'manifests', 'test.csv'))
    cache = os.path.join(a.root, 'cache_y0', 'test')
    ds = TrainSet(ns, crop_size=0, pairs=pairs, y0_cache=cache, split='Test')
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(pairs))

    ck = {
        'Self': os.path.join(V21, 'V2_Self_s42', 'checkpoint_%05d.pt' % a.step),
        'Nano': os.path.join(V21, 'V2_Nano_s42', 'checkpoint_%05d.pt' % a.step),
    }
    models = {}
    for k, p in ck.items():
        m = Refiner().to(a.device)
        m.load_state_dict(torch.load(p, map_location=a.device)['refiner'])
        m.eval()
        m.match.chunk = 1024
        for q in m.parameters():
            q.requires_grad_(False)
        models[k] = m

    conds = ['Base', 'V2-Self', 'V2-Nano-correct', 'V2-Nano-mismatch']
    acc = {c: [] for c in conds}
    per = []
    with torch.no_grad():
        for i in range(len(pairs)):
            _n, lr, hr, ref, y0, _m = ds._load(i)
            y0d, hrd = y0[None].to(a.device), hr[None].to(a.device)
            from dataset.lolv2real_v3a import read_model_image
            mis = read_model_image(
                ds.ref_map[os.path.basename(pairs[perm[i]][1])], size=hr.shape[-2:])
            row = dict(sample_id=os.path.basename(pairs[i][1]),
                       low_path=pairs[i][1], high_path=pairs[i][2],
                       nano_path=ds.ref_map[os.path.basename(pairs[i][1])],
                       mismatch_path=ds.ref_map[os.path.basename(pairs[perm[i]][1])])
            vals = {}
            vals['Base'] = metrics(y0d, hrd)
            vals['V2-Self'] = metrics(models['Self'](y0d, y0d)[0], hrd)
            vals['V2-Nano-correct'] = metrics(
                models['Nano'](y0d, ref[None].to(a.device))[0], hrd)
            vals['V2-Nano-mismatch'] = metrics(
                models['Nano'](y0d, mis[None].to(a.device))[0], hrd)
            for c in conds:
                acc[c].append(vals[c][0])
                row[c + '_psnr'] = vals[c][0]
                row[c + '_ssim_y'] = vals[c][1]
            per.append(row)

    csv_path = os.path.join(a.out_dir, 'per_image_v2_nano_v2.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)

    m = {c: float(np.mean(v)) for c, v in acc.items()}
    summary = dict(
        description='V2 refiner (trained on data1/LSRW) evaluated on LOL-v2-real Test '
                    'with the Nano-v2 references; same base, same 100 images, same metric',
        base_checkpoint=BASE, base_checkpoint_sha256=sha256(BASE),
        self_checkpoint=ck['Self'], self_checkpoint_sha256=sha256(ck['Self']),
        nano_checkpoint=ck['Nano'], nano_checkpoint_sha256=sha256(ck['Nano']),
        ref_dir=os.path.join(a.data_dir, 'Test', a.variant),
        ref_variant=a.variant,
        test_manifest=os.path.join(a.root, 'manifests', 'test.csv'),
        test_manifest_sha256=sha256(os.path.join(a.root, 'manifests', 'test.csv')),
        y0_cache=cache,
        y0_cache_metadata_sha256=sha256(os.path.join(cache, 'metadata.json')),
        n=len(per), step=a.step,
        psnr=m,
        delta=dict(V2_Nano_minus_Base=m['V2-Nano-correct'] - m['Base'],
                   V2_Self_minus_Base=m['V2-Self'] - m['Base'],
                   correct_minus_mismatch=m['V2-Nano-correct'] - m['V2-Nano-mismatch']),
        per_image_csv=csv_path)
    json.dump(summary, open(os.path.join(a.out_dir, 'summary_v2_nano_v2.json'), 'w',
                            encoding='utf-8'), indent=2, sort_keys=True)
    print('n=%d' % len(per))
    for c in conds:
        print('  %-18s %.4f' % (c, m[c]))
    print('  V2-Nano - Base     = %+.4f' % summary['delta']['V2_Nano_minus_Base'])
    print('  V2-Self - Base     = %+.4f' % summary['delta']['V2_Self_minus_Base'])
    print('  correct - mismatch = %+.4f' % summary['delta']['correct_minus_mismatch'])
    print('-> %s' % csv_path)


if __name__ == '__main__':
    main()
