#!/usr/bin/env python
"""V3-A.5C eval: overfit metrics + C0/C1 verdict on the locked tiny set."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3A5Verifier import V3A5Verifier                            # noqa: E402
from option import parser as option_parser                            # noqa: E402
from v3a5_pipeline import load_rows, make_dataset                     # noqa: E402
from v3a5_runtime import prepare_geometry, target_geometry            # noqa: E402
from v3a5c_runtime import dump_json, verdict_c0, verdict_c1           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3a5c_eval_lib import eval_all_pairs                             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/root/data/experiments/v3a5c_tiny_overfit')
    ap.add_argument('--arm', required=True,
                    choices=['C0_current_loss', 'C1_gate_only'])
    ap.add_argument('--step', type=int, default=-1,
                    help='checkpoint step; -1 = last / max available')
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--micro', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    tiny = os.path.join(a.root, 'tiny')
    if a.micro:
        cache = torch.load(os.path.join(tiny, 'micro4_cache.pt'),
                           map_location='cpu')
        ids = json.load(open(os.path.join(tiny, 'micro4_ids.json')))['ids']
        mmap = json.load(open(os.path.join(tiny, 'micro4_mismatch_map.json')))
    else:
        cache = torch.load(os.path.join(tiny, 'cache.pt'), map_location='cpu')
        ids = json.load(open(os.path.join(tiny, 'tiny16_ids.json')))['ids']
        mmap = json.load(open(os.path.join(tiny, 'tiny16_mismatch_map.json')))

    arm_dir = os.path.join(a.root, a.arm)
    ckpt_dir = os.path.join(arm_dir, 'checkpoints')
    if a.step < 0:
        cands = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt_')]
        steps = sorted(int(f[5:11]) for f in cands)
        step = steps[-1]
    else:
        step = a.step
    ckpt_path = os.path.join(ckpt_dir, 'ckpt_%06d.pt' % step)
    blob = torch.load(ckpt_path, map_location=a.device)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    all_rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                         os.path.join(a.v4_root, 'splits', 'split.json'))
    rows_by_name = {r[0]: r for r in all_rows['train']}
    rows = [rows_by_name[i] for i in ids]
    ds = make_dataset(ns, rows, os.path.join(a.src_root, a.cache_name,
                                             'refiner_train'), mmap)
    name_to_i = {name: i for i, name in enumerate(ids)}
    geom = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)

    model = V3A5Verifier('g64').to(a.device)
    model.load_state_dict(blob['model'], strict=True)
    out_w = float(blob.get('out_weight', 0.1 if a.arm.startswith('C0') else 0.0))
    summary = eval_all_pairs(model, ds, name_to_i, cache['pairs'],
                             cache['entries'], geom, thr, a.device,
                             out_weight=out_w)
    summary['step'] = step
    summary['arm'] = a.arm
    summary['ckpt'] = ckpt_path

    if a.arm.startswith('C0'):
        summary['verdict'] = verdict_c0(summary['overall'], summary['by_state'])
    else:
        c0_path = os.path.join(a.root, 'C0_current_loss', 'final_eval.json')
        if not os.path.isfile(c0_path):
            raise SystemExit('need C0 final_eval.json to judge C1')
        c0 = json.load(open(c0_path))
        summary['verdict'] = verdict_c1(c0['overall'], summary['overall'])

    out = os.path.join(arm_dir, 'final_eval.json')
    dump_json(out, summary)
    v = summary['verdict']
    print('arm=%s step=%d label=%s action=%s' % (
        a.arm, step, v['label'], v['action']))
    print('  masked_MAE=%.4f masked_corr=%.4f dec_acc=%.3f std_ratio=%.3f'
          % (summary['overall']['masked_MAE'],
             summary['overall']['masked_corr'],
             summary['overall']['decision_accuracy'],
             summary['overall'].get('std_ratio', float('nan'))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
