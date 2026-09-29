#!/usr/bin/env python
"""V3-A.5D1 eval + verdict: compare A0 vs A1 at a checkpoint step."""

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3A5DVerifier import ARMS, V3A5DVerifier                      # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_rows, make_dataset                       # noqa: E402
from v3a5_runtime import prepare_geometry, target_geometry              # noqa: E402
from v3a5c_runtime import dump_json                                     # noqa: E402
from v3a5d1_eval_lib import eval_all_pairs_d1                           # noqa: E402
from v3a5d1_runtime import OUT_WEIGHT, verdict_d1                       # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT_EXP = '/root/data/experiments/v3a5d1_tiny_evidence'


def eval_arm(root, arm, step, device, src_root, v4_root, data_dir, variant,
             cache_name):
    lock = json.load(open(os.path.join(root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    norm = lock['evidence_norm']
    tiny = os.path.join(root, 'tiny')
    cache = torch.load(os.path.join(tiny, 'cache.pt'), map_location='cpu')
    ids = json.load(open(os.path.join(tiny, 'tiny16_ids.json')))['ids']
    mmap = json.load(open(os.path.join(tiny, 'tiny16_mismatch_map.json')))
    pairs, entries = cache['pairs'], cache['entries']

    ns = option_parser.parse_args([])
    ns.dataset_dir = data_dir
    ns.v3a_ref_variant = variant
    all_rows = load_rows(os.path.join(src_root, 'manifests', 'refiner_train.csv'),
                         os.path.join(v4_root, 'splits', 'split.json'))
    rows_by_name = {r[0]: r for r in all_rows['train']}
    rows = [rows_by_name[i] for i in ids]
    ds = make_dataset(ns, rows, os.path.join(src_root, cache_name, 'refiner_train'),
                      mmap)
    name_to_i = {name: i for i, name in enumerate(ids)}
    geom = prepare_geometry(target_geometry(400, 600, 'g64'), device)

    ckpt_path = os.path.join(root, arm, 'checkpoints', 'ckpt_%06d.pt' % step)
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(root, arm, 'checkpoints', 'last.pt')
    blob = torch.load(ckpt_path, map_location=device)
    model = V3A5DVerifier(arm).to(device)
    model.load_state_dict(blob['model'], strict=True)
    model.set_evidence_norm(norm['mean'], norm['std'], norm['zscore_mask'])
    summary = eval_all_pairs_d1(
        model, ds, name_to_i, pairs, entries, geom, thr, device,
        out_weight=OUT_WEIGHT)
    summary['step'] = int(blob.get('step', step))
    summary['arm'] = arm
    summary['ckpt'] = ckpt_path
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT_EXP)
    ap.add_argument('--step', type=int, default=20000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    a = ap.parse_args(_CLI)

    results = {}
    for arm in ARMS:
        print('eval %s @%d' % (arm, a.step), flush=True)
        results[arm] = eval_arm(
            a.root, arm, a.step, a.device, a.src_root, a.v4_root,
            a.data_dir, a.variant, a.cache_name)
        o = results[arm]['overall']
        print('  MAE=%.4f corr=%.4f acc=%.3f' % (
            o['masked_MAE'], o['masked_corr'], o['decision_accuracy']))

    v = verdict_d1(results['A0_control']['overall'],
                   results['A1_evidence']['overall'])
    v['step'] = a.step
    dump_json(os.path.join(a.root, 'diagnostics', 'd1_verdict.json'), v)
    for arm in ARMS:
        dump_json(os.path.join(a.root, arm, 'eval_step_%d.json' % a.step),
                  dict(step=a.step, overall=results[arm]['overall'],
                       by_state=results[arm]['by_state']))
    print('VERDICT', v['label'], 'next=', v['next_step'], flush=True)
    print(json.dumps({k: v[k] for k in (
        'delta_masked_corr', 'delta_masked_MAE', 'A0', 'A1',
        'success', 'strong', 'close_evidence_route')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
