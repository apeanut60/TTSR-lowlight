#!/usr/bin/env python
"""V3-A.5D2 eval + verdict: compare A0 vs A1_multiscale at a checkpoint."""

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

from model.V3A5D2Verifier import ARMS, V3A5D2Verifier                    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors       # noqa: E402
from v3a5_runtime import (STATES, expand_gate, prepare_geometry,        # noqa: E402
                          target_geometry)
from v3a5c_runtime import (aggregate_pair_metrics, dump_json,           # noqa: E402
                           pair_gate_bundle)
from local_refine_runtime import metrics as _metrics                    # noqa: E402
from v3a5d2_runtime import OUT_WEIGHT, verdict_d2                       # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT_EXP = '/root/data/experiments/v3a5d2_rf_tiny'


def eval_arm(root, arm, step, device, src_root, v4_root, data_dir, variant,
             cache_name):
    lock = json.load(open(os.path.join(root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
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
    model = V3A5D2Verifier(arm).to(device)
    model.load_state_dict(blob['model'], strict=True)

    rows_out = []
    by = {s: [] for s in STATES}
    model.eval()
    with torch.no_grad():
        for p in pairs:
            ent = entries[p['key']]
            t = sample_tensors(ds, name_to_i[p['name']], p['state'], device)
            D = ent['D'].to(device).float()
            q_star = ent['q_grid'].to(device).float()
            mask = ent['mask'].to(device)
            q_v = model(t['X'], t['Y0'], t['R'], geom=geom)
            q_full = expand_gate(q_v, geom)
            bundle = pair_gate_bundle(q_v, q_star, mask)
            y_hat = t['Y0'] + q_full * D
            psnr = _metrics(y_hat, t['H'])[0]
            bundle.update(dict(PSNR=float(psnr), name=p['name'], state=p['state'],
                               Recovery64=float('nan'),
                               energy_threshold=float(thr),
                               out_weight=float(OUT_WEIGHT)))
            rows_out.append(bundle)
            by[p['state']].append(bundle)

    summary = dict(
        step=int(blob.get('step', step)),
        arm=arm,
        ckpt=ckpt_path,
        overall=aggregate_pair_metrics(rows_out),
        by_state={s: aggregate_pair_metrics(by[s]) for s in STATES},
    )
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

    v = verdict_d2(results['A0_control']['overall'],
                   results['A1_multiscale']['overall'])
    v['step'] = a.step
    dump_json(os.path.join(a.root, 'diagnostics', 'd2_verdict.json'), v)
    for arm in ARMS:
        dump_json(os.path.join(a.root, arm, 'eval_step_%d.json' % a.step),
                  dict(step=a.step, overall=results[arm]['overall'],
                       by_state=results[arm]['by_state']))
    print('VERDICT', v['label'], 'next=', v['next_step'], flush=True)
    print(json.dumps({k: v[k] for k in (
        'delta_masked_corr', 'delta_masked_MAE', 'A0', 'A1',
        'success', 'strong', 'close_rf_route')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
