#!/usr/bin/env python
"""V3-A.5D2.1 eval: train64 fit + dev64 transfer; Case A–D classification."""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics                                # noqa: E402
from model.V3A5D2Verifier import V3A5D2Verifier                          # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, block_energy,  # noqa: E402
                          energy_mask, expand_gate, prepare_geometry,
                          target_geometry)
from v3a5c_runtime import (aggregate_pair_metrics, dump_json,           # noqa: E402
                           pair_gate_bundle)
from v3a5d21_runtime import EXPOSURE_STEPS, classify_d21                # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT_EXP = '/root/data/experiments/v3a5d21_scale64'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def nanmean(xs):
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float('nan')


@torch.no_grad()
def eval_ids(model, ds, ids, name_to_i, geom, thr, proposal, device, states=STATES):
    rows = []
    by = {s: [] for s in states}
    for name in ids:
        for state in states:
            t = sample_tensors(ds, name_to_i[name], state, device)
            D, _ = correction(proposal.proposal, t['Y0'], t['R'])
            tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
            mask = energy_mask(block_energy(D, geom), thr)
            q_v = model(t['X'], t['Y0'], t['R'], geom=geom)
            bundle = pair_gate_bundle(q_v, tgt['q_grid'], mask)
            y_hat = t['Y0'] + expand_gate(q_v, geom) * D
            bundle['PSNR'] = float(metrics(y_hat, t['H'])[0])
            bundle['name'] = name
            bundle['state'] = state
            rows.append(bundle)
            by[state].append(bundle)
    overall = aggregate_pair_metrics(rows)
    overall['PSNR'] = nanmean([r['PSNR'] for r in rows])
    return dict(overall=overall,
                by_state={s: aggregate_pair_metrics(by[s]) for s in states})


def load_model(root, arm, step, device):
    ck = os.path.join(root, arm, 'checkpoints', 'ckpt_%06d.pt' % step)
    if not os.path.isfile(ck):
        ck = os.path.join(root, arm, 'checkpoints', 'last.pt')
    blob = torch.load(ck, map_location=device)
    m = V3A5D2Verifier(arm).to(device)
    m.load_state_dict(blob['model'], strict=True)
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT_EXP)
    ap.add_argument('--arm', default='A1_multiscale')
    ap.add_argument('--steps', default=','.join(str(s) for s in EXPOSURE_STEPS))
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    ids64 = json.load(open(os.path.join(a.root, 'subset', 'train64_ids.json')))['ids']
    mmap64 = json.load(open(os.path.join(
        a.root, 'subset', 'train64_mismatch_map.json')))
    steps = [int(x) for x in a.steps.split(',') if x.strip()]

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       os.path.join(a.v4_root, 'splits', 'split.json'))
    rows_by_name = {r[0]: r for r in splits['train']}
    rows64 = [rows_by_name[i] for i in ids64]
    ds64 = make_dataset(ns, rows64,
                        os.path.join(a.src_root, a.cache_name, 'refiner_train'),
                        mmap64)
    name_to_i64 = {n: i for i, n in enumerate(ids64)}

    mmap_dev = json.load(open(os.path.join(
        a.v4_root, 'mappings', 'mismatch_dev_64.json')))
    ds_dev = make_dataset(ns, splits['dev'],
                          os.path.join(a.src_root, a.cache_name, 'refiner_train'),
                          mmap_dev)
    dev_ids = [r[0] for r in splits['dev']]
    name_to_i_dev = {n: i for i, n in enumerate(dev_ids)}

    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    geom = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)

    curve = []
    for step in steps:
        print('=== eval %s @%d ===' % (a.arm, step), flush=True)
        model, ck = load_model(a.root, a.arm, step, a.device)
        print('  ckpt %s' % ck, flush=True)
        print('  train64...', flush=True)
        tr = eval_ids(model, ds64, ids64, name_to_i64, geom, thr, proposal,
                      a.device)
        print('  dev64...', flush=True)
        dv = eval_ids(model, ds_dev, dev_ids, name_to_i_dev, geom, thr,
                      proposal, a.device)
        o_tr, o_dv = tr['overall'], dv['overall']
        print('  train64 MAE=%.4f corr=%.4f acc=%.3f'
              % (o_tr['masked_MAE'], o_tr['masked_corr'], o_tr['decision_accuracy']))
        print('  dev64   MAE=%.4f corr=%.4f acc=%.3f'
              % (o_dv['masked_MAE'], o_dv['masked_corr'], o_dv['decision_accuracy']))
        curve.append(dict(step=step, train64=o_tr, dev64=o_dv,
                          by_state_train=tr['by_state'],
                          by_state_dev=dv['by_state']))

    final = curve[-1]
    verdict = classify_d21(final['train64'], final['dev64'])
    verdict['arm'] = a.arm
    verdict['step'] = final['step']
    verdict['curve'] = [
        dict(step=c['step'],
             train64_corr=c['train64']['masked_corr'],
             train64_mae=c['train64']['masked_MAE'],
             train64_acc=c['train64']['decision_accuracy'],
             dev64_corr=c['dev64']['masked_corr'],
             dev64_mae=c['dev64']['masked_MAE'],
             dev64_acc=c['dev64']['decision_accuracy'])
        for c in curve
    ]
    dump_json(os.path.join(a.root, 'diagnostics', 'd21_verdict.json'), verdict)
    dump_json(os.path.join(a.root, 'diagnostics', 'eval_curve.json'), curve)
    print('VERDICT case=%s %s next=%s' % (
        verdict['case'], verdict['label'], verdict['next_step']), flush=True)
    print(json.dumps(verdict['train64'], indent=2))
    print(json.dumps(verdict['dev64'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
