#!/usr/bin/env python
"""V3-A.5D2-full eval: train575+dev64 × 3 states for A0 vs A1; D2-full verdict."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics                                # noqa: E402
from model.V3A5D2Verifier import ARMS, V3A5D2Verifier                    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, block_energy,  # noqa: E402
                          energy_mask, expand_gate,
                          prepare_geometry, recovery, target_geometry)
from v3a5c_runtime import (aggregate_pair_metrics, dump_json,           # noqa: E402
                           pair_gate_bundle)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT = '/root/data/experiments/v3a5d2_rf_full'
V5A = '/root/data/experiments/v3a5_g64_verifier'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def nanmean(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float('nan')


def verdict_d2_full(dev_by_arm_state, base_psnr_by_state, r1_psnr_by_state,
                    v5a_dev_mcorr_ref=0.20):
    """Plan §37 bars adapted to D2-full (A0 vs A1 MultiScale).

    Pass if A1 on dev:
      - correct: PSNR >= R1 - 0.02
      - dark/mismatch: PSNR >= Base
      - overall mCorr >= 0.40 (and clearly > V5A ~0.08–0.20)
      - Recovery64 > 0 on >= 2/3 states
      - A1 mCorr > A0 mCorr (RF actually helps vs control)
    """
    a1 = dev_by_arm_state['A1_multiscale']
    a0 = dev_by_arm_state['A0_control']

    psnr_ok = {}
    psnr_ok['correct'] = bool(
        a1['correct']['PSNR'] >= r1_psnr_by_state['correct'] - 0.02)
    for st in ('true_dark_g0.5', 'mismatch'):
        psnr_ok[st] = bool(a1[st]['PSNR'] >= base_psnr_by_state[st] - 1e-6)

    mcorrs_a1 = [a1[s]['masked_corr'] for s in STATES]
    mcorrs_a0 = [a0[s]['masked_corr'] for s in STATES]
    mcorr_a1 = nanmean(mcorrs_a1)
    mcorr_a0 = nanmean(mcorrs_a0)
    mcorr_ok = bool(mcorr_a1 >= 0.40 and mcorr_a1 > v5a_dev_mcorr_ref)
    beats_a0 = bool(mcorr_a1 > mcorr_a0 + 0.05)

    rec_hits = sum(1 for s in STATES
                   if a1[s].get('Recovery64') is not None
                   and np.isfinite(a1[s]['Recovery64'])
                   and a1[s]['Recovery64'] > 0)
    recovery_ok = rec_hits >= 2

    passed = bool(all(psnr_ok.values()) and mcorr_ok and recovery_ok and beats_a0)
    strong = bool(passed and mcorr_a1 >= 0.60 and rec_hits == 3)

    if strong:
        label, next_step = 'D2_full_strong_success', 'STOP_RF_proven'
    elif passed:
        label, next_step = 'D2_full_success', 'STOP_RF_proven'
    elif mcorr_a1 >= 0.30 and beats_a0:
        label, next_step = 'D2_full_partial', 'REVIEW'
    else:
        label, next_step = 'D2_full_fail', 'STOP_tiny_only_overfit'

    return dict(
        label=label, passed=passed, strong=strong, next_step=next_step,
        psnr_ok=psnr_ok, mcorr_A1=mcorr_a1, mcorr_A0=mcorr_a0,
        mcorr_ok=mcorr_ok, beats_A0=beats_a0,
        recovery_hits=rec_hits, recovery_ok=recovery_ok,
        per_state_A1={s: dict(PSNR=a1[s]['PSNR'], masked_corr=a1[s]['masked_corr'],
                              Recovery64=a1[s].get('Recovery64'),
                              masked_MAE=a1[s].get('masked_MAE'),
                              decision_accuracy=a1[s].get('decision_accuracy'))
                      for s in STATES},
        per_state_A0={s: dict(PSNR=a0[s]['PSNR'], masked_corr=a0[s]['masked_corr'],
                              Recovery64=a0[s].get('Recovery64'))
                      for s in STATES},
    )


@torch.no_grad()
def eval_split(model, ds, n, geom, thr, proposal, device, limit=0):
    n = n if not limit else min(limit, n)
    by_state = {s: [] for s in STATES}
    base_psnr = {s: [] for s in STATES}
    r1_psnr = {s: [] for s in STATES}
    for i in range(n):
        for state in STATES:
            t = sample_tensors(ds, i, state, device)
            D, _ = correction(proposal.proposal, t['Y0'], t['R'])
            tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
            mask = energy_mask(block_energy(D, geom), thr)
            q_v = model(t['X'], t['Y0'], t['R'], geom=geom)
            q_full = expand_gate(q_v, geom)
            bundle = pair_gate_bundle(q_v, tgt['q_grid'], mask)
            y_hat = t['Y0'] + q_full * D
            y_r1 = t['Y0'] + D
            y_ao = t['Y0'] + tgt['q_full'] * D
            p_base = metrics(t['Y0'], t['H'])[0]
            p_r1 = metrics(y_r1, t['H'])[0]
            p_hat = metrics(y_hat, t['H'])[0]
            p_ao = metrics(y_ao, t['H'])[0]
            rec = recovery(p_hat, p_r1, p_ao)
            bundle.update(dict(
                PSNR=float(p_hat), PSNR_Base=float(p_base), PSNR_R1=float(p_r1),
                PSNR_AO64=float(p_ao),
                Recovery64=float(rec) if rec is not None else float('nan'),
                name=t['name'], state=state,
            ))
            by_state[state].append(bundle)
            base_psnr[state].append(p_base)
            r1_psnr[state].append(p_r1)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print('    %d/%d' % (i + 1, n), flush=True)
    out = {}
    for s in STATES:
        agg = aggregate_pair_metrics(by_state[s])
        # PSNR / Recovery mean over images
        agg['PSNR'] = nanmean([r['PSNR'] for r in by_state[s]])
        agg['PSNR_Base'] = nanmean([r['PSNR_Base'] for r in by_state[s]])
        agg['PSNR_R1'] = nanmean([r['PSNR_R1'] for r in by_state[s]])
        agg['PSNR_AO64'] = nanmean([r['PSNR_AO64'] for r in by_state[s]])
        agg['Recovery64'] = nanmean([r['Recovery64'] for r in by_state[s]])
        out[s] = agg
    return out, {s: nanmean(base_psnr[s]) for s in STATES}, \
        {s: nanmean(r1_psnr[s]) for s in STATES}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--step', type=int, default=3000)
    ap.add_argument('--splits', default='dev,train')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    split_tags = [s.strip() for s in a.splits.split(',') if s.strip()]

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                     os.path.join(a.v4_root, 'splits', 'split.json'))
    mmaps = {
        'train': json.load(open(os.path.join(a.v4_root, 'mappings',
                                             'mismatch_train_575.json'))),
        'dev': json.load(open(os.path.join(a.v4_root, 'mappings',
                                           'mismatch_dev_64.json'))),
    }
    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    geom = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)

    models = {}
    for arm in ARMS:
        ck = os.path.join(a.root, arm, 'checkpoints', 'ckpt_%06d.pt' % a.step)
        if not os.path.isfile(ck):
            ck = os.path.join(a.root, arm, 'checkpoints', 'last.pt')
        blob = torch.load(ck, map_location=a.device)
        m = V3A5D2Verifier(arm).to(a.device)
        m.load_state_dict(blob['model'], strict=True)
        m.eval()
        models[arm] = m
        print('loaded %s from %s' % (arm, ck), flush=True)

    results = {}
    base_by_split = {}
    r1_by_split = {}
    for tag in split_tags:
        print('=== eval %s ===' % tag, flush=True)
        ds = make_dataset(ns, rows[tag],
                          os.path.join(a.src_root, a.cache_name, 'refiner_train'),
                          mmaps[tag])
        n = len(rows[tag])
        results[tag] = {}
        for arm in ARMS:
            print('  arm %s' % arm, flush=True)
            by, base_psnr, r1_psnr = eval_split(
                models[arm], ds, n, geom, thr, proposal, a.device, a.limit)
            results[tag][arm] = by
            if arm == 'A1_multiscale':
                base_by_split[tag] = base_psnr
                r1_by_split[tag] = r1_psnr
            for s in STATES:
                o = by[s]
                print('    %s  PSNR=%.3f mCorr=%.3f mMAE=%.3f Rec64=%s acc=%.3f'
                      % (s, o['PSNR'], o['masked_corr'], o['masked_MAE'],
                         ('%.3f' % o['Recovery64']
                          if np.isfinite(o['Recovery64']) else 'nan'),
                         o['decision_accuracy']), flush=True)

    # V5A A1_g64 mean dev mCorr reference (~0.14)
    v5a_mcorr = 0.14

    if 'dev' not in results:
        raise SystemExit('D2-full verdict requires --splits to include dev')
    verdict = verdict_d2_full(
        results['dev'], base_by_split['dev'], r1_by_split['dev'],
        v5a_dev_mcorr_ref=v5a_mcorr)
    verdict['step'] = a.step
    verdict['v5a_dev_mcorr_ref'] = v5a_mcorr
    dump_json(os.path.join(a.root, 'diagnostics', 'd2_full_verdict.json'), verdict)
    dump_json(os.path.join(a.root, 'diagnostics', 'eval_full.json'), results)
    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print(json.dumps({k: verdict[k] for k in (
        'passed', 'strong', 'mcorr_A1', 'mcorr_A0', 'psnr_ok',
        'recovery_hits', 'beats_A0')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
