#!/usr/bin/env python
"""V3-B.0 eval on dev64: PSNR table, T-mode dependence, residual stats."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK, metrics as _metrics  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, block_energy,  # noqa: E402
                          energy_mask, expand_gate, prepare_geometry,
                          target_geometry)
from v3a6_runtime import dump_json, nanmean, require_ckpt               # noqa: E402
from v3a7_runtime import block_utility                                  # noqa: E402
from v3a72_runtime import binary_block_oracle_q, json_ready, q_to_grid, safety_from_deltas  # noqa: E402
from v3b_runtime import (ARM, b0_forward, match_features_mode,          # noqa: E402
                         proposal_core, residual_freq_stats,
                         residual_magnitude, verdict_v3b0)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3b0_implicit_residual'
REF_H, REF_W = 400, 600
MODES = ('normal', 'self', 'zero', 'shuffled')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official test forbidden')
    ckpt = os.path.join(a.root, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    require_ckpt(ckpt, formal=True)
    blob = torch.load(ckpt, map_location='cpu')
    if int(blob.get('step', -1)) != int(a.step):
        raise SystemExit('ckpt step mismatch')
    if blob.get('arm') not in (None, ARM):
        raise SystemExit('ckpt arm mismatch')

    head = V3B0ResidualFusion().to(a.device).eval()
    head.load_state_dict(blob['model'], strict=True)
    wrapper = load_proposal(lock['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    mmap = json.load(open(lock['mismatch_dev']))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.src_root, lock.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap)
    n = len(splits['dev']) if not a.limit else min(a.limit, len(splits['dev']))
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)
    thr = 0.00012087128488929011

    rows = []
    print('=== B0 eval @%d n=%d ===' % (a.step, n), flush=True)
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                D, sr_r1 = correction(wrapper.proposal, t['Y0'], t['R'])
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
                mask = energy_mask(block_energy(D, geom), thr)
                U = block_utility(t['Y0'], t['H'], D, geom)
                q_bin = binary_block_oracle_q(
                    U.reshape(-1).cpu().numpy(), mask.reshape(-1).cpu().numpy())
                y_bin = t['Y0'] + expand_gate(
                    q_to_grid(q_bin, 64, 64, a.device), geom) * D
                rec = dict(
                    name=t['name'], state=state,
                    psnr_base=float(_metrics(t['Y0'], t['H'])[0]),
                    psnr_r1=float(_metrics(sr_r1, t['H'])[0]),
                    psnr_ao=float(_metrics(t['Y0'] + tgt['q_full'] * D, t['H'])[0]),
                    psnr_bin=float(_metrics(y_bin, t['H'])[0]),
                )
                r_shuf = t['R']
                if state != 'mismatch':
                    # donor already in mismatch tensor of dataset; reload mismatch
                    name, lr, hr, ref, y0, mis = ds._load(i)
                    if mis is None:
                        raise SystemExit('no mismatch donor')
                    r_shuf = mis[None].to(a.device)
                for mode in MODES:
                    r_use = r_shuf if mode in ('shuffled', 'mismatch') else t['R']
                    f0, tt = match_features_mode(wrapper, t['Y0'], r_use, mode=mode)
                    y, dlt = b0_forward(head, f0, tt, t['Y0'])
                    rec['psnr_%s' % mode] = float(_metrics(y, t['H'])[0])
                    if mode == 'normal':
                        mag = residual_magnitude(dlt)
                        freq = residual_freq_stats(dlt)
                        rec['dY_mean'] = mag['mean']
                        rec['dY_p90'] = mag['p90']
                        rec['dY_low_frac'] = freq['low_frac']
                rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d' % (i + 1, n), flush=True)

    def by_state(key):
        out = {}
        for s in STATES:
            out[s] = nanmean([r[key] for r in rows if r['state'] == s])
        out['mean'] = nanmean([r[key] for r in rows])
        return out

    table = dict(
        Base=by_state('psnr_base'),
        R1=by_state('psnr_r1'),
        AO64=by_state('psnr_ao'),
        BinaryBlockOracle=by_state('psnr_bin'),
        B0_normal=by_state('psnr_normal'),
        B0_self=by_state('psnr_self'),
        B0_zero=by_state('psnr_zero'),
        B0_shuffled=by_state('psnr_shuffled'),
    )
    frozen = lock.get('frozen_baseline_psnr') or {}
    a6 = frozen.get('v3a6_A1_by_state') or {}
    dep = dict(
        normal=table['B0_normal']['mean'],
        self=table['B0_self']['mean'],
        zero=table['B0_zero']['mean'],
        shuffled=table['B0_shuffled']['mean'],
    )
    psnr_b0 = {s: table['B0_normal'][s] for s in STATES}
    psnr_base = {s: table['Base'][s] for s in STATES}
    psnr_r1 = {s: table['R1'][s] for s in STATES}
    verdict = verdict_v3b0(psnr_b0, psnr_base, psnr_r1, a6, dep)
    verdict['step'] = a.step

    deltas = {s: [] for s in STATES}
    names = {s: [] for s in STATES}
    for r in rows:
        deltas[r['state']].append(r['psnr_normal'] - r['psnr_base'])
        names[r['state']].append('%s|%s' % (r['name'], r['state']))
    safety = {s: safety_from_deltas(deltas[s], names[s]) for s in STATES}

    resid = {}
    for s in STATES:
        resid[s] = dict(
            dY_mean=nanmean([r['dY_mean'] for r in rows if r['state'] == s]),
            dY_p90=nanmean([r['dY_p90'] for r in rows if r['state'] == s]),
            low_frac=nanmean([r['dY_low_frac'] for r in rows if r['state'] == s]),
        )
    resid['overall'] = dict(
        dY_mean=nanmean([r['dY_mean'] for r in rows]),
        dY_p90=nanmean([r['dY_p90'] for r in rows]),
        low_frac=nanmean([r['dY_low_frac'] for r in rows]),
    )

    out_dir = os.path.join(a.root, 'diagnostics', 'eval_%06d' % a.step)
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), json_ready(table))
    dump_json(os.path.join(out_dir, 'dependence.json'), json_ready(dep))
    dump_json(os.path.join(out_dir, 'safety.json'), json_ready(safety))
    dump_json(os.path.join(out_dir, 'residual_stats.json'), json_ready(resid))
    dump_json(os.path.join(out_dir, 'verdict.json'), json_ready(verdict))

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  B0=%.3f  A6=%.3f  Δ=%.3f  ref_ok=%s  n-s=%.3f n-z=%.3f'
          % (verdict['mean_psnr']['B0'], verdict['mean_psnr']['V3A6_A1'],
             verdict['delta_b0_a6'], verdict['ref_ok'],
             verdict['dep_normal_self'], verdict['dep_normal_zero']), flush=True)
    for s in STATES:
        print('  %s  Base=%.3f R1=%.3f B0=%.3f self=%.3f zero=%.3f shuf=%.3f'
              % (s, table['Base'][s], table['R1'][s], table['B0_normal'][s],
                 table['B0_self'][s], table['B0_zero'][s],
                 table['B0_shuffled'][s]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
