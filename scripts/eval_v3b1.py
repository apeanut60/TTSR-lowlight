#!/usr/bin/env python
"""V3-B.1 eval vs frozen B0 @20k. Optional T-mode dependence at --full_dep."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK, metrics as _metrics  # noqa: E402
from model.V3BReferenceAdapt import ARMS, V3B1Model                     # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES                                         # noqa: E402
from v3a6_runtime import dump_json, nanmean, require_ckpt               # noqa: E402
from v3a72_runtime import json_ready, safety_from_deltas                # noqa: E402
from v3b_runtime import (adapt_aux_stats, match_features,               # noqa: E402
                         match_features_mode, proposal_core, verdict_v3b1)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3b1_reference_adapt'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--full_dep', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official test forbidden')
    ckpt = os.path.join(a.root, a.arm, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    require_ckpt(ckpt, formal=True)
    blob = torch.load(ckpt, map_location='cpu')
    if int(blob.get('step', -1)) != int(a.step):
        raise SystemExit('ckpt step mismatch')

    model = V3B1Model(a.arm).to(a.device).eval()
    model.load_state_dict(blob['model'], strict=True)
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
    modes = ('normal', 'self', 'zero', 'shuffled') if a.full_dep else ('normal',)

    rows = []
    print('=== %s eval @%d n=%d ===' % (a.arm, a.step, n), flush=True)
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                rec = dict(name=t['name'], state=state,
                           psnr_base=float(_metrics(t['Y0'], t['H'])[0]))
                r_shuf = t['R']
                if state != 'mismatch':
                    _n, _lr, _hr, _ref, _y0, mis = ds._load(i)
                    r_shuf = mis[None].to(a.device)
                for mode in modes:
                    r_use = r_shuf if mode in ('shuffled', 'mismatch') else t['R']
                    f0, tt = match_features_mode(wrapper, t['Y0'], r_use, mode=mode)
                    delta, aux = model(f0, tt, t['Y0'].shape[-2:], return_aux=True)
                    y = t['Y0'] + delta
                    rec['psnr_%s' % mode] = float(_metrics(y, t['H'])[0])
                    if mode == 'normal':
                        rec.update(adapt_aux_stats(aux))
                rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d' % (i + 1, n), flush=True)

    def by_state(key):
        out = {}
        for s in STATES:
            out[s] = nanmean([r[key] for r in rows if r['state'] == s])
        out['mean'] = nanmean([r[key] for r in rows])
        return out

    table = dict(Base=by_state('psnr_base'), B1=by_state('psnr_normal'))
    for mode in modes:
        if mode != 'normal':
            table['B1_%s' % mode] = by_state('psnr_%s' % mode)
    adapt = {}
    for key in ('dt_mean_abs', 'dt_rms', 'dgamma_mean', 'dgamma_std',
                'dbeta_mean', 'dbeta_std'):
        adapt[key] = by_state(key)

    b0_psnr = lock['frozen_b0_psnr']
    psnr_b1 = {s: table['B1'][s] for s in STATES}
    psnr_b0 = {s: float(b0_psnr[s]) for s in STATES}

    deltas = {s: [] for s in STATES}
    names = {s: [] for s in STATES}
    for r in rows:
        deltas[r['state']].append(r['psnr_normal'] - r['psnr_base'])
        names[r['state']].append('%s|%s' % (r['name'], r['state']))
    safety = {s: safety_from_deltas(deltas[s], names[s]) for s in STATES}
    verdict = verdict_v3b1(psnr_b1, psnr_b0, safety, lock['frozen_b0_safety'])
    verdict['step'] = a.step
    verdict['arm'] = a.arm

    out_dir = os.path.join(a.root, 'diagnostics', '%s_eval_%06d' % (a.arm, a.step))
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), json_ready(table))
    dump_json(os.path.join(out_dir, 'adapt_stats.json'), json_ready(adapt))
    dump_json(os.path.join(out_dir, 'safety.json'), json_ready(safety))
    dump_json(os.path.join(out_dir, 'verdict.json'), json_ready(verdict))
    print('VERDICT', verdict['label'], 'ΔB0=%.3f' % verdict['delta_b1_b0'],
          'next=', verdict['next_step'], flush=True)
    for s in STATES:
        print('  %s  B0=%.3f B1=%.3f dt=%.4g' % (
            s, psnr_b0[s], psnr_b1[s], adapt['dt_mean_abs'][s]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
