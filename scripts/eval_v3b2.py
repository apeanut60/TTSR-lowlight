#!/usr/bin/env python
"""V3-B.2 eval: A0/A1 PSNR, safety, T-modes, evidence ablations."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK, metrics as _metrics  # noqa: E402
from model.V3BEvidence import (attach_match_evidence, evidence_first_layer_norms,  # noqa: E402
                               extract_match_maps, normalize_evidence)
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES                                         # noqa: E402
from v3a6_runtime import dump_json, git_head, nanmean, require_ckpt     # noqa: E402
from v3a72_runtime import json_ready, safety_from_deltas                # noqa: E402
from v3b_runtime import match_features_mode, proposal_core              # noqa: E402
from v3b2_runtime import (ARM_A0, ARM_A1, ARMS, evidence_channel_shuffle,  # noqa: E402
                          pair_delta_stats_by_state, require_ckpt_blob,
                          verdict_v3b2)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3b2_evidence_fusion'
MODES = ('normal', 'self', 'zero', 'shuffled')


def _load_head(root, arm, step, lock, device, formal=True):
    ckpt = os.path.join(root, arm, 'checkpoints', 'ckpt_%06d.pt' % step)
    require_ckpt(ckpt, formal=formal)
    blob = torch.load(ckpt, map_location='cpu')
    init_sha = lock['init_shas'][arm]
    require_ckpt_blob(
        blob, arm=arm, step=step, init_sha=init_sha,
        proposal_sha=lock['proposal_sha256'],
        repo_commit=None,  # allow commit drift after train
        formal=formal)
    in_ch = 96 if arm == ARM_A0 else 100
    head = V3B0ResidualFusion(in_ch=in_ch).to(device).eval()
    head.load_state_dict(blob['model'], strict=True)
    return head, blob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--full_dep', action='store_true')
    ap.add_argument('--verdict', action='store_true',
                    help='require both A0 and A1 ckpts; write joint verdict')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official test forbidden')
    stats = json.load(open(lock['evidence_stats_path']))
    per_ch = stats['per_channel']

    wrapper = load_proposal(lock['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK
    match_ev = attach_match_evidence(core).to(a.device)
    match_ev.chunk = EVAL_QUERY_CHUNK

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
    modes = MODES if a.full_dep else ('normal',)

    heads = {}
    for arm in ARMS:
        path = os.path.join(a.root, arm, 'checkpoints', 'ckpt_%06d.pt' % a.step)
        if os.path.isfile(path):
            heads[arm], _ = _load_head(a.root, arm, a.step, lock, a.device)

    if a.verdict and (ARM_A0 not in heads or ARM_A1 not in heads):
        raise SystemExit('--verdict needs both A0 and A1 checkpoints')
    if not heads:
        raise SystemExit('no arm checkpoints at step %d' % a.step)

    rows = []
    print('=== B2 eval @%d n=%d arms=%s ===' % (a.step, n, sorted(heads)),
          flush=True)
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
                    if mode == 'self':
                        r_use = t['Y0']
                    if mode == 'zero':
                        f0, tt, e_raw = extract_match_maps(
                            core.encoder, match_ev, t['Y0'], t['R'])
                        tt = torch.zeros_like(tt)
                    else:
                        f0, tt, e_raw = extract_match_maps(
                            core.encoder, match_ev, t['Y0'], r_use)
                    e_norm = normalize_evidence(e_raw, per_ch)

                    for arm, head in heads.items():
                        e_in = e_norm if arm == ARM_A1 else None
                        if arm == ARM_A1 and mode == 'zero':
                            # keep E from normal R for zero-T diagnostic? plan: T modes
                            # use E from same r_use path; for zero T still use that E
                            pass
                        y = t['Y0'] + head(f0, tt, t['Y0'].shape[-2:], E=e_in)
                        rec['psnr_%s_%s' % (arm, mode)] = float(
                            _metrics(y, t['H'])[0])

                    # evidence ablations on A1 @ normal only
                    if ARM_A1 in heads and mode == 'normal':
                        h1 = heads[ARM_A1]
                        y_z = t['Y0'] + h1(f0, tt, t['Y0'].shape[-2:],
                                           E=torch.zeros_like(e_norm))
                        y_s = t['Y0'] + h1(f0, tt, t['Y0'].shape[-2:],
                                           E=evidence_channel_shuffle(e_norm, 0))
                        rec['psnr_A1_zero_e'] = float(_metrics(y_z, t['H'])[0])
                        rec['psnr_A1_shuffle_e'] = float(_metrics(y_s, t['H'])[0])

                rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d' % (i + 1, n), flush=True)

    def by_state(key):
        out = {}
        for s in STATES:
            out[s] = nanmean([r[key] for r in rows if r['state'] == s and key in r])
        out['mean'] = nanmean([r[key] for r in rows if key in r])
        return out

    table = dict(Base=by_state('psnr_base'))
    if 'frozen_b0_psnr' in lock:
        table['old_B0'] = {k: float(v) for k, v in lock['frozen_b0_psnr'].items()}
    for arm in heads:
        table[arm] = by_state('psnr_%s_normal' % arm)
        for mode in modes:
            if mode != 'normal':
                table['%s_%s' % (arm, mode)] = by_state('psnr_%s_%s' % (arm, mode))
    if ARM_A0 in heads and ARM_A1 in heads:
        table['A1_minus_A0'] = {
            s: table[ARM_A1][s] - table[ARM_A0][s]
            for s in list(STATES) + ['mean']
        }
    if ARM_A1 in heads and 'psnr_A1_zero_e' in rows[0]:
        table['A1_zero_e'] = by_state('psnr_A1_zero_e')
        table['A1_shuffle_e'] = by_state('psnr_A1_shuffle_e')

    safety = {}
    for arm in heads:
        deltas = {s: [] for s in STATES}
        names = {s: [] for s in STATES}
        for r in rows:
            key = 'psnr_%s_normal' % arm
            deltas[r['state']].append(r[key] - r['psnr_base'])
            names[r['state']].append('%s|%s' % (r['name'], r['state']))
        safety[arm] = {s: safety_from_deltas(deltas[s], names[s]) for s in STATES}

    e_norms = {}
    for arm, head in heads.items():
        if arm == ARM_A1:
            e_norms = evidence_first_layer_norms(head)

    out_dir = os.path.join(a.root, 'diagnostics', 'eval_%06d' % a.step)
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), json_ready(table))
    dump_json(os.path.join(out_dir, 'safety.json'), json_ready(safety))
    if e_norms:
        dump_json(os.path.join(out_dir, 'evidence_weight_norms.json'),
                  json_ready(dict(step=a.step, **e_norms)))

    if a.verdict and ARM_A0 in heads and ARM_A1 in heads:
        dep = {}
        if a.full_dep:
            dep = dict(
                normal=table[ARM_A1]['mean'],
                self=table['%s_self' % ARM_A1]['mean'],
                zero=table['%s_zero' % ARM_A1]['mean'],
                shuffled=table['%s_shuffled' % ARM_A1]['mean'],
            )
        else:
            dep = dict(normal=table[ARM_A1]['mean'], self=float('nan'),
                       zero=float('nan'))

        deltas_by = {s: [] for s in STATES}
        for r in rows:
            d = r['psnr_%s_normal' % ARM_A1] - r['psnr_%s_normal' % ARM_A0]
            deltas_by[r['state']].append(d)
        pair = pair_delta_stats_by_state(deltas_by)

        e_use = None
        if 'A1_zero_e' in table:
            e_use = dict(
                zero_e_delta_mean=table['A1_zero_e']['mean'] - table[ARM_A1]['mean'],
                shuffle_e_delta_mean=table['A1_shuffle_e']['mean'] - table[ARM_A1]['mean'],
                first_layer_norms=e_norms,
            )
        verdict = verdict_v3b2(
            table[ARM_A0], table[ARM_A1],
            safety[ARM_A0], safety[ARM_A1],
            dep, pair_stats=pair, evidence_use=e_use)
        verdict['step'] = a.step
        dump_json(os.path.join(out_dir, 'verdict.json'), json_ready(verdict))
        print('VERDICT', verdict['label'], 'ΔA1-A0=%.3f' % verdict['delta_a1_a0'],
              'next=', verdict['next_step'], flush=True)
        for s in STATES:
            print('  %s  A0=%.3f A1=%.3f d=%.3f' % (
                s, table[ARM_A0][s], table[ARM_A1][s],
                table['A1_minus_A0'][s]), flush=True)
    else:
        for arm in heads:
            print('ARM', arm, 'mean=%.3f' % table[arm]['mean'], flush=True)
            for s in STATES:
                print('  %s %.3f' % (s, table[arm][s]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
