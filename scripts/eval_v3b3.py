#!/usr/bin/env python
"""V3-B.3 eval: A0/A1 PSNR, safety, global-prior ablations, 2x2 T×G."""

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
from model.V3BGlobalStats import (V3B3A1, global_first_layer_norms,    # noqa: E402
                                  normalize_global_stats, rgb01_mean_std)
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES                                         # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head, nanmean, require_ckpt  # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v3b_runtime import match_features_mode, proposal_core, residual_freq_stats  # noqa: E402
from v3b2_runtime import pair_delta_stats_by_state, require_ckpt_blob   # noqa: E402
from v3b3_runtime import ARM_A0, ARM_A1, ARMS, safety_plus, verdict_v3b3  # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3b3_global_stats'


def _load_model(root, arm, step, lock, device, formal=True):
    ckpt = os.path.join(root, arm, 'checkpoints', 'ckpt_%06d.pt' % step)
    require_ckpt(ckpt, formal=formal)
    blob = torch.load(ckpt, map_location='cpu')
    require_ckpt_blob(
        blob, arm=arm, step=step, init_sha=lock['init_shas'][arm],
        proposal_sha=lock['proposal_sha256'],
        repo_commit=None, formal=formal)
    if arm == ARM_A0:
        model = V3B0ResidualFusion(in_ch=96).to(device).eval()
    else:
        model = V3B3A1().to(device).eval()
    model.load_state_dict(blob['model'], strict=True)
    return model, blob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--full_dep', action='store_true')
    ap.add_argument('--verdict', action='store_true')
    ap.add_argument('--allow_eval_code_drift', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official test forbidden')
    live = git_head()
    if live != lock['repo_commit'] and not a.allow_eval_code_drift:
        raise SystemExit('eval repo HEAD drift: live=%s lock=%s (pass '
                         '--allow_eval_code_drift)' % (live, lock['repo_commit']))
    live_sha = dict(
        proposal=file_sha256(lock['proposal_ckpt']),
        split=file_sha256(lock['split_json']),
        mismatch_dev=file_sha256(lock['mismatch_dev']),
        global_stats=file_sha256(lock['global_stats_path']),
    )
    want = dict(
        proposal=lock['proposal_sha256'],
        split=lock['split_sha256'],
        mismatch_dev=lock['mismatch_dev_sha256'],
        global_stats=lock['global_stats_sha256'],
    )
    bad = [k for k in want if live_sha[k] != want[k]]
    if bad:
        raise SystemExit('eval live artifact SHA fail: %s' % bad)

    per_ch = json.load(open(lock['global_stats_path']))['per_channel']
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

    heads = {}
    for arm in ARMS:
        path = os.path.join(a.root, arm, 'checkpoints', 'ckpt_%06d.pt' % a.step)
        if os.path.isfile(path):
            heads[arm], _ = _load_model(a.root, arm, a.step, lock, a.device)
    if a.verdict and (ARM_A0 not in heads or ARM_A1 not in heads):
        raise SystemExit('--verdict needs both A0 and A1 checkpoints')
    if not heads:
        raise SystemExit('no arm checkpoints at step %d' % a.step)

    rows = []
    print('=== B3 eval @%d n=%d arms=%s ===' % (a.step, n, sorted(heads)),
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

                f0_n, t_n = match_features_mode(wrapper, t['Y0'], t['R'], 'normal')
                s_n = normalize_global_stats(rgb01_mean_std(t['R']), per_ch)
                s_self = normalize_global_stats(rgb01_mean_std(t['Y0']), per_ch)
                s_z = torch.zeros_like(s_n)
                s_sh = normalize_global_stats(rgb01_mean_std(r_shuf), per_ch)

                if ARM_A0 in heads:
                    d = heads[ARM_A0](f0_n, t_n, t['Y0'].shape[-2:], E=None)
                    rec['psnr_%s_normal' % ARM_A0] = float(
                        _metrics(t['Y0'] + d, t['H'])[0])
                    freq = residual_freq_stats(d)
                    rec['a0_e_low'] = freq['e_low']
                    rec['a0_e_high'] = freq['e_high']
                    rec['a0_low_frac'] = freq['low_frac']

                if ARM_A1 in heads:
                    h1 = heads[ARM_A1]
                    d = h1(f0_n, t_n, t['Y0'].shape[-2:], s_n)
                    rec['psnr_%s_normal' % ARM_A1] = float(
                        _metrics(t['Y0'] + d, t['H'])[0])
                    freq = residual_freq_stats(d)
                    rec['a1_e_low'] = freq['e_low']
                    rec['a1_e_high'] = freq['e_high']
                    rec['a1_low_frac'] = freq['low_frac']
                    rec['g_mean'] = float(h1.mlp(s_n).mean())
                    rec['g_std'] = float(h1.mlp(s_n).std(unbiased=False))

                    if a.full_dep:
                        # global ablations: T stays normal
                        for tag, ss in (('self', s_self), ('zero', s_z),
                                        ('shuffled', s_sh)):
                            dd = h1(f0_n, t_n, t['Y0'].shape[-2:], ss)
                            rec['psnr_A1_g_%s' % tag] = float(
                                _metrics(t['Y0'] + dd, t['H'])[0])
                        # 2x2 T × Global
                        f0_s, t_s = match_features_mode(
                            wrapper, t['Y0'], r_shuf, 'shuffled')
                        combos = (
                            ('Tn_Gn', f0_n, t_n, s_n),
                            ('Tn_Gs', f0_n, t_n, s_sh),
                            ('Ts_Gn', f0_s, t_s, s_n),
                            ('Ts_Gs', f0_s, t_s, s_sh),
                        )
                        for tag, ff, tt, ss in combos:
                            dd = h1(ff, tt, t['Y0'].shape[-2:], ss)
                            rec['psnr_2x2_%s' % tag] = float(
                                _metrics(t['Y0'] + dd, t['H'])[0])

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
    if ARM_A0 in heads and ARM_A1 in heads:
        table['A1_minus_A0'] = {
            s: table[ARM_A1][s] - table[ARM_A0][s]
            for s in list(STATES) + ['mean']
        }
    if a.full_dep and ARM_A1 in heads:
        for tag in ('self', 'zero', 'shuffled'):
            table['A1_g_%s' % tag] = by_state('psnr_A1_g_%s' % tag)
        for tag in ('Tn_Gn', 'Tn_Gs', 'Ts_Gn', 'Ts_Gs'):
            table['A1_2x2_%s' % tag] = by_state('psnr_2x2_%s' % tag)

    freq = {}
    for prefix, lab in (('a0', ARM_A0), ('a1', ARM_A1)):
        if lab not in heads:
            continue
        freq[lab] = dict(
            e_low=by_state('%s_e_low' % prefix),
            e_high=by_state('%s_e_high' % prefix),
            low_frac=by_state('%s_low_frac' % prefix),
        )

    safety = {}
    for arm in heads:
        deltas = {s: [] for s in STATES}
        names = {s: [] for s in STATES}
        for r in rows:
            key = 'psnr_%s_normal' % arm
            deltas[r['state']].append(r[key] - r['psnr_base'])
            names[r['state']].append('%s|%s' % (r['name'], r['state']))
        safety[arm] = {s: safety_plus(deltas[s], names[s]) for s in STATES}

    e_norms = {}
    if ARM_A1 in heads:
        e_norms = global_first_layer_norms(heads[ARM_A1])
        if rows and 'g_mean' in rows[0]:
            e_norms['g_mean'] = nanmean([r['g_mean'] for r in rows])
            e_norms['g_std'] = nanmean([r['g_std'] for r in rows])

    out_dir = os.path.join(a.root, 'diagnostics', 'eval_%06d' % a.step)
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), json_ready(table))
    dump_json(os.path.join(out_dir, 'safety.json'), json_ready(safety))
    dump_json(os.path.join(out_dir, 'residual_freq.json'), json_ready(freq))
    if e_norms:
        dump_json(os.path.join(out_dir, 'global_weight_norms.json'),
                  json_ready(dict(step=a.step, **e_norms)))

    if a.verdict and ARM_A0 in heads and ARM_A1 in heads:
        dep = dict(normal=table[ARM_A1]['mean'],
                   zero=float('nan'), self=float('nan'), shuffled=float('nan'))
        if a.full_dep:
            dep['zero'] = table['A1_g_zero']['mean']
            dep['self'] = table['A1_g_self']['mean']
            dep['shuffled'] = table['A1_g_shuffled']['mean']
        deltas_by = {s: [] for s in STATES}
        for r in rows:
            dlt = r['psnr_%s_normal' % ARM_A1] - r['psnr_%s_normal' % ARM_A0]
            deltas_by[r['state']].append(dlt)
        pair = pair_delta_stats_by_state(deltas_by)
        usage = dict(first_layer_norms=e_norms)
        if a.full_dep:
            usage['zero_g_delta_mean'] = table['A1_g_zero']['mean'] - table[ARM_A1]['mean']
            usage['shuffle_g_delta_mean'] = (
                table['A1_g_shuffled']['mean'] - table[ARM_A1]['mean'])
        verdict = verdict_v3b3(
            table[ARM_A0], table[ARM_A1],
            safety[ARM_A0], safety[ARM_A1],
            dep, pair_stats=pair, usage=usage)
        verdict['step'] = a.step
        verdict['eval_repo_commit'] = live
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
