#!/usr/bin/env python
"""V4.0 eval: A0 Y0-canvas vs A1 Ref-canvas + reverse-guidance / retention."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK, metrics as _metrics  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from model.V4RefCanvas import features_a0, features_a1, features_a1_mode  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES                                         # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head, nanmean, require_ckpt  # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v3b_runtime import proposal_core, residual_freq_stats              # noqa: E402
from v3b2_runtime import pair_delta_stats_by_state                      # noqa: E402
from v4_runtime import (ARM_A0, ARM_A1, ARMS, arm_meta, canvas_retention,  # noqa: E402
                        require_ckpt_blob_v4, safety_plus, verdict_v4)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v4_ref_canvas'
REF_MODES = ('normal', 'self', 'zero', 'shuffled_target')


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
        raise SystemExit('eval repo HEAD drift: live=%s lock=%s' % (live, lock['repo_commit']))
    live_sha = dict(
        proposal=file_sha256(lock['proposal_ckpt']),
        split=file_sha256(lock['split_json']),
        mismatch_dev=file_sha256(lock['mismatch_dev']),
        cache_meta=file_sha256(lock['cache_metadata_path']),
    )
    want = dict(
        proposal=lock['proposal_sha256'],
        split=lock['split_sha256'],
        mismatch_dev=lock['mismatch_dev_sha256'],
        cache_meta=lock['cache_metadata_sha256'],
    )
    bad = [k for k in want if live_sha[k] != want[k]]
    if bad:
        raise SystemExit('eval live artifact SHA fail: %s' % bad)

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
    modes = REF_MODES if a.full_dep else ('normal',)

    heads = {}
    for arm in ARMS:
        path = os.path.join(a.root, arm, 'checkpoints', 'ckpt_%06d.pt' % a.step)
        if not os.path.isfile(path):
            continue
        require_ckpt(path, formal=True)
        blob = torch.load(path, map_location='cpu')
        direction, canvas = arm_meta(arm)
        require_ckpt_blob_v4(
            blob, arm=arm, step=a.step, init_sha=lock['init_shas'][arm],
            proposal_sha=lock['proposal_sha256'],
            repo_commit=lock['repo_commit'],
            direction=direction, canvas=canvas,
            shared_init_sha=lock['shared_head_init_sha'], formal=True)
        m = V3B0ResidualFusion(in_ch=96).to(a.device).eval()
        m.load_state_dict(blob['model'], strict=True)
        heads[arm] = m
    if a.verdict and (ARM_A0 not in heads or ARM_A1 not in heads):
        raise SystemExit('--verdict needs both arms')
    if not heads:
        raise SystemExit('no checkpoints at step %d' % a.step)

    rows = []
    print('=== V4 eval @%d n=%d arms=%s full_dep=%s ==='
          % (a.step, n, sorted(heads), a.full_dep), flush=True)
    with torch.no_grad():
        for i in range(n):
            t_next = sample_tensors(ds, (i + 1) % n, 'correct', a.device)
            y0_donor = t_next['Y0']
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                rec = dict(name=t['name'], state=state,
                           psnr_base=float(_metrics(t['Y0'], t['H'])[0]),
                           psnr_raw_ref=float(_metrics(t['R'], t['H'])[0]))
                if ARM_A0 in heads:
                    f0, tt = features_a0(wrapper, t['Y0'], t['R'])
                    y = t['Y0'] + heads[ARM_A0](f0, tt, t['Y0'].shape[-2:], E=None)
                    rec['psnr_%s_normal' % ARM_A0] = float(_metrics(y, t['H'])[0])
                if ARM_A1 in heads:
                    for mode in modes:
                        fr, tlow = features_a1_mode(
                            wrapper, t['Y0'], t['R'], mode=mode,
                            y0_donor=y0_donor if mode == 'shuffled_target' else None)
                        y = t['R'] + heads[ARM_A1](fr, tlow, t['Y0'].shape[-2:], E=None)
                        rec['psnr_%s_%s' % (ARM_A1, mode)] = float(_metrics(y, t['H'])[0])
                        if mode == 'normal':
                            dlt = heads[ARM_A1](fr, tlow, t['Y0'].shape[-2:], E=None)
                            rec.update(canvas_retention(y, t['R'], t['Y0']))
                            mag = dlt.detach().float().abs()
                            rec['dR_mean'] = float(mag.mean())
                            rec['dR_p50'] = float(torch.quantile(mag.reshape(-1), 0.50))
                            rec['dR_p90'] = float(torch.quantile(mag.reshape(-1), 0.90))
                            rec['dR_max'] = float(mag.max())
                            rec['dR_low_frac'] = residual_freq_stats(dlt)['low_frac']
                rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d' % (i + 1, n), flush=True)

    def by_state(key):
        out = {}
        for s in STATES:
            out[s] = nanmean([r[key] for r in rows if r['state'] == s and key in r])
        out['mean'] = nanmean([r[key] for r in rows if key in r])
        return out

    table = dict(Base=by_state('psnr_base'), raw_ref=by_state('psnr_raw_ref'))
    if 'frozen_b0_psnr' in lock:
        table['old_B0'] = {k: float(v) for k, v in lock['frozen_b0_psnr'].items()}
    for arm in heads:
        table[arm] = by_state('psnr_%s_normal' % arm)
        for mode in modes:
            if mode != 'normal':
                k = 'psnr_%s_%s' % (arm, mode)
                if any(k in r for r in rows):
                    table['%s_%s' % (arm, mode)] = by_state(k)
    if ARM_A0 in heads and ARM_A1 in heads:
        table['A1_minus_A0'] = {
            s: table[ARM_A1][s] - table[ARM_A0][s]
            for s in list(STATES) + ['mean']
        }
    if ARM_A1 in heads:
        table['A1_minus_raw_ref'] = {
            s: table[ARM_A1][s] - table['raw_ref'][s]
            for s in list(STATES) + ['mean']
        }

    retain = {}
    if ARM_A1 in heads and rows and 'rho_R' in rows[0]:
        for k in ('dY_R', 'dY_Y0', 'dR_Y0', 'rho_R',
                  'dR_mean', 'dR_p50', 'dR_p90', 'dR_max', 'dR_low_frac'):
            retain[k] = by_state(k)

    safety = {}
    for arm in heads:
        deltas = {s: [] for s in STATES}
        names = {s: [] for s in STATES}
        for r in rows:
            key = 'psnr_%s_normal' % arm
            deltas[r['state']].append(r[key] - r['psnr_base'])
            names[r['state']].append('%s|%s' % (r['name'], r['state']))
        safety[arm] = {s: safety_plus(deltas[s], names[s]) for s in STATES}

    out_dir = os.path.join(a.root, 'diagnostics', 'eval_%06d' % a.step)
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), json_ready(table))
    dump_json(os.path.join(out_dir, 'safety.json'), json_ready(safety))
    if retain:
        dump_json(os.path.join(out_dir, 'canvas_stats.json'), json_ready(retain))

    if a.verdict and ARM_A0 in heads and ARM_A1 in heads:
        dep = dict(normal=table[ARM_A1]['mean'], self=float('nan'),
                   zero=float('nan'), shuffled_target=float('nan'))
        if a.full_dep:
            dep['self'] = table['%s_self' % ARM_A1]['mean']
            dep['zero'] = table['%s_zero' % ARM_A1]['mean']
            dep['shuffled_target'] = table['%s_shuffled_target' % ARM_A1]['mean']
        deltas_by = {s: [] for s in STATES}
        for r in rows:
            dlt = r['psnr_%s_normal' % ARM_A1] - r['psnr_%s_normal' % ARM_A0]
            deltas_by[r['state']].append(dlt)
        verdict = verdict_v4(
            table[ARM_A0], table[ARM_A1], table['raw_ref'], table['Base'],
            safety[ARM_A0], safety[ARM_A1], dep,
            pair_stats=pair_delta_stats_by_state(deltas_by))
        verdict['step'] = a.step
        verdict['eval_repo_commit'] = live
        dump_json(os.path.join(out_dir, 'verdict.json'), json_ready(verdict))
        print('VERDICT', verdict['label'],
              'ΔA1-A0=%.3f ΔA1-raw=%.3f' % (verdict['delta_a1_a0'], verdict['delta_a1_raw']),
              'next=', verdict['next_step'], flush=True)
        for s in STATES:
            print('  %s  A0=%.3f A1=%.3f raw=%.3f dA0=%.3f dRaw=%.3f' % (
                s, table[ARM_A0][s], table[ARM_A1][s], table['raw_ref'][s],
                table['A1_minus_A0'][s], table['A1_minus_raw_ref'][s]), flush=True)
        if a.full_dep:
            print('  dep normal/self/zero/shufT = %.3f / %.3f / %.3f / %.3f' % (
                dep['normal'], dep['self'], dep['zero'], dep['shuffled_target']),
                flush=True)
    else:
        for arm in heads:
            print('ARM', arm, 'mean=%.3f' % table[arm]['mean'], flush=True)
        print('raw_ref mean=%.3f' % table['raw_ref']['mean'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
