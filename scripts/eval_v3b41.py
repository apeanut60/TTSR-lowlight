#!/usr/bin/env python
"""V3-B.4.1 eval: A0 RGB vs A1 base-conditioned H/2 (+cond / ref ablations)."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import (EVAL_QUERY_CHUNK, load_frozen_n0,     # noqa: E402
                                  metrics as _metrics)
from model.V3BBaseConditionedFeatureResidual import (  # noqa: E402
    BaseConditionedFeatureResidual, delta_fn_conditioned)
from model.V3BFeatureBridge import (INJECTION_POINT, prefix_to_h2,      # noqa: E402
                                    tiled_bridge_decode)
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES                                         # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head, nanmean, require_ckpt  # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v3b_runtime import match_features_mode, proposal_core              # noqa: E402
from v3b2_runtime import pair_delta_stats_by_state                      # noqa: E402
from v3b41_runtime import (ARM_A0, ARM_A1, ARMS, require_ckpt_blob_b41,  # noqa: E402
                           safety_plus, verdict_v3b41)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3b41_base_conditioned_h2'
REF_MODES = ('normal', 'self', 'zero', 'shuffled')
COND_MODES = ('normal', 'zero', 'shuffled')


def _feat_stats(delta, f_h2, y, y0):
    a = delta.detach().float().abs()
    f = f_h2.detach().float().abs()
    dy = (y - y0).detach().float().abs()
    md = float(a.mean())
    # cos(ΔF, F) over flattened channels/spatial
    dflat = delta.detach().float().reshape(-1)
    fflat = f_h2.detach().float().reshape(-1)
    cos = float(torch.nn.functional.cosine_similarity(
        dflat.unsqueeze(0), fflat.unsqueeze(0)).item())
    return dict(
        dF_mean=md,
        dF_p50=float(torch.quantile(a.reshape(-1), 0.50)),
        dF_p90=float(torch.quantile(a.reshape(-1), 0.90)),
        dF_max=float(a.max()),
        F_mean=float(f.mean()),
        r_F=float(md / (float(f.mean()) + 1e-8)),
        dY_mean=float(dy.mean()),
        r_decode=float(float(dy.mean()) / (md + 1e-8)),
        cos_dF_F=cos,
    )


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
    if file_sha256(lock['bridge_file']) != lock['bridge_file_sha']:
        raise SystemExit('eval bridge_file_sha drift')
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
    ref_modes = REF_MODES if a.full_dep else ('normal',)
    do_cond = bool(a.full_dep)

    heads = {}
    for arm in ARMS:
        path = os.path.join(a.root, arm, 'checkpoints', 'ckpt_%06d.pt' % a.step)
        if not os.path.isfile(path):
            continue
        require_ckpt(path, formal=True)
        blob = torch.load(path, map_location='cpu')
        require_ckpt_blob_b41(
            blob, arm=arm, step=a.step, init_sha=lock['init_shas'][arm],
            proposal_sha=lock['proposal_sha256'],
            repo_commit=lock['repo_commit'],
            injection_point=INJECTION_POINT, formal=True)
        if arm == ARM_A0:
            m = V3B0ResidualFusion(in_ch=96).to(a.device).eval()
        else:
            m = BaseConditionedFeatureResidual().to(a.device).eval()
        m.load_state_dict(blob['model'], strict=True)
        heads[arm] = m
    if a.verdict and (ARM_A0 not in heads or ARM_A1 not in heads):
        raise SystemExit('--verdict needs both arms')
    if not heads:
        raise SystemExit('no checkpoints at step %d' % a.step)

    mainnet = None
    if ARM_A1 in heads:
        n0, _tr, _cfg = load_frozen_n0(
            lock['base_ckpt'], lock['base_run_dir'], a.device)
        mainnet = n0.MainNet

    # Precompute donor F_dec for shuffled-cond (cyclic next image, same size)
    donor_f = [None] * n
    if do_cond and ARM_A1 in heads:
        with torch.no_grad():
            for i in range(n):
                j = (i + 1) % n
                t_d = sample_tensors(ds, j, 'correct', a.device)
                fh, _ = prefix_to_h2(mainnet, t_d['X'])
                # crop pad away for storage at original H/2
                h0, w0 = int(t_d['X'].shape[-2]), int(t_d['X'].shape[-1])
                donor_f[i] = fh[..., : h0 // 2, : w0 // 2].detach().cpu()

    rows = []
    print('=== B4.1 eval @%d n=%d arms=%s full_dep=%s ==='
          % (a.step, n, sorted(heads), a.full_dep), flush=True)
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
                for mode in ref_modes:
                    r_use = r_shuf if mode in ('shuffled',) else t['R']
                    if mode == 'self':
                        r_use = t['Y0']
                    f0, tt = match_features_mode(wrapper, t['Y0'], r_use, mode=mode)
                    if ARM_A0 in heads:
                        y = t['Y0'] + heads[ARM_A0](f0, tt, t['Y0'].shape[-2:], E=None)
                        rec['psnr_%s_%s' % (ARM_A0, mode)] = float(
                            _metrics(y, t['H'])[0])
                    if ARM_A1 in heads:
                        dfn = delta_fn_conditioned(
                            heads[ARM_A1], f0, tt, cond_mode='normal')
                        y = tiled_bridge_decode(mainnet, t['X'], delta_fn=dfn)
                        rec['psnr_%s_%s' % (ARM_A1, mode)] = float(
                            _metrics(y, t['H'])[0])
                        if mode == 'normal':
                            fh, _ctx = prefix_to_h2(mainnet, t['X'])
                            h0, w0 = int(t['X'].shape[-2]), int(t['X'].shape[-1])
                            fh_c = fh[..., : h0 // 2, : w0 // 2]
                            if fh_c.shape[-2:] == f0.shape[-2:]:
                                dlt = heads[ARM_A1](fh_c, f0, tt)
                                rec.update(_feat_stats(dlt, fh_c, y, t['Y0']))
                if do_cond and ARM_A1 in heads:
                    f0n, ttn = match_features_mode(
                        wrapper, t['Y0'], t['R'], mode='normal')
                    for cm in COND_MODES:
                        f_cond = None
                        if cm == 'shuffled':
                            f_cond = donor_f[i].to(a.device)
                            if f_cond.shape[-2:] != f0n.shape[-2:]:
                                raise SystemExit(
                                    'donor F spatial %s != F0 %s on %s'
                                    % (tuple(f_cond.shape[-2:]),
                                       tuple(f0n.shape[-2:]), t['name']))
                        dfn = delta_fn_conditioned(
                            heads[ARM_A1], f0n, ttn, cond_mode=cm,
                            F_cond_full=f_cond)
                        y = tiled_bridge_decode(mainnet, t['X'], delta_fn=dfn)
                        rec['psnr_%s_cond_%s' % (ARM_A1, cm)] = float(
                            _metrics(y, t['H'])[0])
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
    if 'frozen_blind_h2_psnr' in lock:
        table['old_blind_H2'] = {
            k: float(v) for k, v in lock['frozen_blind_h2_psnr'].items()}
    for arm in heads:
        table[arm] = by_state('psnr_%s_normal' % arm)
        for mode in ref_modes:
            if mode != 'normal':
                table['%s_%s' % (arm, mode)] = by_state('psnr_%s_%s' % (arm, mode))
    if do_cond and ARM_A1 in heads:
        for cm in COND_MODES:
            table['%s_cond_%s' % (ARM_A1, cm)] = by_state(
                'psnr_%s_cond_%s' % (ARM_A1, cm))
    if ARM_A0 in heads and ARM_A1 in heads:
        table['A1_minus_A0'] = {
            s: table[ARM_A1][s] - table[ARM_A0][s]
            for s in list(STATES) + ['mean']
        }
        if 'old_blind_H2' in table:
            table['A1_minus_blind'] = {
                s: table[ARM_A1][s] - table['old_blind_H2'][s]
                for s in list(STATES) + ['mean']
                if s in table['old_blind_H2']
            }

    feat = {}
    if ARM_A1 in heads and rows and 'dF_mean' in rows[0]:
        for k in ('dF_mean', 'dF_p50', 'dF_p90', 'dF_max', 'F_mean', 'r_F',
                  'dY_mean', 'r_decode', 'cos_dF_F'):
            feat[k] = by_state(k)

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
    if feat:
        dump_json(os.path.join(out_dir, 'feature_stats.json'), json_ready(feat))

    if a.verdict and ARM_A0 in heads and ARM_A1 in heads:
        dep = dict(normal=table[ARM_A1]['mean'], self=float('nan'),
                   zero=float('nan'), shuffled=float('nan'))
        if a.full_dep:
            dep['self'] = table['%s_self' % ARM_A1]['mean']
            dep['zero'] = table['%s_zero' % ARM_A1]['mean']
            dep['shuffled'] = table['%s_shuffled' % ARM_A1]['mean']
        cond = dict(normal=float('nan'), zero=float('nan'), shuffled=float('nan'))
        if do_cond:
            cond['normal'] = table['%s_cond_normal' % ARM_A1]['mean']
            cond['zero'] = table['%s_cond_zero' % ARM_A1]['mean']
            cond['shuffled'] = table['%s_cond_shuffled' % ARM_A1]['mean']
        deltas_by = {s: [] for s in STATES}
        for r in rows:
            dlt = r['psnr_%s_normal' % ARM_A1] - r['psnr_%s_normal' % ARM_A0]
            deltas_by[r['state']].append(dlt)
        blind = lock.get('frozen_blind_h2_psnr')
        pair_blind = None
        if blind is not None:
            # approximate pair vs frozen blind means only at state level in verdict;
            # per-image blind not available — leave pair_vs_blind None
            pass
        verdict = verdict_v3b41(
            table[ARM_A0], table[ARM_A1],
            safety[ARM_A0], safety[ARM_A1], dep, cond,
            blind_psnr=blind,
            pair_stats=pair_delta_stats_by_state(deltas_by),
            pair_vs_blind=pair_blind)
        verdict['step'] = a.step
        verdict['eval_repo_commit'] = live
        dump_json(os.path.join(out_dir, 'verdict.json'), json_ready(verdict))
        print('VERDICT', verdict['label'],
              'ΔA1-A0=%.3f ΔA1-blind=%.3f' % (
                  verdict['delta_a1_a0'], verdict['delta_a1_blind']),
              'next=', verdict['next_step'], flush=True)
        for s in STATES:
            print('  %s  A0=%.3f A1=%.3f d=%.3f' % (
                s, table[ARM_A0][s], table[ARM_A1][s],
                table['A1_minus_A0'][s]), flush=True)
        if do_cond:
            print('  cond normal/zero/shuffled = %.3f / %.3f / %.3f' % (
                cond['normal'], cond['zero'], cond['shuffled']), flush=True)
    else:
        for arm in heads:
            print('ARM', arm, 'mean=%.3f' % table[arm]['mean'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
