#!/usr/bin/env python
"""V3-A.7.2 Selective Coverage Closure Audit (zero training).

Reads artifact_lock.json; never writes/overwrites it.
Train-calibrated thresholds only. Official Test forbidden.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics                    # noqa: E402
from model.V3A5D2Verifier import V3A5D2Verifier                          # noqa: E402
from model.V3A5DEvidenceProbe import V3A5DEvidenceProbe                 # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, block_energy,  # noqa: E402
                          energy_mask, expand_gate, prepare_geometry,
                          target_geometry)
from v3a5d_runtime import analysis_maps_from_evidence                   # noqa: E402
from v3a5e_runtime import fit_ridge, predict_ridge                      # noqa: E402
from v3a5e1b_runtime import file_sha256                                 # noqa: E402
from v3a6_runtime import dump_json, git_head, hard_verify_lock          # noqa: E402
from v3a7_runtime import block_utility                                  # noqa: E402
from v3a71_runtime import mse_image                                     # noqa: E402
from v3a72_runtime import (COVERAGES, FORMAL_LOCK_KEYS,                 # noqa: E402
                           PRIMARY_COVERAGE, QUALITY_COVERAGES,
                           apply_selective_q, binary_block_oracle_q,
                           calibrate_train_thresholds,
                           cluster_bootstrap_bins,
                           coverage_quality_monotonic, extract_z,
                           inspect_f2_ckpt, json_ready, per_image_topk_q,
                           q_to_grid, ranked_bins_with_ids,
                           regional_regret, safety_from_deltas,
                           stack_f0, state_composition, verdict_v3a72)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3a72_selective_closure'
REF_H, REF_W = 400, 600
GH, GW = 64, 64


def std_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    return mu, sd


def apply_std(X, mu, sd):
    return (X - mu) / sd


def mean_dict(d):
    vals = [float(v) for v in d.values()]
    return float(np.mean(vals)) if vals else float('nan')


def summarize_pairs(rows, key):
    by = {s: [] for s in STATES}
    allv = []
    for r in rows:
        by[r['state']].append(r[key])
        allv.append(r[key])
    out = dict(mean=float(np.mean(allv)), n=len(allv))
    for s in STATES:
        out[s] = float(np.mean(by[s])) if by[s] else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--formal', action='store_true', default=True)
    a = ap.parse_args(_CLI)

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if not os.path.isfile(lock_path):
        raise SystemExit('missing lock; run scripts/setup_v3a72.py first')
    # audit must not overwrite lock
    lock_stat = os.stat(lock_path)
    lock = json.load(open(lock_path))
    missing = [k for k in FORMAL_LOCK_KEYS if k not in lock]
    if missing:
        raise SystemExit('lock missing keys: %s' % missing)
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official_test_allowed must be false')

    live = dict(
        proposal_sha256=file_sha256(lock['proposal_ckpt']),
        split_sha256=file_sha256(lock['split_json']),
        mismatch_train_sha256=file_sha256(lock['mismatch_train']),
        mismatch_dev_sha256=file_sha256(lock['mismatch_dev']),
        f2_ckpt_sha256=file_sha256(lock['f2_ckpt']),
        official_test_allowed=False,
        geometry='g64',
        f2_ckpt_step=20000,
        f2_ckpt_arm='A1_multiscale',
        f2_source_stage='V3-A.5D2.1',
        reference_variant=lock['reference_variant'],
        f0_source='V3A5E_FEATURE_NAMES',
        per_image=lock['per_image'],
        bootstrap_B=lock['bootstrap_B'],
        coverage_grid=lock['coverage_grid'],
        seed=lock['seed'],
        f2_ckpt=lock['f2_ckpt'],
        repo_commit=git_head(),
        energy_stats_sha256=lock['energy_stats_sha256'],
    )
    expected = {k: lock[k] for k in FORMAL_LOCK_KEYS}
    hard_verify_lock(lock, expected, formal=a.formal)
    # live file hashes + frozen semantics; repo_commit may drift after setup
    file_keys = ('proposal_sha256', 'split_sha256', 'mismatch_train_sha256',
                 'mismatch_dev_sha256', 'f2_ckpt_sha256', 'official_test_allowed',
                 'geometry', 'f2_ckpt_step', 'f2_ckpt_arm', 'f2_source_stage',
                 'f0_source', 'per_image', 'bootstrap_B', 'coverage_grid',
                 'seed', 'f2_ckpt', 'energy_stats_sha256', 'reference_variant')
    hard_verify_lock(lock, {k: live[k] for k in file_keys}, formal=a.formal)

    blob = torch.load(lock['f2_ckpt'], map_location='cpu')
    meta = inspect_f2_ckpt(blob, lock['f2_ckpt'])
    if not meta['ok_step'] or not meta['ok_arm']:
        raise SystemExit('F2 ckpt semantic fail %s' % meta)

    rows_path = lock['v71_block_rows']
    npz = np.load(rows_path, allow_pickle=True)
    Xtr = {k: npz['train_' + k].astype(np.float64) for k in ('F0', 'F2')}
    Xdv = {k: npz['dev_' + k].astype(np.float64) for k in ('F0', 'F2')}
    ytr = npz['train_U'].astype(np.float64)
    ydv = npz['dev_U'].astype(np.float64)
    id_tr = npz['train_image_id']
    id_dv = npz['dev_image_id']
    st_tr = npz['train_state']
    st_dv = npz['dev_state']

    probes, train_scores, dev_scores, taus = {}, {}, {}, {}
    for feat in ('F0', 'F2'):
        mu, sd = std_fit(Xtr[feat])
        w = fit_ridge(apply_std(Xtr[feat], mu, sd), ytr, l2=1e-1)
        trs = predict_ridge(w, apply_std(Xtr[feat], mu, sd))
        dvs = predict_ridge(w, apply_std(Xdv[feat], mu, sd))
        train_scores[feat] = trs
        dev_scores[feat] = dvs
        taus[feat] = calibrate_train_thresholds(trs, coverages=QUALITY_COVERAGES)
        probes[feat] = dict(mu=mu, sd=sd, w=w)
        print('probe %s train_score mean=%.4g tau10=%.4g' % (
            feat, float(trs.mean()), taus[feat][0.10]), flush=True)

    bins = {'overall': {}, 'by_state': {}}
    composition = {}
    bootstrap = {}
    for feat in ('F0', 'F2'):
        bins['overall'][feat] = ranked_bins_with_ids(
            dev_scores[feat], ydv, id_dv)
        bins['by_state'][feat] = {}
        for st in STATES:
            m = st_dv == st
            bins['by_state'][feat][st] = ranked_bins_with_ids(
                dev_scores[feat][m], ydv[m], id_dv[m])
        composition[feat] = dict(
            top10=state_composition(dev_scores[feat], st_dv, 0.10),
            top25=state_composition(dev_scores[feat], st_dv, 0.25),
        )
        bootstrap[feat] = dict(
            overall=cluster_bootstrap_bins(
                id_dv, dev_scores[feat], ydv, st_dv, None,
                B=int(lock['bootstrap_B']), seed=int(lock['seed'])),
        )
        for st in STATES:
            bootstrap[feat][st] = cluster_bootstrap_bins(
                id_dv, dev_scores[feat], ydv, st_dv, st,
                B=int(lock['bootstrap_B']), seed=int(lock['seed']))

    dump_json(os.path.join(a.root, 'diagnostics', 'ranked_by_state.json'),
              json_ready(bins))
    dump_json(os.path.join(a.root, 'diagnostics', 'state_composition.json'),
              json_ready(composition))
    dump_json(os.path.join(a.root, 'diagnostics', 'bootstrap_ci.json'),
              json_ready(bootstrap))
    dump_json(os.path.join(a.root, 'diagnostics', 'threshold_calibration.json'),
              json_ready(dict(taus=taus, note='train-only percentiles')))

    # GPU full-map policy on dev
    thr = float(lock['energy_threshold'])
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

    proposal = load_proposal(lock['proposal_ckpt'], a.device)
    probe = V3A5DEvidenceProbe(proposal.proposal).to(a.device).eval()
    z_tr = V3A5D2Verifier('A1_multiscale').to(a.device).eval()
    z_tr.load_state_dict(blob['model'], strict=True)
    for p in z_tr.parameters():
        p.requires_grad_(False)
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    pair_rows = []
    sel_u = {feat: {c: [] for c in QUALITY_COVERAGES} for feat in ('F0', 'F2')}
    t0 = time.time()
    print('=== full-map policy eval n=%d ===' % n, flush=True)
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                D, _ = correction(proposal.proposal, t['Y0'], t['R'])
                U = block_utility(t['Y0'], t['H'], D, geom)
                en = block_energy(D, geom)
                mask_t = energy_mask(en, thr)
                mask = mask_t.reshape(-1).float().cpu().numpy()
                uflat = U.reshape(-1).float().cpu().numpy()
                _, _, evidence = probe(t['Y0'], t['R'], check_equiv=False)
                maps = analysis_maps_from_evidence(evidence)
                F0 = stack_f0(maps, D, geom)
                Z = extract_z(z_tr, t['X'], t['Y0'], t['R'], geom)
                f0 = F0.reshape(F0.shape[1], -1).float().cpu().numpy().T
                f2 = Z.reshape(Z.shape[1], -1).float().cpu().numpy().T
                sc = {}
                for feat, arr in (('F0', f0), ('F2', f2)):
                    sc[feat] = predict_ridge(
                        probes[feat]['w'],
                        apply_std(arr, probes[feat]['mu'], probes[feat]['sd']))
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
                q_bin = binary_block_oracle_q(uflat, mask)
                y_base = t['Y0']
                y_r1 = t['Y0'] + D
                y_ao = t['Y0'] + tgt['q_full'] * D
                y_bin = t['Y0'] + expand_gate(
                    q_to_grid(q_bin, GH, GW, a.device), geom) * D
                y_gq = t['Y0'] + 0.5 * D
                p_base = float(_metrics(y_base, t['H'])[0])
                p_r1 = float(_metrics(y_r1, t['H'])[0])
                p_ao = float(_metrics(y_ao, t['H'])[0])
                p_bin = float(_metrics(y_bin, t['H'])[0])
                p_gq = float(_metrics(y_gq, t['H'])[0])
                mse_b = float(mse_image(y_base, t['H']))
                mse_r = float(mse_image(y_r1, t['H']))
                mse_bin = float(mse_image(y_bin, t['H']))
                rec = dict(
                    name=t['name'], state=state,
                    psnr_base=p_base, psnr_r1=p_r1, psnr_ao=p_ao,
                    psnr_bin_oracle=p_bin, psnr_gconst=p_gq,
                    mse_base=mse_b, mse_r1=mse_r, mse_bin_oracle=mse_bin,
                    n_valid=int((mask >= 0.5).sum()),
                )
                for feat in ('F0', 'F2'):
                    for c in QUALITY_COVERAGES:
                        q = apply_selective_q(sc[feat], taus[feat][c], mask)
                        yg = q_to_grid(q, GH, GW, a.device)
                        yhat = t['Y0'] + expand_gate(yg, geom) * D
                        p = float(_metrics(yhat, t['H'])[0])
                        mse_p = float(mse_image(yhat, t['H']))
                        cov = float(q[mask >= 0.5].mean()) if (mask >= 0.5).any() else 0.0
                        rec['p1_%s_c%02d_psnr' % (feat, int(round(c * 100)))] = p
                        rec['p1_%s_c%02d_regret' % (feat, int(round(c * 100)))] = (
                            regional_regret(mse_p, mse_bin))
                        rec['p1_%s_c%02d_cov' % (feat, int(round(c * 100)))] = cov
                        sel = (q > 0.5)
                        if sel.any():
                            sel_u[feat][c].extend(uflat[sel].tolist())
                        q2 = per_image_topk_q(sc[feat], mask, c)
                        yg2 = q_to_grid(q2, GH, GW, a.device)
                        y2 = t['Y0'] + expand_gate(yg2, geom) * D
                        rec['p2_%s_c%02d_psnr' % (feat, int(round(c * 100)))] = (
                            float(_metrics(y2, t['H'])[0]))
                        rec['p2_%s_c%02d_regret' % (feat, int(round(c * 100)))] = (
                            regional_regret(float(mse_image(y2, t['H'])), mse_bin))
                pair_rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d (%.0fs)' % (i + 1, n, time.time() - t0), flush=True)

    if os.stat(lock_path).st_mtime != lock_stat.st_mtime:
        raise SystemExit('artifact_lock was mutated during audit')

    frozen = lock['frozen_baseline_psnr']
    oracle = dict(
        Base=summarize_pairs(pair_rows, 'psnr_base'),
        R1=summarize_pairs(pair_rows, 'psnr_r1'),
        BinaryBlockOracle=summarize_pairs(pair_rows, 'psnr_bin_oracle'),
        AO64=summarize_pairs(pair_rows, 'psnr_ao'),
        global_constant_q05=summarize_pairs(pair_rows, 'psnr_gconst'),
        V3A6_A1=frozen['v3a6_A1'],
        V3A7_A1=frozen['v3a7_A1'],
        V3A6_A1_by_state=frozen['v3a6_A1_by_state'],
        V3A7_A1_by_state=frozen['v3a7_A1_by_state'],
        BinaryBlockOracle_minus_Base=(
            summarize_pairs(pair_rows, 'psnr_bin_oracle')['mean']
            - summarize_pairs(pair_rows, 'psnr_base')['mean']),
        BinaryBlockOracle_minus_R1=(
            summarize_pairs(pair_rows, 'psnr_bin_oracle')['mean']
            - summarize_pairs(pair_rows, 'psnr_r1')['mean']),
        legacy_v3a71_unmasked_block_F2=dict(psnr=19.824, regret=0.00284,
                                            note='diagnostic only'),
    )
    dump_json(os.path.join(a.root, 'diagnostics', 'oracle_bounds.json'),
              json_ready(oracle))

    policies = {}
    quality = {}
    safety = {}
    for feat in ('F0', 'F2'):
        policies[feat] = {'P1': {}, 'P2': {}}
        quality[feat] = {}
        for c in QUALITY_COVERAGES:
            kps = 'p1_%s_c%02d_psnr' % (feat, int(round(c * 100)))
            krg = 'p1_%s_c%02d_regret' % (feat, int(round(c * 100)))
            kcv = 'p1_%s_c%02d_cov' % (feat, int(round(c * 100)))
            su = np.asarray(sel_u[feat][c], dtype=np.float64)
            quality[feat][c] = dict(
                coverage=summarize_pairs(pair_rows, kcv)['mean'],
                mean_true_u=float(su.mean()) if su.size else float('nan'),
                pos_rate=float((su > 0).mean()) if su.size else float('nan'),
                mean_psnr=summarize_pairs(pair_rows, kps)['mean'],
                regional_regret=summarize_pairs(pair_rows, krg)['mean'],
            )
            if c in COVERAGES:
                policies[feat]['P1'][c] = dict(
                    realized_coverage=quality[feat][c]['coverage'],
                    mean_psnr=quality[feat][c]['mean_psnr'],
                    by_state={s: summarize_pairs(pair_rows, kps)[s] for s in STATES},
                    regional_regret=quality[feat][c]['regional_regret'],
                )
                k2p = 'p2_%s_c%02d_psnr' % (feat, int(round(c * 100)))
                k2r = 'p2_%s_c%02d_regret' % (feat, int(round(c * 100)))
                policies[feat]['P2'][c] = dict(
                    mean_psnr=summarize_pairs(pair_rows, k2p)['mean'],
                    by_state={s: summarize_pairs(pair_rows, k2p)[s] for s in STATES},
                    regional_regret=summarize_pairs(pair_rows, k2r)['mean'],
                    diagnostic_only=True,
                )
        mu_map = {c: quality[feat][c]['mean_true_u'] for c in QUALITY_COVERAGES}
        quality[feat]['monotonic_mean_u'] = coverage_quality_monotonic(mu_map)

    # safety for primary P1-10 F0 and F2, and R1
    def deltas(key_psnr):
        out = {s: [] for s in STATES}
        names = {s: [] for s in STATES}
        for r in pair_rows:
            d = r[key_psnr] - r['psnr_base']
            out[r['state']].append(d)
            names[r['state']].append('%s|%s' % (r['name'], r['state']))
        return out, names

    for feat in ('F0', 'F2'):
        key = 'p1_%s_c%02d_psnr' % (feat, int(round(PRIMARY_COVERAGE * 100)))
        dlt, nms = deltas(key)
        safety['P1_%s_10' % feat] = {
            s: safety_from_deltas(dlt[s], nms[s]) for s in STATES}
        safety['P1_%s_10' % feat]['overall'] = safety_from_deltas(
            np.concatenate([dlt[s] for s in STATES]),
            np.concatenate([nms[s] for s in STATES]))
    d_r1, n_r1 = deltas('psnr_r1')
    safety['R1'] = {s: safety_from_deltas(d_r1[s], n_r1[s]) for s in STATES}
    safety['R1']['overall'] = safety_from_deltas(
        np.concatenate([d_r1[s] for s in STATES]),
        np.concatenate([n_r1[s] for s in STATES]))

    dump_json(os.path.join(a.root, 'diagnostics', 'policy_results.json'),
              json_ready(dict(policies=policies, quality=quality, safety=safety)))

    other_neg = False
    for st in ('true_dark_g0.5', 'mismatch'):
        m = bins['by_state']['F2'][st]['top_10']['mean_u']
        if math_isnan_or_neg(m):
            other_neg = True

    p1_primary = policies['F2']['P1'][PRIMARY_COVERAGE]
    payload = dict(
        bootstrap_correct=bootstrap['F2']['correct'],
        bins_correct=bins['by_state']['F2']['correct'],
        composition_top10=composition['F2']['top10'],
        p1_primary=p1_primary,
        psnr_base=oracle['Base']['mean'],
        psnr_v3a6_a1=float(frozen['v3a6_A1']),
        safety_p1=safety['P1_F2_10'],
        safety_r1=safety['R1'],
        coverage_monotonic=bool(quality['F2']['monotonic_mean_u']),
        other_state_top10_negative=other_neg,
    )
    verdict = verdict_v3a72(payload)
    verdict['f0_p1_10_psnr'] = policies['F0']['P1'][PRIMARY_COVERAGE]['mean_psnr']
    verdict['prefer_f0'] = bool(
        policies['F0']['P1'][PRIMARY_COVERAGE]['mean_psnr']
        >= policies['F2']['P1'][PRIMARY_COVERAGE]['mean_psnr'] - 1e-6)
    dump_json(os.path.join(a.root, 'diagnostics', 'verdict.json'),
              json_ready(verdict))

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  meaning:', verdict['meaning'], flush=True)
    print('  correct top10 CI_lo=%.4g gap_lo=%.4g rank_ok=%s'
          % (verdict['correct_top10_ci_lo'], verdict['correct_gap_ci_lo'],
             verdict['rank_ok']), flush=True)
    print('  P1-10 F2 PSNR=%.3f  F0=%.3f  Base=%.3f  BinOracle=%.3f  V3A6A1=%.3f'
          % (p1_primary['mean_psnr'],
             policies['F0']['P1'][PRIMARY_COVERAGE]['mean_psnr'],
             oracle['Base']['mean'], oracle['BinaryBlockOracle']['mean'],
             float(frozen['v3a6_A1'])), flush=True)
    return 0


def math_isnan_or_neg(m, thr=-1e-3):
    import math
    m = float(m)
    return (not math.isfinite(m)) or m < thr


if __name__ == '__main__':
    raise SystemExit(main())
