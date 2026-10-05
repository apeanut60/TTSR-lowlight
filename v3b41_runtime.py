"""V3-B.4.1 base-conditioned H/2 feature residual — lock, verdict, ckpt hygiene."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from model.V3BFeatureBridge import INJECTION_POINT
from v3a5_runtime import STATES
from v3b_runtime import CKPT_STEPS, DEFAULT_UPDATES, EVAL_STEPS, GRAD_ACCUM, LR, SEED
from v3b2_runtime import pair_delta_stats_by_state
from v3b3_runtime import safety_plus

ARMS = ('A0_b0_replay', 'A1_base_cond_h2')
ARM_A0 = 'A0_b0_replay'
ARM_A1 = 'A1_base_cond_h2'
BASE_CONDITIONED = True
BASE_FEATURE_CH = 80
REF_INPUT_CH = 96

FORMAL_LOCK_KEYS_B41 = (
    'repo_commit', 'base_cache_metadata_sha256', 'proposal_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant', 'injection_point', 'bridge_file_sha',
    'base_conditioned', 'base_feature_ch', 'ref_input_ch',
    'reference_feature_ch', 'architecture_a0', 'architecture_a1',
    'arms', 'a0_init_sha', 'a1_init_sha', 'adapter_n_params',
    'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
)


def require_ckpt_blob_b41(blob, *, arm, step, init_sha, proposal_sha,
                          repo_commit, injection_point=INJECTION_POINT,
                          base_conditioned=BASE_CONDITIONED,
                          base_feature_ch=BASE_FEATURE_CH,
                          ref_input_ch=REF_INPUT_CH,
                          formal=True):
    errs = []
    if int(blob.get('step', -1)) != int(step):
        errs.append('step: blob=%r want=%r' % (blob.get('step'), step))
    if blob.get('arm') != arm:
        errs.append('arm: blob=%r want=%r' % (blob.get('arm'), arm))
    if blob.get('objective') != 'mse_reconstruction':
        errs.append('objective: %r' % blob.get('objective'))
    if int(blob.get('seed', -1)) != int(SEED):
        errs.append('seed: %r' % blob.get('seed'))
    if blob.get('init_sha') != init_sha:
        errs.append('init_sha mismatch')
    if blob.get('proposal_sha') != proposal_sha:
        errs.append('proposal_sha mismatch')
    if blob.get('repo_commit') != repo_commit:
        errs.append('ckpt repo_commit mismatch')
    if blob.get('injection_point') != injection_point:
        errs.append('injection_point: blob=%r want=%r'
                    % (blob.get('injection_point'), injection_point))
    if arm == ARM_A1:
        if blob.get('base_conditioned') is not True:
            errs.append('base_conditioned: blob=%r want=True'
                        % (blob.get('base_conditioned'),))
        if int(blob.get('base_feature_ch', -1)) != int(base_feature_ch):
            errs.append('base_feature_ch: blob=%r want=%r'
                        % (blob.get('base_feature_ch'), base_feature_ch))
        if int(blob.get('ref_input_ch', -1)) != int(ref_input_ch):
            errs.append('ref_input_ch: blob=%r want=%r'
                        % (blob.get('ref_input_ch'), ref_input_ch))
    if errs:
        msg = 'ckpt HARD FAIL:\n  ' + '\n  '.join(errs)
        if formal:
            raise SystemExit(msg)
        raise ValueError(msg)
    return True


def verdict_v3b41(psnr_a0: Dict, psnr_a1: Dict, safety_a0: Dict,
                  safety_a1: Dict, dep_a1: Dict, cond_a1: Dict,
                  blind_psnr: Optional[Dict] = None,
                  pair_stats: Optional[Dict] = None,
                  pair_vs_blind: Optional[Dict] = None) -> Dict:
    def mean(d):
        vals = [float(d[s]) for s in STATES if s in d]
        return float(np.mean(vals)) if vals else float('nan')

    a0, a1 = mean(psnr_a0), mean(psnr_a1)
    delta = a1 - a0
    g_cor = float(psnr_a1.get('correct', float('nan'))
                  - psnr_a0.get('correct', float('nan')))
    g_dark = float(psnr_a1.get('true_dark_g0.5', float('nan'))
                   - psnr_a0.get('true_dark_g0.5', float('nan')))
    g_mis = float(psnr_a1.get('mismatch', float('nan'))
                  - psnr_a0.get('mismatch', float('nan')))

    def lh(safety, st):
        return float((safety.get(st) or {}).get('large_harm_rate', float('nan')))

    lh_a0 = lh(safety_a0, 'mismatch')
    lh_a1 = lh(safety_a1, 'mismatch')
    tail_ok = bool(math.isfinite(lh_a0) and math.isfinite(lh_a1)
                   and lh_a1 <= lh_a0 + 0.02)
    correct_ok = bool(g_cor >= -0.02)
    dark_ok = bool(g_dark >= -0.02)
    mismatch_ok = bool(g_mis >= -0.02)

    nrm = float(dep_a1.get('normal', float('nan')))
    slf = float(dep_a1.get('self', float('nan')))
    ref_ok = bool(math.isfinite(nrm) and math.isfinite(slf) and nrm >= slf + 0.03)
    ignored = bool(math.isfinite(nrm) and math.isfinite(slf) and abs(nrm - slf) < 0.02)

    c_n = float(cond_a1.get('normal', float('nan')))
    c_z = float(cond_a1.get('zero', float('nan')))
    c_s = float(cond_a1.get('shuffled', float('nan')))
    cond_ok = bool(math.isfinite(c_n) and math.isfinite(c_s) and c_n >= c_s + 0.02)
    cond_unused = bool(
        math.isfinite(c_n) and math.isfinite(c_s) and abs(c_n - c_s) < 0.02
        and math.isfinite(c_z) and abs(c_n - c_z) < 0.02)

    blind_mean = float('nan')
    delta_vs_blind = float('nan')
    if blind_psnr is not None:
        blind_mean = mean(blind_psnr)
        delta_vs_blind = a1 - blind_mean

    mean_strong = bool(delta >= 0.05)
    beats_blind = bool(math.isfinite(delta_vs_blind) and delta_vs_blind >= 0.05)
    not_above_blind = bool(math.isfinite(delta_vs_blind) and delta_vs_blind <= 0.0)
    le_a0 = bool(delta <= 0.02)
    below_a0 = bool(delta < 0.0)
    mis_better = bool(g_mis >= 0.03 or (
        math.isfinite(lh_a0) and math.isfinite(lh_a1) and (lh_a0 - lh_a1) >= 0.05))
    weak_dep = bool((not ref_ok) or cond_unused or (not cond_ok))

    if mean_strong and correct_ok and dark_ok and mismatch_ok and tail_ok and ref_ok and cond_ok:
        label, nxt, meaning = (
            'V3B41_CASE_A_STRONG_GO', 'B4.2_minimal_rebair',
            'Base-conditioned H/2 beats RGB residual with real T and F_dec use')
    elif (not mean_strong) and mis_better and correct_ok and dark_ok and ref_ok and cond_ok:
        label, nxt, meaning = (
            'V3B41_CASE_B_ROBUSTNESS_GO', 'B4.2_minimal_rebair',
            'Base-conditioned H/2 improves mismatch reliability vs RGB residual')
    elif beats_blind and le_a0:
        # Conditioning helps vs blind H/2 but does not beat RGB B0 (may be slightly below).
        label, nxt, meaning = (
            'V3B41_CASE_C_FIXES_BLIND_STILL_BELOW_B0',
            'close_feature_level_no_mha_rdb',
            'Conditioning fixes blind H/2 but still not better than RGB B0')
    elif not_above_blind or below_a0 or weak_dep:
        label, nxt, meaning = (
            'V3B41_CASE_D_CLOSED', 'feature_level_route_closed',
            'Base-conditioned H/2 fails closure vs B0/blind or weak dependence')
    else:
        label, nxt, meaning = (
            'V3B41_CASE_D_CLOSED', 'feature_level_route_closed',
            'Base-conditioned H/2 inconclusive / closure fail')
    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(A0=a0, A1=a1, blind_h2=blind_mean),
        delta_a1_a0=delta, delta_a1_blind=delta_vs_blind,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        correct_ok=correct_ok, dark_ok=dark_ok, mismatch_ok=mismatch_ok,
        tail_ok=tail_ok, ref_ok=ref_ok, ignored=ignored,
        cond_ok=cond_ok, cond_unused=cond_unused,
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        cond_normal_zero=float(c_n - c_z) if math.isfinite(c_n) and math.isfinite(c_z)
        else float('nan'),
        cond_normal_shuffled=float(c_n - c_s) if math.isfinite(c_n) and math.isfinite(c_s)
        else float('nan'),
        large_harm_mismatch=dict(A0=lh_a0, A1=lh_a1),
        injection_point=INJECTION_POINT,
        base_conditioned=True,
    )
    if pair_stats is not None:
        out['pair_a1_minus_a0'] = pair_stats
    if pair_vs_blind is not None:
        out['pair_a1_minus_blind'] = pair_vs_blind
    return out
