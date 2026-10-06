"""V3-B.4 H/2 decoder feature residual — lock, verdict, ckpt hygiene."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from model.V3BFeatureBridge import INJECTION_POINT
from v3a5_runtime import STATES
from v3b_runtime import CKPT_STEPS, DEFAULT_UPDATES, EVAL_STEPS, GRAD_ACCUM, LR, SEED
from v3b2_runtime import pair_delta_stats_by_state
from v3b3_runtime import safety_plus

ARMS = ('A0_b0_replay', 'A1_h2_feature')
ARM_A0 = 'A0_b0_replay'
ARM_A1 = 'A1_h2_feature'
FORMAL_LOCK_KEYS_B4 = (
    'repo_commit', 'base_cache_metadata_sha256', 'proposal_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant', 'injection_point', 'base_feature_ch',
    'reference_feature_ch', 'architecture_a0', 'architecture_a1',
    'arms', 'a0_init_sha', 'a1_init_sha', 'adapter_n_params',
    'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
)


def require_ckpt_blob_b4(blob, *, arm, step, init_sha, proposal_sha,
                         repo_commit, injection_point=INJECTION_POINT,
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
    if errs:
        msg = 'ckpt HARD FAIL:\n  ' + '\n  '.join(errs)
        if formal:
            raise SystemExit(msg)
        raise ValueError(msg)
    return True


def verdict_v3b4(psnr_a0: Dict, psnr_a1: Dict, safety_a0: Dict,
                 safety_a1: Dict, dep_a1: Dict,
                 pair_stats: Optional[Dict] = None) -> Dict:
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
    tail_worse = bool(math.isfinite(lh_a0) and math.isfinite(lh_a1)
                      and lh_a1 > lh_a0 + 0.02)
    correct_ok = bool(g_cor >= -0.02)
    dark_ok = bool(g_dark >= -0.02)
    mismatch_ok = bool(g_mis >= -0.02)

    nrm = float(dep_a1.get('normal', float('nan')))
    slf = float(dep_a1.get('self', float('nan')))
    ref_ok = bool(math.isfinite(nrm) and math.isfinite(slf) and nrm >= slf + 0.03)
    ignored = bool(math.isfinite(nrm) and math.isfinite(slf) and abs(nrm - slf) < 0.02)

    mean_strong = bool(delta >= 0.05)
    nullish = bool(abs(delta) < 0.02)
    mis_better = bool(g_mis >= 0.03 or (
        math.isfinite(lh_a0) and math.isfinite(lh_a1) and (lh_a0 - lh_a1) >= 0.05))

    if (delta < 0.0) or (g_mis < -0.02) or tail_worse:
        label, nxt, meaning = (
            'V3B4_CASE_D_HARM', 'h2_injection_fail_no_h4',
            'H/2 feature injection hurts mean or mismatch/tail; do not return to H/4')
    elif mean_strong and correct_ok and dark_ok and mismatch_ok and tail_ok and ref_ok:
        label, nxt, meaning = (
            'V3B4_CASE_A_STRONG_GO', 'B4.1_base_feature_aware',
            'H/2 feature residual beats RGB residual with real T use')
    elif (not mean_strong) and mis_better and correct_ok and dark_ok and ref_ok:
        label, nxt, meaning = (
            'V3B4_CASE_B_ROBUSTNESS_GO', 'B4.1_base_feature_aware',
            'H/2 improves mismatch reliability vs RGB residual')
    elif nullish:
        label, nxt, meaning = (
            'V3B4_CASE_C_EQUIVALENT', 'injection_point_closure_fullres_only',
            'H/2 ≈ RGB residual; only full-res-before-mapping closure allowed')
    else:
        label, nxt, meaning = (
            'V3B4_CASE_C_EQUIVALENT', 'injection_point_closure_fullres_only',
            'H/2 feature residual inconclusive vs RGB A0')

    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(A0=a0, A1=a1), delta_a1_a0=delta,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        correct_ok=correct_ok, dark_ok=dark_ok, mismatch_ok=mismatch_ok,
        tail_ok=tail_ok, ref_ok=ref_ok, ignored=ignored,
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        large_harm_mismatch=dict(A0=lh_a0, A1=lh_a1),
        injection_point=INJECTION_POINT,
    )
    if pair_stats is not None:
        out['pair_a1_minus_a0'] = pair_stats
    return out
