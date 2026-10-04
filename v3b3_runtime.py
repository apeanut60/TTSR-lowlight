"""V3-B.3 global RGB statistics prior — lock, cache, verdict."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from model.V3BGlobalStats import GLOBAL_STAT_NAMES
from v3a5_runtime import STATES
from v3b_runtime import (CKPT_STEPS, DEFAULT_UPDATES, EVAL_STEPS, GRAD_ACCUM, LR,
                         SEED, b0_loss, proposal_core)
from v3b2_runtime import pair_delta_stats_by_state, require_ckpt_blob

ARMS = ('A0_b0_replay', 'A1_global_stats')
ARM_A0 = 'A0_b0_replay'
ARM_A1 = 'A1_global_stats'
FORMAL_LOCK_KEYS_B3 = (
    'repo_commit', 'base_cache_metadata_sha256', 'proposal_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant', 'architecture', 'arms',
    'a0_init_sha', 'a1_init_sha', 'common_weight_sha',
    'global_stat_names', 'stat_source_space', 'std_unbiased',
    'global_stats_sha256', 'normalization_method',
    'mlp_architecture', 'broadcast_channels',
    'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
)


def safety_plus(delta_psnr, names=None):
    from v3a72_runtime import safety_from_deltas
    s = safety_from_deltas(delta_psnr, names)
    d = np.asarray(delta_psnr, dtype=np.float64).reshape(-1)
    s['p05'] = float(np.percentile(d, 5)) if d.size else float('nan')
    return s


def verdict_v3b3(psnr_a0: Dict, psnr_a1: Dict, safety_a0: Dict,
                 safety_a1: Dict, dep_global: Dict,
                 pair_stats: Optional[Dict] = None,
                 usage: Optional[Dict] = None) -> Dict:
    """Case A–D from V3-B.3 plan §13. Primary = A1 − A0."""
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

    lh_a0_m = lh(safety_a0, 'mismatch')
    lh_a1_m = lh(safety_a1, 'mismatch')
    tail_worse = (math.isfinite(lh_a0_m) and math.isfinite(lh_a1_m)
                  and lh_a1_m > lh_a0_m + 0.02)
    mismatch_ok = bool(g_mis >= -0.02)
    correct_ok = bool(g_cor >= -0.02)
    dark_ok = bool(g_dark >= -0.02)

    nrm = float(dep_global.get('normal', float('nan')))
    zro = float(dep_global.get('zero', float('nan')))
    abl_ok = bool(math.isfinite(nrm) and math.isfinite(zro) and nrm >= zro + 0.02)
    abl_weak = bool(math.isfinite(nrm) and math.isfinite(zro) and abs(nrm - zro) < 0.01)

    mean_strong = bool(delta >= 0.05)
    mean_weak = bool(delta >= 0.02 and delta < 0.05)
    nullish = bool(abs(delta) < 0.02)
    one_state_plus = bool(g_cor >= 0.05 or g_dark >= 0.05)

    if (not mismatch_ok) or tail_worse:
        label, nxt, meaning = (
            'V3B3_CASE_D_UNSAFE', 'drop_B3_start_B4',
            'global prior repeats B1-style mismatch/tail harm')
    elif mean_strong and correct_ok and dark_ok and mismatch_ok and abl_ok:
        label, nxt, meaning = (
            'V3B3_CASE_A_STRONG_GO', 'B3.1_richer_global_or_B4',
            'global RGB stats beat A0 on mean without mismatch harm')
    elif mean_weak and one_state_plus and mismatch_ok and (not tail_worse) and abl_ok:
        label, nxt, meaning = (
            'V3B3_CASE_B_WEAK_GO', 'B3.1_richer_global_prior',
            'weak but useful global prior with real ablation')
    elif nullish or abl_weak:
        label, nxt, meaning = (
            'V3B3_CASE_C_NULL', 'drop_B3_start_B4',
            'global stats null or unused; go B4 feature-level')
    else:
        label, nxt, meaning = (
            'V3B3_CASE_C_NULL', 'drop_B3_start_B4',
            'global stats inconclusive vs A0')

    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(A0=a0, A1=a1), delta_a1_a0=delta,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        correct_ok=correct_ok, dark_ok=dark_ok, mismatch_ok=mismatch_ok,
        tail_worse=tail_worse, abl_ok=abl_ok, abl_weak=abl_weak,
        mean_strong=mean_strong, mean_weak=mean_weak,
        dep_normal_zero=float(nrm - zro) if math.isfinite(nrm) and math.isfinite(zro)
        else float('nan'),
        large_harm_mismatch=dict(A0=lh_a0_m, A1=lh_a1_m),
        global_stat_names=list(GLOBAL_STAT_NAMES),
    )
    if pair_stats is not None:
        out['pair_a1_minus_a0'] = pair_stats
    if usage is not None:
        out['usage'] = usage
    return out
