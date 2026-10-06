"""V4.1a ground-then-transfer — lock, verdict, cache, ckpt hygiene."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch

from model.V4RefGrounder import FORWARD_DIR, GROUND_DIR
from v3a5_runtime import STATES
from v3b_runtime import CKPT_STEPS, DEFAULT_UPDATES, EVAL_STEPS, GRAD_ACCUM, LR, SEED
from v3b2_runtime import pair_delta_stats_by_state
from v3b3_runtime import safety_plus

ARM_A0 = 'A0_frozen_b0'
ARM_A1 = 'A1_grounded_ref'
ARMS = (ARM_A0, ARM_A1)

FORMAL_LOCK_KEYS_V41 = (
    'repo_commit', 'base_cache_metadata_sha256', 'proposal_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant',
    'b0_head_ckpt', 'b0_head_ckpt_sha256', 'b0_head_state_sha',
    'ground_direction', 'forward_direction',
    'architecture', 'grounder_zero_output', 'frozen_b0',
    'trainable_modules', 'grounder_init_sha',
    'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
)


def require_ckpt_blob_v41(blob, *, arm, step, init_sha, proposal_sha,
                          repo_commit, b0_head_state_sha,
                          ground_direction=GROUND_DIR,
                          forward_direction=FORWARD_DIR, formal=True):
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
    if blob.get('b0_head_state_sha') != b0_head_state_sha:
        errs.append('b0_head_state_sha mismatch')
    if blob.get('ground_direction') != ground_direction:
        errs.append('ground_direction: blob=%r want=%r'
                    % (blob.get('ground_direction'), ground_direction))
    if blob.get('forward_direction') != forward_direction:
        errs.append('forward_direction: blob=%r want=%r'
                    % (blob.get('forward_direction'), forward_direction))
    if blob.get('frozen_b0') is not True:
        errs.append('frozen_b0: blob=%r want=True' % (blob.get('frozen_b0'),))
    if errs:
        msg = 'ckpt HARD FAIL:\n  ' + '\n  '.join(errs)
        if formal:
            raise SystemExit(msg)
        raise ValueError(msg)
    return True


class FrozenQuadCache(object):
    """Cache frozen (F0, FR, T_low, T_raw)."""

    def __init__(self, max_gb=16.0):
        self.store = {}
        self.hits = 0
        self.misses = 0
        self.bytes = 0
        self.max_bytes = int(float(max_gb) * (1 << 30))

    def get(self, key, produce, device):
        v = self.store.get(key)
        if v is None:
            self.misses += 1
            packed = produce()
            cpu = tuple(t.detach().to('cpu') for t in packed)
            nbytes = sum(t.numel() * t.element_size() for t in cpu)
            if self.bytes + nbytes <= self.max_bytes:
                self.store[key] = tuple(t.clone() for t in cpu)
                self.bytes += nbytes
            return tuple(t.to(device) for t in cpu)
        self.hits += 1
        return tuple(t.to(device) for t in v)


def feat_delta_stats(delta, fr):
    a = delta.detach().float().abs()
    f = fr.detach().float().abs()
    md = float(a.mean())
    return dict(
        dFR_mean=md,
        dFR_p50=float(torch.quantile(a.reshape(-1), 0.50)),
        dFR_p90=float(torch.quantile(a.reshape(-1), 0.90)),
        dFR_max=float(a.max()),
        FR_mean=float(f.mean()),
        r_R=float(md / (float(f.mean()) + 1e-8)),
    )


def dist_to_f0(fr, fr_star, f0):
    return dict(
        d_before=float((fr - f0).abs().mean()),
        d_after=float((fr_star - f0).abs().mean()),
    )


def t_change_stats(t_raw, t_star):
    d = (t_star - t_raw).detach().float()
    a = d.abs()
    flat_a = t_raw.detach().float().reshape(1, -1)
    flat_b = t_star.detach().float().reshape(1, -1)
    cos = float(torch.nn.functional.cosine_similarity(flat_a, flat_b).item())
    return dict(
        dT_mean=float(a.mean()),
        dT_p90=float(torch.quantile(a.reshape(-1), 0.90)),
        cos_T=cos,
    )


def verdict_v41(psnr_b0: Dict, psnr_a1: Dict, safety_b0: Dict, safety_a1: Dict,
                dep_a1: Dict, pair_stats: Optional[Dict] = None) -> Dict:
    def mean(d):
        vals = [float(d[s]) for s in STATES if s in d]
        return float(np.mean(vals)) if vals else float('nan')

    b0, a1 = mean(psnr_b0), mean(psnr_a1)
    delta = a1 - b0
    g_cor = float(psnr_a1.get('correct', float('nan'))
                  - psnr_b0.get('correct', float('nan')))
    g_dark = float(psnr_a1.get('true_dark_g0.5', float('nan'))
                   - psnr_b0.get('true_dark_g0.5', float('nan')))
    g_mis = float(psnr_a1.get('mismatch', float('nan'))
                  - psnr_b0.get('mismatch', float('nan')))

    def lh(safety, st):
        return float((safety.get(st) or {}).get('large_harm_rate', float('nan')))

    lh_b0 = lh(safety_b0, 'mismatch')
    lh_a1 = lh(safety_a1, 'mismatch')
    tail_ok = bool(math.isfinite(lh_b0) and math.isfinite(lh_a1)
                   and lh_a1 <= lh_b0 + 0.02)
    tail_worse = bool(math.isfinite(lh_b0) and math.isfinite(lh_a1)
                      and lh_a1 > lh_b0 + 0.02)

    nrm = float(dep_a1.get('normal', float('nan')))
    slf = float(dep_a1.get('self', float('nan')))
    shf = float(dep_a1.get('shuffled', dep_a1.get('shuffled_target', float('nan'))))
    ground_ok = bool(
        (math.isfinite(nrm) and math.isfinite(slf) and nrm >= slf + 0.02)
        or (math.isfinite(nrm) and math.isfinite(shf) and nrm >= shf + 0.02))
    collapsed = bool(math.isfinite(nrm) and math.isfinite(slf) and abs(nrm - slf) < 0.02)
    ref_better_self = bool(math.isfinite(nrm) and math.isfinite(slf) and nrm >= slf + 0.03)
    ground_weak = bool(not ground_ok)

    correct_ok = bool(g_cor >= -0.02)
    dark_ok = bool(g_dark >= -0.02)
    mismatch_ok = bool(g_mis >= -0.02)
    mean_strong = bool(delta >= 0.05)
    nullish = bool(abs(delta) < 0.02)
    mis_better = bool(g_mis >= 0.05 or (
        math.isfinite(lh_b0) and math.isfinite(lh_a1) and (lh_b0 - lh_a1) >= 0.05))
    unsafe = bool(g_mis < -0.02 or tail_worse)

    if unsafe and (g_cor >= 0.02 or g_dark >= 0.02):
        label, nxt, meaning = (
            'V41_CASE_D_UNSAFE', 'close_ref_rectification',
            'Grounder helps correct/dark but mismatch/tail repeats B1/B3')
    elif mean_strong and correct_ok and dark_ok and mismatch_ok and tail_ok and ground_ok and ref_better_self and not collapsed:
        label, nxt, meaning = (
            'V41_CASE_A_STRONG_GO', 'V4.1b_joint_finetune',
            'Grounded FR improves frozen B0 with real reverse-grounding')
    elif (not mean_strong) and mis_better and correct_ok and dark_ok and ground_ok and not collapsed:
        label, nxt, meaning = (
            'V41_CASE_B_ROBUSTNESS_GO', 'V4.1b_joint_finetune',
            'Grounded FR improves mismatch reliability vs frozen B0')
    elif nullish and ground_weak:
        label, nxt, meaning = (
            'V41_CASE_C_NULL', 'drop_RefGrounder',
            'Grounder unused / null vs frozen B0')
    elif unsafe:
        label, nxt, meaning = (
            'V41_CASE_D_UNSAFE', 'close_ref_rectification',
            'Grounded FR hurts mismatch or tail vs frozen B0')
    else:
        label, nxt, meaning = (
            'V41_CASE_C_NULL', 'drop_RefGrounder',
            'Ground-then-transfer inconclusive vs frozen B0')

    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(B0=b0, A1=a1), delta_a1_b0=delta,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        correct_ok=correct_ok, dark_ok=dark_ok, mismatch_ok=mismatch_ok,
        tail_ok=tail_ok, ground_ok=ground_ok, collapsed=collapsed,
        ref_better_self=ref_better_self,
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        dep_normal_shuffled=float(nrm - shf) if math.isfinite(nrm) and math.isfinite(shf)
        else float('nan'),
        large_harm_mismatch=dict(B0=lh_b0, A1=lh_a1),
        ground_direction=GROUND_DIR, forward_direction=FORWARD_DIR,
    )
    if pair_stats is not None:
        out['pair_a1_minus_b0'] = pair_stats
    return out
