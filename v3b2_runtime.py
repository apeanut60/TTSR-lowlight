"""V3-B.2 evidence-conditioned implicit fusion — lock, cache, verdict."""

from __future__ import annotations

import inspect
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from model.V3BEvidence import EVIDENCE_NAMES
from v3a5_runtime import STATES
from v3a71_runtime import json_ready
from v3a72_runtime import safety_from_deltas
from v3b_runtime import (CKPT_STEPS, DEFAULT_UPDATES, EVAL_STEPS, GRAD_ACCUM, LR,
                         SEED, b0_loss, proposal_core)

ARMS = ('A0_b0_replay', 'A1_evidence')
ARM_A0 = 'A0_b0_replay'
ARM_A1 = 'A1_evidence'
FORMAL_LOCK_KEYS_B2 = (
    'repo_commit', 'base_cache_metadata_sha256', 'proposal_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant', 'architecture', 'arms',
    'a0_init_sha', 'a1_init_sha', 'common_weight_sha',
    'evidence_names', 'evidence_stats_sha256', 'normalization_method',
    'evidence_resolution', 'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
)


class EvidenceMapCache(object):
    """CPU cache of (F0, T, E_raw)."""

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
            f0, t, e = produce()
            f0 = f0.detach().to('cpu')
            t = t.detach().to('cpu')
            e = e.detach().to('cpu')
            nbytes = (f0.numel() + t.numel() + e.numel()) * f0.element_size()
            if self.bytes + nbytes <= self.max_bytes:
                self.store[key] = (f0.clone(), t.clone(), e.clone())
                self.bytes += nbytes
            return f0.to(device), t.to(device), e.to(device)
        self.hits += 1
        return v[0].to(device), v[1].to(device), v[2].to(device)


def assert_no_gt_in_forward(fn):
    names = set(inspect.signature(fn).parameters)
    banned = {'H', 'h', 'q_star', 'qstar', 'U', 'AO', 'q_action', 'D', 'g_v2'}
    hit = names & banned
    if hit:
        raise SystemExit('GT/action leakage in forward args: %s' % sorted(hit))


def require_ckpt_blob(blob, *, arm, step, init_sha, proposal_sha,
                      repo_commit=None, formal=True):
    """Hard-check checkpoint metadata (plan §18 C3)."""
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
    if repo_commit is not None and blob.get('repo_commit') != repo_commit:
        errs.append('repo_commit mismatch')
    if errs:
        msg = 'ckpt HARD FAIL:\n  ' + '\n  '.join(errs)
        if formal:
            raise SystemExit(msg)
        raise ValueError(msg)
    return True


def evidence_channel_shuffle(e: torch.Tensor, seed=0) -> torch.Tensor:
    """Permute evidence channels with a fixed seed (diagnostic only)."""
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    perm = torch.randperm(e.shape[1], generator=g)
    return e[:, perm]


def verdict_v3b2(psnr_a0: Dict, psnr_a1: Dict, safety_a0: Dict,
                 safety_a1: Dict, dep_a1: Dict,
                 pair_stats: Optional[Dict] = None,
                 evidence_use: Optional[Dict] = None) -> Dict:
    """Case A–D from V3-B.2 plan §14. Primary = A1 − A0."""
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
    mis_tail_drop = (math.isfinite(lh_a0_m) and math.isfinite(lh_a1_m)
                     and (lh_a0_m - lh_a1_m) >= 0.05)
    mis_mean_ok = bool(g_mis >= 0.05)
    correct_ok = bool(g_cor >= -0.02)
    dark_ok = bool(g_dark >= -0.02)

    nrm = float(dep_a1.get('normal', float('nan')))
    slf = float(dep_a1.get('self', float('nan')))
    zro = float(dep_a1.get('zero', float('nan')))
    ref_ok = bool(math.isfinite(nrm) and math.isfinite(slf) and math.isfinite(zro)
                  and nrm >= slf + 0.03 and nrm >= zro + 0.03)
    ignored = bool(math.isfinite(nrm) and math.isfinite(slf)
                   and abs(nrm - slf) < 0.02)

    mean_strong = bool(delta >= 0.05)
    nullish = bool(abs(delta) < 0.02)
    e_weak = False
    if evidence_use:
        z = evidence_use.get('zero_e_delta_mean')
        if z is not None and math.isfinite(float(z)) and abs(float(z)) < 0.005:
            e_weak = True

    safety_go = bool(
        (mis_mean_ok or mis_tail_drop)
        and correct_ok and dark_ok and ref_ok and not ignored
        and delta > -0.02)

    if mean_strong and (mis_mean_ok or mis_tail_drop) and correct_ok and dark_ok and ref_ok:
        label, nxt, meaning = (
            'V3B2_CASE_A_STRONG_GO', 'V3B3_global_stat_prior',
            'evidence beats A0 on mean with mismatch/tail gain and real T use')
    elif safety_go and not mean_strong:
        label, nxt, meaning = (
            'V3B2_CASE_B_SAFETY_GO', 'V3B3_global_stat_prior',
            'evidence improves mismatch reliability without mean collapse')
    elif (delta < 0.0) or (g_mis < -0.02 and not mis_tail_drop) or (
            lh_a1_m > lh_a0_m + 0.02 if math.isfinite(lh_a1_m) and math.isfinite(lh_a0_m) else False):
        label, nxt, meaning = (
            'V3B2_CASE_D_HARM', 'drop_evidence_start_B3_from_B0',
            'evidence hurts mean or mismatch/tail vs A0')
    elif nullish or e_weak:
        label, nxt, meaning = (
            'V3B2_CASE_C_NULL', 'drop_B2_start_B3_from_B0',
            'evidence gain null or unused; B3 from raw-T B0')
    else:
        label, nxt, meaning = (
            'V3B2_CASE_C_NULL', 'drop_B2_start_B3_from_B0',
            'evidence inconclusive vs A0')

    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(A0=a0, A1=a1), delta_a1_a0=delta,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        mismatch_tail_drop=mis_tail_drop,
        correct_ok=correct_ok, dark_ok=dark_ok, ref_ok=ref_ok,
        ignored=ignored, mean_strong=mean_strong,
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        dep_normal_zero=float(nrm - zro) if math.isfinite(nrm) and math.isfinite(zro)
        else float('nan'),
        large_harm_mismatch=dict(A0=lh_a0_m, A1=lh_a1_m),
    )
    if pair_stats is not None:
        out['pair_a1_minus_a0'] = pair_stats
    if evidence_use is not None:
        out['evidence_use'] = evidence_use
    return out


def pair_delta_stats(psnr_a1_rows, psnr_a0_rows, states=STATES) -> Dict:
    """Per image-state A1−A0 PSNR stats; focus mismatch."""
    out = {}
    for st in list(states) + ['all']:
        if st == 'all':
            d = [a - b for a, b in zip(psnr_a1_rows, psnr_a0_rows)]
        else:
            d = [a - b for (a, sa), (b, sb) in zip(psnr_a1_rows, psnr_a0_rows)
                 if sa == st and sb == st]
            # if rows are (psnr, state) tuples
        if d and isinstance(d[0], tuple):
            continue
        arr = np.asarray(d, dtype=np.float64)
        if arr.size == 0:
            out[st] = dict(mean=float('nan'), p10=float('nan'),
                           frac_neg=float('nan'), frac_pos=float('nan'), n=0)
            continue
        out[st] = dict(
            mean=float(arr.mean()),
            p10=float(np.percentile(arr, 10)),
            frac_neg=float(np.mean(arr < -0.02)),
            frac_pos=float(np.mean(arr > 0.02)),
            n=int(arr.size),
        )
    return out


def pair_delta_stats_by_state(deltas_by_state: Dict[str, Sequence[float]]) -> Dict:
    out = {}
    all_d = []
    for st in STATES:
        arr = np.asarray(list(deltas_by_state.get(st, [])), dtype=np.float64)
        all_d.append(arr)
        if arr.size == 0:
            out[st] = dict(mean=float('nan'), p10=float('nan'),
                           frac_neg=float('nan'), frac_pos=float('nan'), n=0)
        else:
            out[st] = dict(
                mean=float(arr.mean()),
                p10=float(np.percentile(arr, 10)),
                frac_neg=float(np.mean(arr < -0.02)),
                frac_pos=float(np.mean(arr > 0.02)),
                n=int(arr.size),
            )
    cat = np.concatenate([a for a in all_d if a.size]) if any(a.size for a in all_d) else np.asarray([])
    if cat.size:
        out['all'] = dict(
            mean=float(cat.mean()),
            p10=float(np.percentile(cat, 10)),
            frac_neg=float(np.mean(cat < -0.02)),
            frac_pos=float(np.mean(cat > 0.02)),
            n=int(cat.size),
        )
    return out
