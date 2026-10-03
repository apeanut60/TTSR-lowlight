"""V3-A.7.2 Selective Coverage Closure — mask, oracle, thresholds, verdict.

Zero new training. Train-calibrated coverage only. Official Test forbidden.
"""

from __future__ import annotations

import inspect
import math
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from v3a5_runtime import STATES, block_energy, expand_gate
from v3a5d_runtime import CHANNEL_SOURCE_SCALE, pool_spatial_map_to_geom
from v3a5e_runtime import FEATURE_NAMES
from v3a71_runtime import json_ready

COVERAGES = (0.05, 0.10, 0.20, 0.30)
QUALITY_COVERAGES = (0.05, 0.10, 0.20, 0.30, 0.50)
PRIMARY_COVERAGE = 0.10
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
FORMAL_LOCK_KEYS = (
    'repo_commit', 'proposal_sha256', 'split_sha256',
    'mismatch_train_sha256', 'mismatch_dev_sha256', 'energy_stats_sha256',
    'reference_variant', 'geometry',
    'f0_source', 'f2_ckpt', 'f2_ckpt_sha256', 'f2_ckpt_step', 'f2_ckpt_arm',
    'f2_source_stage', 'per_image', 'bootstrap_B', 'coverage_grid', 'seed',
    'official_test_allowed',
)


def extract_z(model, X, Y0, R, geom):
    h = model.common_features(X, Y0, R)
    if model.use_context:
        h = model.context(h)
    return model.prepare_features(h, geom)


def stack_f0(maps, D, geom):
    feats = []
    for name in FEATURE_NAMES:
        if name == 'energy':
            feats.append(block_energy(D, geom))
            continue
        raw = maps[name]
        scale = CHANNEL_SOURCE_SCALE.get(name)
        if scale is None:
            scale = 2 if name == 'confidence_entropy' else 1
        feats.append(pool_spatial_map_to_geom(raw, scale, geom))
    return torch.cat(feats, dim=1)


def apply_selective_q(score, tau, mask):
    """q = 1[score > tau] * M. Invalid-energy blocks forced to 0."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    q = ((score > float(tau)) & (mask >= 0.5)).astype(np.float64)
    q[mask < 0.5] = 0.0
    return q


def q_to_grid(q_flat, nby=64, nbx=64, device='cpu'):
    t = torch.as_tensor(q_flat.reshape(1, 1, nby, nbx), dtype=torch.float32)
    return t.to(device)


def binary_block_oracle_q(U, mask):
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    q = ((U > 0.0) & (mask >= 0.5)).astype(np.float64)
    q[mask < 0.5] = 0.0
    return q


def regional_regret(mse_policy, mse_bin_oracle):
    return float(mse_policy - mse_bin_oracle)


def calibrate_train_thresholds(train_scores, coverages=COVERAGES):
    """Percentile thresholds from TRAIN scores only. No dev argument.

    coverage c → tau = (1-c) quantile of train scores (e.g. 5% → 95th pct).
    """
    if 'dev' in inspect.signature(calibrate_train_thresholds).parameters:
        raise RuntimeError('threshold calibration must not take dev scores')
    s = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError('empty train scores')
    out = {}
    for c in coverages:
        c = float(c)
        tau = float(np.percentile(s, 100.0 * (1.0 - c)))
        out[c] = tau
    return out


def per_image_topk_q(score, mask, coverage):
    """P2: among energy-valid blocks of this map, take top coverage fraction."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    q = np.zeros_like(score, dtype=np.float64)
    valid = np.flatnonzero(mask >= 0.5)
    if valid.size == 0:
        return q
    k = max(1, int(round(float(coverage) * valid.size)))
    k = min(k, valid.size)
    order = valid[np.argsort(-score[valid])]
    q[order[:k]] = 1.0
    return q


def _bin_pack(sel, u, image_ids=None):
    ut = np.asarray(u, dtype=np.float64).reshape(-1)[sel]
    out = dict(
        n=int(sel.size if hasattr(sel, 'size') else len(sel)),
        mean_u=float(ut.mean()) if ut.size else float('nan'),
        median_u=float(np.median(ut)) if ut.size else float('nan'),
        pos_rate=float((ut > 0).mean()) if ut.size else float('nan'),
    )
    if image_ids is not None:
        imgs = np.asarray(image_ids).reshape(-1)[sel]
        out['n_images'] = int(np.unique(imgs).size) if imgs.size else 0
    return out


def ranked_bins_with_ids(score, u, image_ids=None,
                         fractions=(0.10, 0.25, 0.50)):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    n = score.size
    order = np.argsort(-score)
    out = {}
    for f in fractions:
        k = max(1, int(round(f * n)))
        sel = order[:k]
        out['top_%d' % int(round(f * 100))] = _bin_pack(sel, u, image_ids)
    k_lo = max(1, n // 2)
    out['bottom_50'] = _bin_pack(order[-k_lo:], u, image_ids)
    return out


def state_composition(score, states, frac=0.10, labels=STATES):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    states = np.asarray(states)
    k = max(1, int(round(float(frac) * score.size)))
    sel = np.argsort(-score)[:k]
    picked = states[sel]
    tot = float(k)
    out = dict(n=int(k), frac=float(frac))
    for st in labels:
        out[str(st)] = float((picked == st).mean()) if tot else float('nan')
    return out


def cluster_resample_indices(image_ids, chosen_images):
    """Repeat every block of each chosen image (cluster unit = image_id)."""
    image_ids = np.asarray(image_ids)
    buckets = {}
    for i, im in enumerate(image_ids):
        buckets.setdefault(im, []).append(i)
    chunks = []
    for im in chosen_images:
        chunks.append(np.asarray(buckets[im], dtype=np.int64))
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(chunks)


def cluster_bootstrap_bins(image_ids, scores, u, states=None, state_filter=None,
                           B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED,
                           fractions=(0.10, 0.25)):
    image_ids = np.asarray(image_ids)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    uniq = np.unique(image_ids)
    rng = np.random.default_rng(int(seed))
    keys = []
    for f in fractions:
        keys.append('top_%d_mean_u' % int(round(f * 100)))
        keys.append('top_%d_pos_rate' % int(round(f * 100)))
    keys.extend(['top10_minus_bottom50', 'top10_mean_u', 'top10_pos_rate'])
    store = {k: [] for k in ('top10_mean_u', 'top25_mean_u',
                              'top10_minus_bottom50', 'top10_pos_rate')}
    n = len(uniq)
    for _ in range(int(B)):
        chosen = rng.choice(uniq, size=n, replace=True)
        idx = cluster_resample_indices(image_ids, chosen)
        if states is not None and state_filter is not None:
            st = np.asarray(states)[idx]
            idx = idx[st == state_filter]
        if idx.size < 8:
            continue
        sc, uu = scores[idx], u[idx]
        bins = ranked_bins_with_ids(sc, uu, image_ids[idx])
        t10 = bins['top_10']['mean_u']
        t25 = bins.get('top_25', {}).get('mean_u', float('nan'))
        b50 = bins['bottom_50']['mean_u']
        store['top10_mean_u'].append(t10)
        store['top25_mean_u'].append(t25)
        store['top10_minus_bottom50'].append(float(t10 - b50))
        store['top10_pos_rate'].append(bins['top_10']['pos_rate'])
    out = {}
    for k, vals in store.items():
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            out[k] = dict(mean=float('nan'), lo=float('nan'), hi=float('nan'),
                          n=0)
            continue
        out[k] = dict(
            mean=float(a.mean()),
            lo=float(np.percentile(a, 2.5)),
            hi=float(np.percentile(a, 97.5)),
            n=int(a.size),
        )
    return out


def safety_from_deltas(delta_psnr, names=None):
    d = np.asarray(delta_psnr, dtype=np.float64).reshape(-1)
    harm = float((d < -0.02).mean()) if d.size else float('nan')
    large = float((d < -0.10).mean()) if d.size else float('nan')
    p10 = float(np.percentile(d, 10)) if d.size else float('nan')
    worst = []
    if names is not None and d.size:
        order = np.argsort(d)[: min(5, d.size)]
        nm = np.asarray(names)
        for i in order:
            worst.append(dict(name=str(nm[i]), delta_psnr=float(d[i])))
    return dict(n=int(d.size), mean=float(d.mean()) if d.size else float('nan'),
                harm_rate=harm, large_harm_rate=large, p10=p10, worst5=worst)


def coverage_quality_monotonic(mean_u_by_coverage, tol=1e-6):
    """True if mean true U of selected blocks weakly decreases with coverage."""
    items = sorted((float(c), float(v)) for c, v in mean_u_by_coverage.items()
                   if math.isfinite(float(v)))
    if len(items) < 2:
        return False
    return all(items[i][1] + tol >= items[i + 1][1] for i in range(len(items) - 1))


def correct_ranking_ok(bins_correct):
    t10 = bins_correct.get('top_10') or {}
    t50 = bins_correct.get('top_50') or {}
    b50 = bins_correct.get('bottom_50') or {}
    m10 = float(t10.get('mean_u', float('nan')))
    m50 = float(t50.get('mean_u', float('nan')))
    mb = float(b50.get('mean_u', float('nan')))
    pos = float(t10.get('pos_rate', float('nan')))
    return bool(
        math.isfinite(m10) and m10 > 0
        and math.isfinite(pos) and pos >= 0.65
        and math.isfinite(m50) and m10 > m50
        and math.isfinite(mb) and m50 > mb)


def inspect_f2_ckpt(blob, path):
    step = int(blob.get('step', -1))
    arm = blob.get('arm') or blob.get('name')
    if not arm:
        if 'A1_multiscale' in str(path):
            arm = 'A1_multiscale'
        else:
            arm = None
    ok_step = step == 20000
    ok_arm = arm == 'A1_multiscale'
    return dict(step=step, arm=arm, ok_step=ok_step, ok_arm=ok_arm,
                source_stage='V3-A.5D2.1')


def verdict_v3a72(payload: Dict) -> Dict:
    """Case A/B/C/D from plan §15. Case A uses pre-registered P1-10% only."""
    ci = payload.get('bootstrap_correct') or {}
    t10 = (ci.get('top10_mean_u') or {})
    gap = (ci.get('top10_minus_bottom50') or {})
    t10_lo = float(t10.get('lo', float('nan')))
    gap_lo = float(gap.get('lo', float('nan')))
    ci_ok = bool(math.isfinite(t10_lo) and t10_lo > 0
                 and math.isfinite(gap_lo) and gap_lo > 0)

    bins_c = payload.get('bins_correct') or {}
    rank_ok = correct_ranking_ok(bins_c)
    comp = payload.get('composition_top10') or {}
    corr_share = float(comp.get('correct', float('nan')))
    coarse_dom = bool(math.isfinite(corr_share) and corr_share >= 0.70
                      and not rank_ok)

    p1 = payload.get('p1_primary') or {}
    base = float(payload.get('psnr_base', float('nan')))
    a1 = float(payload.get('psnr_v3a6_a1', float('nan')))
    p1_psnr = float(p1.get('mean_psnr', float('nan')))
    p1_ok = bool(math.isfinite(p1_psnr) and math.isfinite(base)
                 and p1_psnr >= base + 0.05
                 and (not math.isfinite(a1) or p1_psnr >= a1 - 0.01))

    harm = payload.get('safety_p1') or {}
    r1_harm = payload.get('safety_r1') or {}
    dark = harm.get('true_dark_g0.5') or {}
    mis = harm.get('mismatch') or {}
    r1d = r1_harm.get('true_dark_g0.5') or {}
    r1m = r1_harm.get('mismatch') or {}
    mean_dark = float(dark.get('mean', float('nan')))
    mean_mis = float(mis.get('mean', float('nan')))
    lh_d = float(dark.get('large_harm_rate', float('nan')))
    lh_m = float(mis.get('large_harm_rate', float('nan')))
    r1_ld = float(r1d.get('large_harm_rate', float('nan')))
    r1_lm = float(r1m.get('large_harm_rate', float('nan')))
    harm_ok = bool(
        (not math.isfinite(lh_d) or not math.isfinite(r1_ld) or lh_d <= r1_ld + 1e-12)
        and (not math.isfinite(lh_m) or not math.isfinite(r1_lm) or lh_m <= r1_lm + 1e-12)
        and (not math.isfinite(mean_dark) or mean_dark >= -0.02)
        and (not math.isfinite(mean_mis) or mean_mis >= -0.02))

    mono = bool(payload.get('coverage_monotonic', False))
    top10_neg_other = bool(payload.get('other_state_top10_negative', False))

    if (ci_ok and rank_ok and p1_ok and harm_ok and mono
            and not top10_neg_other):
        label, nxt, meaning = (
            'V3A72_CASE_A_REGIONAL_SELECTIVE_GO', 'V3A8_selective_coverage_gate',
            'within-state ranking + train-calibrated P1 transfer')
    elif coarse_dom or (not rank_ok and math.isfinite(corr_share)
                        and corr_share >= 0.55):
        label, nxt, meaning = (
            'V3A72_CASE_B_COARSE_STATE_ONLY', 'global_coarse_trust_only',
            'overall bins driven by correct vs mismatch, not regional')
    elif rank_ok and (not ci_ok or not p1_ok):
        label, nxt, meaning = (
            'V3A72_CASE_C_EXPLORATORY_WEAK', 'NO_FORMAL_GO',
            'correct-only trend exists but CI or P1 transfer fails')
    else:
        label, nxt, meaning = (
            'V3A72_CASE_D_SELECTIVE_ROUTE_CLOSE',
            'implicit_fusion_or_proposal_redesign',
            'no deployable regional selective signal')

    return dict(
        label=label, next_step=nxt, meaning=meaning,
        ci_ok=ci_ok, rank_ok=rank_ok, p1_ok=p1_ok, harm_ok=harm_ok,
        coverage_monotonic=mono, coarse_dom=coarse_dom,
        correct_top10_ci_lo=t10_lo, correct_gap_ci_lo=gap_lo,
        p1_10_psnr=p1_psnr, composition_correct_share=corr_share,
        primary_coverage=PRIMARY_COVERAGE,
    )
