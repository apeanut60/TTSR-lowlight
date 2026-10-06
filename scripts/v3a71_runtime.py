"""V3-A.7.1 Utility Predictability Audit — math, probes, regret, verdict."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import torch

from v3a5_runtime import block_mean, expand_gate
from v3a5e_runtime import mae, pearson, r2_score, spearman
from v3a7_runtime import block_utility

FORMAL_LOCK_KEYS = (
    'repo_commit', 'proposal_sha256', 'cache_metadata_sha256', 'split_sha256',
    'mismatch_train_sha256', 'mismatch_dev_sha256', 'energy_stats_sha256',
    'reference_variant', 'geometry', 'architecture', 'bottleneck', 'init_sha',
    'updates', 'grad_accum', 'lr', 'seed', 'pair_schedule_seed', 'states',
    'mask_mode', 'official_test_allowed',
)


def image_utility(y0, H, D):
    """Scalar U_I = MSE(Y0,H) - MSE(Y0+D,H) over RGB×pixels."""
    e0 = (y0 - H).pow(2).mean()
    e1 = (y0 + D - H).pow(2).mean()
    return e0 - e1


def mse_image(a, b):
    return (a - b).pow(2).mean()


def decision_regret(mse_pred, mse_base, mse_r1):
    return float(mse_pred - min(float(mse_base), float(mse_r1)))


def apply_binary_image_q(y0, D, q01):
    return y0 + float(q01) * D


def ranked_bins(u_pred, u_true, fractions=(0.10, 0.25, 0.50)):
    """High predicted-utility coverage vs realized U."""
    u_pred = np.asarray(u_pred, dtype=np.float64).reshape(-1)
    u_true = np.asarray(u_true, dtype=np.float64).reshape(-1)
    n = u_pred.size
    order = np.argsort(-u_pred)
    out = {}
    for f in fractions:
        k = max(1, int(round(f * n)))
        sel = order[:k]
        ut = u_true[sel]
        out['top_%d' % int(round(f * 100))] = dict(
            n=int(k), mean_u=float(ut.mean()),
            pos_rate=float((ut > 0).mean()),
            median_u=float(np.median(ut)))
    k_lo = max(1, n // 2)
    ut = u_true[order[-k_lo:]]
    out['bottom_50'] = dict(
        n=int(k_lo), mean_u=float(ut.mean()),
        pos_rate=float((ut > 0).mean()),
        median_u=float(np.median(ut)))
    return out


def sign_metrics(u_true, score):
    """Diagnostic classification of 1[U>0] from a real score (e.g. U_pred)."""
    y = (np.asarray(u_true, dtype=np.float64).reshape(-1) > 0).astype(np.int32)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    m = np.isfinite(s) & np.isfinite(y)
    y, s = y[m], s[m]
    if y.size < 10 or y.min() == y.max():
        return dict(auroc=float('nan'), auprc=float('nan'),
                    balanced_acc=float('nan'), brier=float('nan'), n=int(y.size))
    pos, neg = s[y == 1], s[y == 0]
    n_pos, n_neg = int(pos.size), int(neg.size)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty_like(s, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sum_ranks = np.bincount(inv, weights=ranks)
        ranks = (sum_ranks / cnt)[inv]
    auroc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    order_desc = np.argsort(-s, kind='mergesort')
    yt = y[order_desc]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / float(n_pos)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    auprc = float(np.sum((rec - rec_prev) * prec))
    zs = (s - s.mean()) / (s.std() + 1e-8)
    p = 1.0 / (1.0 + np.exp(-zs))
    brier = float(np.mean((p - y) ** 2))
    pred = (s > 0).astype(np.int32)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return dict(auroc=auroc, auprc=auprc, brier=brier,
                balanced_acc=float(0.5 * (tpr + tnr)), n=int(y.size))


def pack_reg(y, yp):
    return dict(
        r2=r2_score(y, yp), pearson=pearson(y, yp), spearman=spearman(y, yp),
        mae=mae(y, yp), n=int(np.asarray(y).size))


def bins_monotonic(bins: Dict) -> bool:
    """True if top10 mean U > top25 > top50 > bottom50 (strict-ish)."""
    keys = ('top_10', 'top_25', 'top_50', 'bottom_50')
    if any(k not in bins for k in keys):
        return False
    vals = [bins[k]['mean_u'] for k in keys]
    return all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def verdict_v3a71(table: Dict, regret: Dict) -> Dict:
    """Case A/B/C/D from plan §12. table rows keyed like block_F2 / image_F2."""
    def _row(level, feat):
        return table.get('%s_%s' % (level, feat), {})

    img = _row('image', 'F2')
    blk = _row('image', 'F2')  # placeholder
    blk = _row('block', 'F2')
    img_c = float(img.get('dev_corr', float('nan')))
    blk_c = float(blk.get('dev_corr', float('nan')))
    img_auc = float(img.get('auroc', float('nan')))
    blk_auc = float(blk.get('auroc', float('nan')))
    img_bins = img.get('bins') or {}
    blk_bins = blk.get('bins') or {}
    img_reg = float(regret.get('image_F2', {}).get('mean_regret', float('nan')))
    blk_reg = float(regret.get('block_F2', {}).get('mean_regret', float('nan')))
    gq_reg = float(regret.get('global_constant', {}).get('mean_regret', float('nan')))

    img_ok = bool(math.isfinite(img_c) and img_c >= 0.45
                  and (not math.isfinite(img_auc) or img_auc >= 0.70))
    # Case B needs transferable magnitude; monotone bins alone is Case C.
    blk_ok = bool(math.isfinite(blk_c) and blk_c >= 0.40)
    rank_ok = bins_monotonic(img_bins) or bins_monotonic(blk_bins)
    auc_dead = (
        (not math.isfinite(img_auc) or img_auc < 0.65)
        and (not math.isfinite(blk_auc) or blk_auc < 0.65))
    both_fail = bool(
        (not math.isfinite(blk_c) or blk_c < 0.25)
        and (not math.isfinite(img_c) or img_c < 0.25)
        and auc_dead
        and not rank_ok)

    regret_better = bool(math.isfinite(img_reg) and math.isfinite(gq_reg)
                         and img_reg < gq_reg - 1e-8)

    if img_ok and (not math.isfinite(blk_c) or blk_c < img_c - 0.05) and (
            regret_better or not math.isfinite(gq_reg)):
        label, nxt, meaning = (
            'V3A71_CASE_A_IMAGE_TRANSFERABLE', 'V3A8_global_selective_trust',
            'image/global utility is the learnable signal; pause regional')
    elif blk_ok and (not math.isfinite(img_c) or blk_c >= img_c):
        label, nxt, meaning = (
            'V3A71_CASE_B_BLOCK_TRANSFERABLE', 'regional_expected_utility',
            'block utility magnitude more transferable than q*')
    elif rank_ok and not img_ok and not blk_ok:
        label, nxt, meaning = (
            'V3A71_CASE_C_RANKING_COVERAGE', 'selective_high_confidence_coverage',
            'overall corr weak; top utility bins still reliable')
    elif both_fail:
        label, nxt, meaning = (
            'V3A71_CASE_D_BOTH_FAIL', 'revisit_proposal_or_implicit_fusion',
            'X/Y0/R cannot forecast proposal consequence at block or image')
    else:
        label, nxt, meaning = (
            'V3A71_CASE_MIXED', 'REVIEW',
            'partial utility signal; inspect table before GO')

    return dict(
        label=label, next_step=nxt, meaning=meaning,
        image_F2_dev_corr=img_c, block_F2_dev_corr=blk_c,
        image_F2_auroc=img_auc, block_F2_auroc=blk_auc,
        image_F2_mean_regret=img_reg, block_F2_mean_regret=blk_reg,
        image_bins_monotonic=bins_monotonic(img_bins),
        block_bins_monotonic=bins_monotonic(blk_bins),
    )


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def utility_target_stats(u, states=None):
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return dict(n=0, mean=float('nan'), std=float('nan'), pos_frac=float('nan'),
                    p10=float('nan'), median=float('nan'), p90=float('nan'))
    return dict(
        n=int(u.size), mean=float(u.mean()), std=float(u.std()),
        pos_frac=float((u > 0).mean()),
        p10=float(np.percentile(u, 10)),
        median=float(np.median(u)),
        p90=float(np.percentile(u, 90)),
    )
