"""V3-A.5D0 runtime: exact multi-scale G64 pooling + evidence audit metrics.

Stage 0 / D0 only. Does not train. Reuses V3-A.5 G64 geometry and energy mask.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

import numpy as np
import torch

from v3a5_runtime import (STATES, action_optimal_target, block_energy,
                          energy_mask, prepare_geometry, target_geometry)

# ── locked evidence channel schema ──────────────────────────────────────────

MATCH_EVIDENCE_NAMES = (
    'sim_max', 'pmax', 'entropy', 'margin', 'dx', 'dy', 'disp_var',
)
PROPOSAL_FULL_NAMES = ('gate_v2', 'abs_D', 'abs_delta')
PROPOSAL_FEATURE_NAMES = ('f0_minus_t',)

# Analysis channels (order locked). entropy is reported as confidence=1-entropy.
ANALYSIS_CHANNELS = (
    # name, source group, raw key, signed direction for polarized (+1 / -1 / 0=raw)
    ('sim_max', 'match', 'sim_max', +1),
    ('pmax', 'match', 'pmax', +1),
    ('confidence_entropy', 'match', 'entropy', -1),  # score = 1 - entropy
    ('margin', 'match', 'margin', +1),
    ('dx', 'match', 'dx', 0),
    ('dy', 'match', 'dy', 0),
    ('disp_var', 'match', 'disp_var', -1),           # score = -disp_var
    ('gate_v2', 'proposal_full', 'gate_v2', +1),
    ('abs_D', 'proposal_full', 'abs_D', 0),
    ('abs_delta', 'proposal_full', 'abs_delta', 0),
    ('f0_minus_t', 'proposal_feature', 'f0_minus_t', 0),
)

# source_scale for pooling: match / f0_minus_t at H/2 => 2; full-res => 1
CHANNEL_SOURCE_SCALE = {
    'sim_max': 2, 'pmax': 2, 'confidence_entropy': 2, 'margin': 2,
    'dx': 2, 'dy': 2, 'disp_var': 2, 'f0_minus_t': 2,
    'gate_v2': 1, 'abs_D': 1, 'abs_delta': 1,
}


def evidence_schema_payload():
    return dict(
        match_names=list(MATCH_EVIDENCE_NAMES),
        proposal_full_names=list(PROPOSAL_FULL_NAMES),
        proposal_feature_names=list(PROPOSAL_FEATURE_NAMES),
        analysis_channels=[
            dict(name=n, group=g, raw_key=k, direction=d)
            for n, g, k, d in ANALYSIS_CHANNELS
        ],
        source_scale=dict(CHANNEL_SOURCE_SCALE),
        notes='confidence_entropy := 1 - entropy; disp_var polarity for '
              'polarized ROC uses -disp_var',
    )


def evidence_schema_sha():
    blob = json.dumps(evidence_schema_payload(), sort_keys=True,
                      separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


# ── exact spatial pooling onto G64 pixel edges ──────────────────────────────

@lru_cache(maxsize=64)
def _source_to_block_index(full_h, full_w, source_scale, ey, ex):
    """Map every source cell -> contribution weights into G64 blocks.

    Returns (src_h, src_w, nby, nbx, flat_src_ids [n_pix], flat_blk_ids [n_pix],
             flat_weights [n_pix]) where each full-res pixel contributes weight 1
             to exactly one block via its owning source cell.
    """
    s = int(source_scale)
    if full_h % s or full_w % s:
        raise SystemExit('full resolution %dx%d not divisible by source_scale=%d'
                         % (full_h, full_w, s))
    src_h, src_w = full_h // s, full_w // s
    ey = [int(v) for v in ey]
    ex = [int(v) for v in ex]
    nby, nbx = len(ey) - 1, len(ex) - 1
    # For each full pixel: block id and source cell id
    # Build sparse (src_cell, block) accumulation counts via pixel loop in numpy
    # (cached once per geometry).
    src_ids = []
    blk_ids = []
    weights = []
    for bi in range(nby):
        y0, y1 = ey[bi], ey[bi + 1]
        for bj in range(nbx):
            x0, x1 = ex[bj], ex[bj + 1]
            bid = bi * nbx + bj
            # pixels in this block
            ys = np.arange(y0, y1, dtype=np.int64)
            xs = np.arange(x0, x1, dtype=np.int64)
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            sy = yy // s
            sx = xx // s
            sc = (sy * src_w + sx).reshape(-1)
            # count per source cell inside this block
            uniq, cnt = np.unique(sc, return_counts=True)
            src_ids.append(uniq)
            blk_ids.append(np.full(uniq.shape, bid, dtype=np.int64))
            weights.append(cnt.astype(np.float64))
    src_ids = np.concatenate(src_ids)
    blk_ids = np.concatenate(blk_ids)
    weights = np.concatenate(weights)
    # block pixel counts (for mean)
    blk_counts = np.zeros(nby * nbx, dtype=np.float64)
    for bi in range(nby):
        for bj in range(nbx):
            bid = bi * nbx + bj
            blk_counts[bid] = float((ey[bi + 1] - ey[bi]) * (ex[bj + 1] - ex[bj]))
    if blk_counts.min() < 1:
        raise SystemExit('G64 geometry has an empty block')
    return (src_h, src_w, nby, nbx,
            src_ids, blk_ids, weights, blk_counts)


def pool_spatial_map_to_geom(amap, source_scale, geom):
    """Exact non-overlap block MEAN of a spatial map onto G64 pixel edges.

    ``amap``: [B,C,H_s,W_s] where H_s = full_H / source_scale.
    ``source_scale``: 1 (full) or 2 (H/2). Forbidden: adaptive / bilinear to
    100x150 then pool — this function maps source cells onto full-pixel
    support then averages with geom['edges'].
    """
    s = int(source_scale)
    if s not in (1, 2):
        raise SystemExit('source_scale must be 1 or 2, got %r' % source_scale)
    ey, ex = geom['edges']
    full_h = int(ey[-1])
    full_w = int(ex[-1])
    src_h, src_w, nby, nbx, src_ids, blk_ids, weights, blk_counts = \
        _source_to_block_index(full_h, full_w, s,
                               tuple(int(v) for v in ey),
                               tuple(int(v) for v in ex))
    if tuple(amap.shape[-2:]) != (src_h, src_w):
        raise SystemExit('pool_spatial_map_to_geom expected %dx%d (scale=%d), got %s'
                         % (src_h, src_w, s, tuple(amap.shape[-2:])))
    b, c = amap.shape[0], amap.shape[1]
    flat = amap.reshape(b, c, -1)  # [B,C,src_h*src_w]
    device = amap.device
    dtype = amap.dtype
    src_t = torch.as_tensor(src_ids, dtype=torch.long, device=device)
    blk_t = torch.as_tensor(blk_ids, dtype=torch.long, device=device)
    w_t = torch.as_tensor(weights, dtype=dtype, device=device)
    # gather source values at contributing cells: [B,C,N]
    vals = flat.index_select(2, src_t)
    weighted = vals * w_t.view(1, 1, -1)
    acc = amap.new_zeros((b, c, nby * nbx))
    acc.index_add_(2, blk_t, weighted)
    cnt = torch.as_tensor(blk_counts, dtype=dtype, device=device)
    return (acc / cnt.clamp(min=1.0)).reshape(b, c, nby, nbx)


def pool_spatial_map_to_geom_python(amap_np, source_scale, geom):
    """Slow ground-truth block mean for tests. amap_np: [H_s,W_s] or [C,H,W]."""
    s = int(source_scale)
    ey, ex = [int(v) for v in geom['edges'][0]], [int(v) for v in geom['edges'][1]]
    full_h, full_w = ey[-1], ex[-1]
    nby, nbx = len(ey) - 1, len(ex) - 1
    x = np.asarray(amap_np, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    c, sh, sw = x.shape
    if (sh, sw) != (full_h // s, full_w // s):
        raise SystemExit('python GT shape mismatch')
    out = np.zeros((c, nby, nbx), dtype=np.float64)
    for bi in range(nby):
        for bj in range(nbx):
            ys = np.arange(ey[bi], ey[bi + 1])
            xs = np.arange(ex[bj], ex[bj + 1])
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            cells = x[:, yy // s, xx // s]
            out[:, bi, bj] = cells.reshape(c, -1).mean(axis=1)
    return out


def pool_geometry_coverage(source_scale, geom):
    """Audit: every full pixel covered once; no empty G64 block."""
    s = int(source_scale)
    ey, ex = geom['edges']
    full_h, full_w = int(ey[-1]), int(ex[-1])
    src_h, src_w, nby, nbx, src_ids, blk_ids, weights, blk_counts = \
        _source_to_block_index(full_h, full_w, s,
                               tuple(int(v) for v in ey),
                               tuple(int(v) for v in ex))
    # reconstruct coverage: sum of weights over blocks must equal full pixels
    total = float(weights.sum())
    return dict(
        source_scale=s, source_shape=[src_h, src_w],
        full_shape=[full_h, full_w], g64_shape=[nby, nbx],
        n_blocks=int(nby * nbx),
        min_block_pixels=int(blk_counts.min()),
        max_block_pixels=int(blk_counts.max()),
        total_pixel_weight=total,
        all_pixels_covered_once=bool(abs(total - full_h * full_w) < 1e-9),
        no_empty_block=bool(blk_counts.min() >= 1),
    )


# ── extract analysis tensors from probe evidence ────────────────────────────

def analysis_maps_from_evidence(evidence):
    """-> dict name -> [B,1,H_s,W_s] raw analysis map (pre-pool).

    confidence_entropy is already 1-entropy; disp_var / abs_* stay raw.
    """
    match = evidence['match']
    full = evidence['proposal']['full']
    feat = evidence['proposal']['feature']
    out = {}
    for name, group, key, direction in ANALYSIS_CHANNELS:
        if group == 'match':
            raw = match[key]
            if name == 'confidence_entropy':
                out[name] = 1.0 - raw
            else:
                out[name] = raw
        elif group == 'proposal_full':
            out[name] = full[key]
        else:
            out[name] = feat[key]
    return out


def pool_all_evidence_to_g64(evidence, geom):
    """-> dict name -> [B,1,64,64] G64 block means."""
    maps = analysis_maps_from_evidence(evidence)
    out = {}
    for name, tensor in maps.items():
        out[name] = pool_spatial_map_to_geom(
            tensor, CHANNEL_SOURCE_SCALE[name], geom)
    return out


# ── statistics ──────────────────────────────────────────────────────────────

def _finite_pair(x, y, mask=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        m = m & np.asarray(mask, dtype=bool).reshape(-1)
    return x[m], y[m]


def pearson_corr(x, y, mask=None):
    a, b = _finite_pair(x, y, mask)
    if a.size < 3:
        return float('nan')
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(x, y, mask=None):
    from scipy.stats import spearmanr
    a, b = _finite_pair(x, y, mask)
    if a.size < 3:
        return float('nan')
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    rho, _ = spearmanr(a, b)
    return float(rho)


def roc_auc_binary(scores, labels):
    """ROC-AUC for binary labels {0,1}. NaN if degenerate."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + 1 + j + 1)
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc_binary(scores, labels):
    """Average precision (PR-AUC) for binary labels."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n_pos = int((y == 1).sum())
    if n_pos == 0 or y.size == 0:
        return float('nan')
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(n_pos)
    # add sentinel
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def polarized_labels(q, reject_thr=0.05, accept_thr=0.95):
    """-> (mask_keep, labels) where labels: 1=accept, 0=reject; drop middle."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    keep = (q <= reject_thr) | (q >= accept_thr)
    labels = (q >= accept_thr).astype(np.int64)
    return keep, labels


def polarized_score(ev_values, direction):
    """Apply locked polarity. direction 0 => raw (no flip for AUC reporting)."""
    v = np.asarray(ev_values, dtype=np.float64)
    if direction > 0:
        return v
    if direction < 0:
        return -v
    return v  # raw association; still scored as-is for AUC


def decile_analysis(ev, q, n_bins=10):
    """10 equal-count bins by evidence; report mean q*, accept/reject frac."""
    ev = np.asarray(ev, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    m = np.isfinite(ev) & np.isfinite(q)
    ev, q = ev[m], q[m]
    if ev.size < n_bins * 5:
        return []
    # equal-count edges
    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(ev, qs)
    # ensure strictly increasing edges for digitize
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    bins = np.digitize(ev, edges[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        sel = bins == b
        if not sel.any():
            rows.append(dict(bin=b, n=0, mean_q=float('nan'),
                             accept_frac=float('nan'), reject_frac=float('nan'),
                             mean_ev=float('nan')))
            continue
        qb = q[sel]
        rows.append(dict(
            bin=int(b), n=int(sel.sum()),
            mean_ev=float(ev[sel].mean()),
            mean_q=float(qb.mean()),
            accept_frac=float((qb >= 0.95).mean()),
            reject_frac=float((qb <= 0.05).mean()),
        ))
    return rows


def decile_monotonic_score(rows):
    """+1 if mean_q rises with bin, -1 if falls, 0 if unstable/flat."""
    means = [r['mean_q'] for r in rows if r['n'] > 0 and math.isfinite(r['mean_q'])]
    if len(means) < 5:
        return 0
    # Spearman of bin index vs mean_q
    rho = spearman_corr(np.arange(len(means)), np.asarray(means))
    if not math.isfinite(rho):
        return 0
    if rho >= 0.6:
        return +1
    if rho <= -0.6:
        return -1
    return 0


# ── D0 verdict (plan §21–§23) ───────────────────────────────────────────────

def verdict_d0(per_channel_stats):
    """Judge D0 from aggregated per-channel stats.

    ``per_channel_stats``: dict channel -> {
        'spearman': {(split, state): float, ...},
        'roc_auc':  {(split, state): float, ...},
        'decile_dir': {(split,): +1/-1/0, ...},  # typically train/dev overall
    }
    Keys may also be 'train|correct' string form.
    """
    strong = False
    pass_a = False
    pass_b = False
    pass_c = False
    reasons = []
    fail_all_weak = True

    for ch, st in per_channel_stats.items():
        spears = st.get('spearman', {})
        aucs = st.get('roc_auc', {})
        # A: |Spearman|>=0.30 and train/dev same sign (any matching state pair)
        for state in STATES:
            tr = spears.get(('train', state), spears.get('train|%s' % state))
            dv = spears.get(('dev', state), spears.get('dev|%s' % state))
            if tr is None or dv is None:
                continue
            if not (math.isfinite(tr) and math.isfinite(dv)):
                continue
            if abs(tr) >= 0.20 or abs(dv) >= 0.20:
                fail_all_weak = False
            if abs(tr) >= 0.30 and abs(dv) >= 0.30 and (tr * dv) > 0:
                pass_a = True
                reasons.append('A:%s|%s spearman train=%.3f dev=%.3f' % (ch, state, tr, dv))
            if abs(tr) >= 0.45 or abs(dv) >= 0.45:
                strong = True
                reasons.append('strong_spearman:%s|%s' % (ch, state))

        # B: polarized ROC-AUC>=0.65 on >=2/3 states (prefer overall-per-state
        # using train+dev pooled under key ('*', state) or check train & dev)
        # Plan: "至少 2/3 states 成立" — use per-state AUC with train|state
        # and require the state to pass on both splits OR on a pooled key.
        state_hits = 0
        for state in STATES:
            vals = []
            for split in ('train', 'dev'):
                v = aucs.get((split, state), aucs.get('%s|%s' % (split, state)))
                if v is not None and math.isfinite(v):
                    vals.append(v)
                    if v >= 0.60:
                        fail_all_weak = False
                    if v >= 0.75:
                        strong = True
                        reasons.append('strong_auc:%s|%s|%s=%.3f' % (ch, split, state, v))
            if vals and min(vals) >= 0.65:
                state_hits += 1
        if state_hits >= 2:
            pass_b = True
            reasons.append('B:%s state_hits=%d/3' % (ch, state_hits))

        # C: decile monotonic same direction train & dev
        d_tr = st.get('decile_dir', {}).get('train', st.get('decile_dir', {}).get(('train',)))
        d_dv = st.get('decile_dir', {}).get('dev', st.get('decile_dir', {}).get(('dev',)))
        if d_tr and d_dv and d_tr == d_dv and d_tr != 0:
            pass_c = True
            fail_all_weak = False
            reasons.append('C:%s decile_dir=%+d' % (ch, d_tr))

    passed = bool(pass_a or pass_b or pass_c)
    if fail_all_weak and not passed:
        verdict = 'FAIL_CLOSE_EVIDENCE_ROUTE'
        next_step = 'D2'
    elif passed and strong:
        verdict = 'STRONG_PASS'
        next_step = 'D1'
    elif passed:
        verdict = 'PASS'
        next_step = 'D1'
    else:
        verdict = 'INCONCLUSIVE'
        next_step = 'REVIEW'
        # not clear fail AND not clear pass — still do not auto-train
    return dict(
        verdict=verdict,
        passed=passed,
        strong=strong,
        pass_A=pass_a,
        pass_B=pass_b,
        pass_C=pass_c,
        fail_close_route=verdict == 'FAIL_CLOSE_EVIDENCE_ROUTE',
        next_step=next_step,
        reasons=reasons,
        schema_sha=evidence_schema_sha(),
    )


__all__ = [
    'ANALYSIS_CHANNELS', 'CHANNEL_SOURCE_SCALE', 'MATCH_EVIDENCE_NAMES',
    'PROPOSAL_FEATURE_NAMES', 'PROPOSAL_FULL_NAMES', 'STATES',
    'action_optimal_target', 'analysis_maps_from_evidence', 'block_energy',
    'decile_analysis', 'decile_monotonic_score', 'energy_mask',
    'evidence_schema_payload', 'evidence_schema_sha', 'pearson_corr',
    'pool_all_evidence_to_g64', 'pool_geometry_coverage',
    'pool_spatial_map_to_geom', 'pool_spatial_map_to_geom_python',
    'polarized_labels', 'polarized_score', 'pr_auc_binary',
    'prepare_geometry', 'roc_auc_binary', 'spearman_corr',
    'target_geometry', 'verdict_d0',
]
