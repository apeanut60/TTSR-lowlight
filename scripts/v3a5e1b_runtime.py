"""V3-A.5E1b — trained-Z NN regimes + verdict (strict image ids + spatial-far)."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional

import numpy as np

from v3a5e_runtime import pearson, spearman


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _pack(amb, y_self, y_nn, extra=None):
    if amb:
        ys = np.asarray(y_self, dtype=np.float64)
        yn = np.asarray(y_nn, dtype=np.float64)
        sign_agree = float(np.mean((yn > 0) == (ys > 0)))
    else:
        sign_agree = float('nan')
    out = dict(
        mean_abs=float(np.mean(amb)) if amb else float('nan'),
        median_abs=float(np.median(amb)) if amb else float('nan'),
        spearman=spearman(y_self, y_nn) if len(y_self) >= 5 else float('nan'),
        pearson=pearson(y_self, y_nn) if len(y_self) >= 5 else float('nan'),
        sign_agree=sign_agree,
        n=int(len(amb)),
    )
    if extra:
        out.update(extra)
    return out


def nn_regimes_e1b(X, y, image_ids, pair_ids, states, gy, gx,
                   k=5, max_q=6000, far_cheb=4, far_cheb2=8,
                   cross_pool=20000, seed=0):
    """Three NN regimes + unrestricted same-pair baseline.

    1. same_pair_far: same pair_id, chebyshev(|dy|,|dx|) >= far_cheb
    2. same_image_diff_state: same image_id, different state
    3. strict_cross_image: different image_id
    Also: same_pair_any (no spatial constraint) for comparison.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    image_ids = np.asarray(image_ids)
    pair_ids = np.asarray(pair_ids)
    states = np.asarray(states)
    gy = np.asarray(gy, dtype=np.int32)
    gx = np.asarray(gx, dtype=np.int32)
    N = X.shape[0]

    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    Xs = (X - mu) / sd

    by_pair, by_img = {}, {}
    for i in range(N):
        by_pair.setdefault(pair_ids[i], []).append(i)
        by_img.setdefault(image_ids[i], []).append(i)
    by_pair = {k: np.asarray(v, dtype=np.int64) for k, v in by_pair.items()}
    by_img = {k: np.asarray(v, dtype=np.int64) for k, v in by_img.items()}

    # queries: need ≥k far neighbors in same pair when possible; else any with pair size
    eligible = []
    for i in range(N):
        pool = by_pair[pair_ids[i]]
        if pool.size < k + 1:
            continue
        cheb = np.maximum(np.abs(gy[pool] - gy[i]), np.abs(gx[pool] - gx[i]))
        cheb[pool == i] = -1
        if (cheb >= far_cheb).sum() >= k:
            eligible.append(i)
    eligible = np.asarray(eligible, dtype=np.int64)
    if eligible.size == 0:
        # fallback: any with enough same-pair blocks
        eligible = np.asarray(
            [i for i in range(N) if by_pair[pair_ids[i]].size >= k + 1],
            dtype=np.int64)
    q = eligible if eligible.size <= max_q else rng.choice(
        eligible, size=max_q, replace=False)

    buckets = {
        'same_pair_any': ([], [], []),
        'same_pair_far_%d' % far_cheb: ([], [], []),
        'same_pair_far_%d' % far_cheb2: ([], [], []),
        'same_image_diff_state': ([], [], []),
        'strict_cross_image': ([], [], []),
    }

    for i in q:
        pi, ii, si = pair_ids[i], image_ids[i], states[i]
        # same-pair any
        pool = by_pair[pi]
        d = np.sqrt(((Xs[pool] - Xs[i]) ** 2).sum(axis=1))
        d[pool == i] = np.inf
        if np.isfinite(d).sum() >= k:
            nn = pool[np.argpartition(d, k)[:k]]
            pred = float(y[nn].mean())
            a, b, c = buckets['same_pair_any']
            a.append(abs(float(y[i]) - pred)); b.append(float(y[i])); c.append(pred)

        cheb = np.maximum(np.abs(gy[pool] - gy[i]), np.abs(gx[pool] - gx[i]))
        cheb[pool == i] = -1
        for thr in (far_cheb, far_cheb2):
            key = 'same_pair_far_%d' % thr
            mask = cheb >= thr
            if mask.sum() < k:
                continue
            sub = pool[mask]
            d2 = np.sqrt(((Xs[sub] - Xs[i]) ** 2).sum(axis=1))
            nn = sub[np.argpartition(d2, k)[:k]]
            pred = float(y[nn].mean())
            a, b, c = buckets[key]
            a.append(abs(float(y[i]) - pred)); b.append(float(y[i])); c.append(pred)

        # same image, different state
        pool_i = by_img[ii]
        st = states[pool_i]
        sub = pool_i[st != si]
        if sub.size >= k:
            d2 = np.sqrt(((Xs[sub] - Xs[i]) ** 2).sum(axis=1))
            nn = sub[np.argpartition(d2, k)[:k]]
            pred = float(y[nn].mean())
            a, b, c = buckets['same_image_diff_state']
            a.append(abs(float(y[i]) - pred)); b.append(float(y[i])); c.append(pred)

        # strict cross-image
        others = np.flatnonzero(image_ids != ii)
        if others.size >= k:
            if others.size > cross_pool:
                others = rng.choice(others, size=cross_pool, replace=False)
            d2 = np.sqrt(((Xs[others] - Xs[i]) ** 2).sum(axis=1))
            nn = others[np.argpartition(d2, k)[:k]]
            pred = float(y[nn].mean())
            a, b, c = buckets['strict_cross_image']
            a.append(abs(float(y[i]) - pred)); b.append(float(y[i])); c.append(pred)

    out = {}
    for key, (a, b, c) in buckets.items():
        out[key] = _pack(a, b, c, extra=dict(k=int(k)))
    out['meta'] = dict(
        n_query=int(len(q)), far_cheb=int(far_cheb), far_cheb2=int(far_cheb2),
        k=int(k), n_total=int(N))
    return out


def nn_dev_to_train(Xtr, ytr, Xdv, ydv, k=5, max_q=6000, pool_n=30000, seed=1):
    rng = np.random.default_rng(seed)
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xdv = np.asarray(Xdv, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64).reshape(-1)
    ydv = np.asarray(ydv, dtype=np.float64).reshape(-1)
    n_q = min(max_q, Xdv.shape[0])
    qix = rng.choice(Xdv.shape[0], size=n_q, replace=False)
    n_pool = min(pool_n, Xtr.shape[0])
    pool = rng.choice(Xtr.shape[0], size=n_pool, replace=False)
    Xp, yp = Xtr[pool], ytr[pool]
    amb, ys, ynn = [], [], []
    for i in qix:
        d = np.sqrt(((Xp - Xdv[i]) ** 2).sum(axis=1))
        nn = np.argpartition(d, k)[:k]
        pred = float(yp[nn].mean())
        amb.append(abs(float(ydv[i]) - pred))
        ys.append(float(ydv[i]))
        ynn.append(pred)
    return _pack(amb, ys, ynn, extra=dict(k=int(k), pool_n=int(n_pool)))


def verdict_e1b(probe: Dict, nn: Dict, dev_to_train: Dict) -> Dict:
    """Closure verdict on trained MultiScale Z (train64 vs dev64).

    Important: ``strict_cross_image`` is computed **inside train64**.
    After memorizing that set, within-train cross can be high without
    implying held-out transfer — use ``dev_to_train`` / ``dev_corr`` for that.
    """
    tr_corr = float(probe.get('train_corr', float('nan')))
    dv_corr = float(probe.get('dev_corr', float('nan')))
    tr_r2 = float(probe.get('train_r2', float('nan')))
    dv_r2 = float(probe.get('dev_r2', float('nan')))

    same_any = float(nn.get('same_pair_any', {}).get('spearman', float('nan')))
    same_far = float(nn.get('same_pair_far_4', {}).get('spearman', float('nan')))
    same_far8 = float(nn.get('same_pair_far_8', {}).get('spearman', float('nan')))
    same_diff = float(nn.get('same_image_diff_state', {}).get('spearman', float('nan')))
    cross_tr = float(nn.get('strict_cross_image', {}).get('spearman', float('nan')))
    d2t = float(dev_to_train.get('spearman', float('nan')))

    heldout_dead = bool(
        dv_corr < 0.25
        and (not math.isfinite(d2t) or d2t < 0.25))
    heldout_alive = bool(
        dv_corr >= 0.45
        or (math.isfinite(d2t) and d2t >= 0.45))

    # head gap: held-out Z already readable
    head_gap = bool(tr_corr >= 0.50 and heldout_alive and dv_corr >= 0.50)
    transferable = bool(heldout_alive)
    # train fitted, held-out dead (within-train cross may still be high)
    memorize = bool(tr_corr >= 0.45 and heldout_dead)
    # far still high + held-out dead → not merely local spatial continuity
    identity = bool(
        math.isfinite(same_far) and same_far >= 0.50
        and heldout_dead
        and tr_corr >= 0.40)
    spatial_only = bool(
        math.isfinite(same_any) and same_any >= 0.50
        and math.isfinite(same_far) and same_far < 0.30
        and heldout_dead)
    unpredictable = bool(
        tr_corr < 0.25 and tr_r2 < 0.15 and heldout_dead)

    if head_gap:
        label = 'E1B_Z_HEAD_GAP'
        next_step = 'fix_head_or_optimization'
        meaning = 'trained Z carries held-out q*; do NOT redefine target yet'
    elif transferable and not memorize:
        label = 'E1B_Z_TRANSFERABLE'
        next_step = 'fix_head_or_optimization'
        meaning = 'trained Z shows held-out / dev→train signal; head/opt more likely'
    elif identity:
        label = 'E1B_Z_IDENTITY_CONFIRMED'
        next_step = 'redefine_target'
        meaning = ('trained Z: same-pair-far strong; held-out probe/dev→train dead '
                   '(within-train cross may be high via set memorization)')
    elif memorize:
        label = 'E1B_Z_MEMORIZE_NOT_TRANSFER'
        next_step = 'redefine_target'
        meaning = 'trained Z fits train64 but not held-out dev64'
    elif spatial_only:
        label = 'E1B_Z_SPATIAL_LOCAL'
        next_step = 'REVIEW'
        meaning = 'same-pair signal collapses under spatial-far; not clear identity'
    elif unpredictable:
        label = 'E1B_Z_UNPREDICTABLE'
        next_step = 'redefine_target'
        meaning = 'even trained MultiScale Z barely predicts q*'
    else:
        label = 'E1B_Z_MIXED'
        next_step = 'REVIEW'
        meaning = 'partial trained-Z signal; inspect far vs held-out before redefine'

    return dict(
        label=label, next_step=next_step, meaning=meaning,
        head_gap=head_gap, transferable=transferable, memorize=memorize,
        identity_confirmed=identity, spatial_only=spatial_only,
        unpredictable=unpredictable,
        block_probe=dict(train_r2=tr_r2, dev_r2=dv_r2,
                         train_corr=tr_corr, dev_corr=dv_corr),
        nn_same_pair_any=same_any, nn_same_pair_far4=same_far,
        nn_same_pair_far8=same_far8, nn_same_image_diff_state=same_diff,
        nn_strict_cross_within_train64=cross_tr, nn_dev_to_train=d2t,
        note_cross=('strict_cross_image is within train64 only; '
                    'held-out transfer = nn_dev_to_train / dev_corr'),
    )
