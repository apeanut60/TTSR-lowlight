"""V3-A.5E1 — Visual-feature (Z) predictability helpers."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from v3a5e_runtime import mae, pearson, r2_score, spearman


def fit_mlp_torch(X, y, hidden=128, steps=1500, lr=3e-3, seed=0,
                  batch=4096, device='cpu', l2=1e-4):
    """Adam MLP probe on standardized X; returns cpu numpy-friendly dict."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    Xs = (X - mu) / sd
    N, F = Xs.shape
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(F, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.from_numpy(Xs).to(device)
    yt = torch.from_numpy(y).to(device)
    net.train()
    for t in range(steps):
        idx = rng.choice(N, size=min(batch, N), replace=False)
        pred = net(Xt[idx]).squeeze(-1)
        loss = ((pred - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (t + 1) % 500 == 0:
            for g in opt.param_groups:
                g['lr'] *= 0.5
    net.eval()
    return dict(net=net, mu=mu, sd=sd, device=device)


def predict_mlp_torch(params, X):
    X = np.asarray(X, dtype=np.float32)
    Xs = (X - params['mu']) / params['sd']
    with torch.no_grad():
        t = torch.from_numpy(Xs).to(params['device'])
        out = params['net'](t).squeeze(-1).float().cpu().numpy()
    return out


def nn_same_vs_cross(X, y, img_ids, k=5, max_q=8000, seed=0):
    """Block kNN: same-image vs cross-image leave-one-out.

    Returns spearman/pearson/mean_abs for each regime on a query subsample.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    img_ids = np.asarray(img_ids)
    N = X.shape[0]
    if N < k + 2:
        nan = float('nan')
        empty = dict(mean_abs=nan, median_abs=nan, spearman=nan, pearson=nan, n=0)
        return dict(same=empty, cross=empty, unrestricted=empty)

    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    Xs = (X - mu) / sd

    # ensure queries have enough same-image neighbors
    counts = {}
    for i, g in enumerate(img_ids):
        counts[g] = counts.get(g, 0) + 1
    eligible = np.array([i for i in range(N) if counts[img_ids[i]] >= k + 1])
    if eligible.size == 0:
        nan = float('nan')
        empty = dict(mean_abs=nan, median_abs=nan, spearman=nan, pearson=nan, n=0)
        return dict(same=empty, cross=empty, unrestricted=empty)
    q = eligible if eligible.size <= max_q else rng.choice(
        eligible, size=max_q, replace=False)

    def _pack(amb, y_self, y_nn):
        return dict(
            mean_abs=float(np.mean(amb)),
            median_abs=float(np.median(amb)),
            spearman=spearman(y_self, y_nn),
            pearson=pearson(y_self, y_nn),
            n=int(len(amb)),
            k=int(k),
        )

    same_a, same_ys, same_ynn = [], [], []
    cross_a, cross_ys, cross_ynn = [], [], []
    any_a, any_ys, any_ynn = [], [], []

    # pre-group indices by image for same-image search
    by_img = {}
    for i, g in enumerate(img_ids):
        by_img.setdefault(g, []).append(i)
    by_img = {g: np.asarray(ix, dtype=np.int64) for g, ix in by_img.items()}

    for i in q:
        gi = img_ids[i]
        # same-image
        pool = by_img[gi]
        if pool.size > 1:
            d = np.sqrt(((Xs[pool] - Xs[i]) ** 2).sum(axis=1))
            d[pool == i] = np.inf
            nn = pool[np.argpartition(d, k)[:k]]
            pred = float(y[nn].mean())
            same_a.append(abs(float(y[i]) - pred))
            same_ys.append(float(y[i]))
            same_ynn.append(pred)
        # cross-image: sample pool to keep O(max_q * pool) bounded
        # use up to 20k others
        others = np.flatnonzero(img_ids != gi)
        if others.size >= k:
            if others.size > 20000:
                others = rng.choice(others, size=20000, replace=False)
            d = np.sqrt(((Xs[others] - Xs[i]) ** 2).sum(axis=1))
            nn = others[np.argpartition(d, k)[:k]]
            pred = float(y[nn].mean())
            cross_a.append(abs(float(y[i]) - pred))
            cross_ys.append(float(y[i]))
            cross_ynn.append(pred)
        # unrestricted (leave-one-out on global subsample for reference)
        # reuse cross pool + same excluding self via full eligible sample
        # cheap: distance on a 15k random pool
        pool_u = rng.choice(N, size=min(15000, N), replace=False)
        if i not in pool_u:
            pool_u = np.concatenate([pool_u, [i]])
        d = np.sqrt(((Xs[pool_u] - Xs[i]) ** 2).sum(axis=1))
        d[pool_u == i] = np.inf
        nn = pool_u[np.argpartition(d, k)[:k]]
        pred = float(y[nn].mean())
        any_a.append(abs(float(y[i]) - pred))
        any_ys.append(float(y[i]))
        any_ynn.append(pred)

    return dict(
        same=_pack(same_a, same_ys, same_ynn) if same_a else
        dict(mean_abs=float('nan'), median_abs=float('nan'),
             spearman=float('nan'), pearson=float('nan'), n=0),
        cross=_pack(cross_a, cross_ys, cross_ynn) if cross_a else
        dict(mean_abs=float('nan'), median_abs=float('nan'),
             spearman=float('nan'), pearson=float('nan'), n=0),
        unrestricted=_pack(any_a, any_ys, any_ynn) if any_a else
        dict(mean_abs=float('nan'), median_abs=float('nan'),
             spearman=float('nan'), pearson=float('nan'), n=0),
    )


def verdict_e1(probe: Dict, nn: Dict) -> Dict:
    """Interpret Z predictability.

    E1_Z_PREDICTABLE / E1_Z_HEAD_GAP: Z carries transferable q* → not target death
    E1_Z_IDENTITY: same-image NN strong, cross/dev weak
    E1_Z_UNPREDICTABLE: even Z barely predicts in-sample → redefine_target
    """
    tr_r2 = float(probe.get('train_r2', float('nan')))
    dv_r2 = float(probe.get('dev_r2', float('nan')))
    tr_corr = float(probe.get('train_corr', float('nan')))
    dv_corr = float(probe.get('dev_corr', float('nan')))
    same_sp = float(nn.get('same', {}).get('spearman', float('nan')))
    cross_sp = float(nn.get('cross', {}).get('spearman', float('nan')))

    head_gap = bool(tr_corr >= 0.50 and dv_corr >= 0.50)
    predictable = bool(
        (tr_r2 >= 0.25 or tr_corr >= 0.40) and (dv_r2 >= 0.20 or dv_corr >= 0.35))
    # Prefer same≫cross NN as the identity signal (user-requested test).
    identity = bool(
        math.isfinite(same_sp) and same_sp >= 0.50
        and math.isfinite(cross_sp) and cross_sp < 0.25
        and dv_corr < 0.35
        and (same_sp - cross_sp) >= 0.35)
    # fallback: probe train/dev gap without NN
    if not identity:
        identity = bool(
            (tr_corr >= 0.40 or (math.isfinite(same_sp) and same_sp >= 0.40))
            and (dv_corr < 0.25)
            and (not math.isfinite(cross_sp) or cross_sp < 0.25))
    unpredictable = bool(
        tr_r2 < 0.15 and tr_corr < 0.25
        and (not math.isfinite(cross_sp) or cross_sp < 0.20)
        and (not math.isfinite(same_sp) or same_sp < 0.30))

    if head_gap:
        label = 'E1_Z_HEAD_GAP'
        next_step = 'fix_head_or_optimization'
        meaning = 'Z carries strong transferable q* signal; head/opt unlikely to be reading it'
    elif predictable:
        label = 'E1_Z_PREDICTABLE'
        next_step = 'retrain_verifier_with_budget'
        meaning = 'MultiScale Z shares a transferable map to q*; under-training more likely'
    elif identity:
        label = 'E1_Z_IDENTITY'
        next_step = 'redefine_target'
        meaning = 'Z predicts q* within image / train but not across images'
    elif unpredictable:
        label = 'E1_Z_UNPREDICTABLE'
        next_step = 'redefine_target'
        meaning = 'even verifier-visible Z barely predicts q* in-sample or across images'
    else:
        label = 'E1_Z_MIXED'
        next_step = 'REVIEW'
        meaning = 'partial Z signal; prefer redesign unless stronger probe appears'

    return dict(
        label=label, next_step=next_step, meaning=meaning,
        head_gap=head_gap, predictable=predictable,
        identity_heavy=identity, unpredictable=unpredictable,
        block_probe=dict(train_r2=tr_r2, dev_r2=dv_r2,
                         train_corr=tr_corr, dev_corr=dv_corr),
        nn_same_spearman=same_sp, nn_cross_spearman=cross_sp,
    )
