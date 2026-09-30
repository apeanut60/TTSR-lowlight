"""V3-A.5E Target Predictability Audit — metrics & verdict."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


FEATURE_NAMES = (
    'sim_max', 'pmax', 'confidence_entropy', 'margin', 'disp_var',
    'gate_v2', 'abs_D', 'abs_delta', 'f0_minus_t', 'energy',
)


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float('nan')
    if a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float('nan')
    rho, _ = spearmanr(a[m], b[m])
    return float(rho)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float('nan')
    if a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if yt.size < 5:
        return float('nan')
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float('nan')
    return float(np.abs(y_true[m] - y_pred[m]).mean())


def fit_ridge(X, y, l2=1e-2):
    """Closed-form ridge with bias column. X: [N,F], y: [N]."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    N, F = X.shape
    xb = np.concatenate([X, np.ones((N, 1))], axis=1)
    reg = l2 * np.eye(F + 1)
    reg[-1, -1] = 0.0  # no penalty on bias
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return w


def predict_ridge(w, X):
    X = np.asarray(X, dtype=np.float64)
    xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return xb @ w


def fit_mlp(X, y, hidden=64, steps=800, lr=1e-2, seed=0, l2=1e-4):
    """Tiny numpy MLP (one hidden relu) for probe upper bound."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    N, F = X.shape
    # standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    Xs = (X - mu) / sd
    W1 = rng.normal(0, 0.05, size=(F, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.05, size=(hidden,))
    b2 = 0.0
    for t in range(steps):
        h = np.maximum(0.0, Xs @ W1 + b1)
        pred = h @ W2 + b2
        err = pred - y
        loss_grad = (2.0 / N) * err
        gW2 = h.T @ loss_grad + l2 * W2
        gb2 = float(loss_grad.sum())
        dh = np.outer(loss_grad, W2)
        dh = dh * (h > 0)
        gW1 = Xs.T @ dh + l2 * W1
        gb1 = dh.sum(axis=0)
        W2 -= lr * gW2
        b2 -= lr * gb2
        W1 -= lr * gW1
        b1 -= lr * gb1
        if (t + 1) % 200 == 0:
            lr *= 0.5
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, mu=mu, sd=sd)


def predict_mlp(params, X):
    Xs = (np.asarray(X, dtype=np.float64) - params['mu']) / params['sd']
    h = np.maximum(0.0, Xs @ params['W1'] + params['b1'])
    return h @ params['W2'] + params['b2']


def nn_ambiguity(X, y, k=5, max_n=20000, seed=0):
    """For each of max_n rows, |y - mean(y of k NN among other rows)|.

    Uses Euclidean distance on X. Returns mean/median ambiguity and corr(y, y_nn).
    Search pool is capped at max_n*2 so runtime stays O(max_n^2).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    N = X.shape[0]
    if N < k + 2:
        return dict(mean_abs=float('nan'), median_abs=float('nan'),
                    spearman_nn=float('nan'), n=0)
    # standardize features for distance
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    Xs = (X - mu) / sd
    # pool for NN search + query set (same pool keeps leave-one-out valid)
    pool_n = min(N, max(max_n, min(30000, N)))
    if N > pool_n:
        pool = rng.choice(N, size=pool_n, replace=False)
        Xs, y = Xs[pool], y[pool]
        N = pool_n
    idx = np.arange(N)
    if N > max_n:
        idx = rng.choice(N, size=max_n, replace=False)
    amb = []
    y_nn = []
    y_self = []
    for i in idx:
        d = np.sqrt(((Xs - Xs[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        nn = np.argpartition(d, k)[:k]
        pred = float(y[nn].mean())
        amb.append(abs(float(y[i]) - pred))
        y_nn.append(pred)
        y_self.append(float(y[i]))
    return dict(
        mean_abs=float(np.mean(amb)),
        median_abs=float(np.median(amb)),
        spearman_nn=spearman(y_self, y_nn),
        pearson_nn=pearson(y_self, y_nn),
        n=int(len(amb)),
        k=int(k),
        pool_n=int(N),
    )


def conditional_variance(x, y, n_bins=10):
    """Var(y|x) via equal-count bins of scalar x; compare to Var(y)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if y.size < n_bins * 20:
        return dict(var_y=float('nan'), mean_cond_var=float('nan'),
                    explained_frac=float('nan'))
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    bins = np.digitize(x, edges[1:-1], right=False)
    cond = []
    for b in range(n_bins):
        sel = bins == b
        if sel.sum() >= 5:
            cond.append(float(y[sel].var()))
    var_y = float(y.var())
    mean_cv = float(np.mean(cond)) if cond else float('nan')
    explained = 1.0 - mean_cv / var_y if var_y > 1e-12 and math.isfinite(mean_cv) else float('nan')
    return dict(var_y=var_y, mean_cond_var=mean_cv, explained_frac=explained,
                n_bins_used=len(cond))


def verdict_e(probe: Dict, nn: Dict, image_probe: Dict) -> Dict:
    """Interpret predictability.

    UNPREDICTABLE: train probe R² low (<0.15) and NN spearman low
    IDENTITY_ONLY: train probe/MLP high, dev low (mirrors D2.1 Case B)
    PREDICTABLE: train and dev probe both decent (R²>=0.25 or corr>=0.40)
    """
    tr_r2 = float(probe.get('train_r2', float('nan')))
    dv_r2 = float(probe.get('dev_r2', float('nan')))
    tr_corr = float(probe.get('train_corr', float('nan')))
    dv_corr = float(probe.get('dev_corr', float('nan')))
    nn_sp = float(nn.get('block_spearman_nn', float('nan')))
    img_tr = float(image_probe.get('train_corr', float('nan')))
    img_dv = float(image_probe.get('dev_corr', float('nan')))

    predictable = bool(
        (tr_r2 >= 0.25 or tr_corr >= 0.40) and (dv_r2 >= 0.20 or dv_corr >= 0.35))
    identity = bool(
        (tr_r2 >= 0.25 or tr_corr >= 0.50 or img_tr >= 0.50)
        and (dv_r2 < 0.15 and dv_corr < 0.25 and img_dv < 0.30))
    unpredictable = bool(
        tr_r2 < 0.15 and tr_corr < 0.25 and (not math.isfinite(nn_sp) or nn_sp < 0.20))

    if predictable:
        label = 'E_PREDICTABLE'
        next_step = 'retrain_verifier_with_budget'
        meaning = 'observables share a transferable map to q*; under-training more likely'
    elif identity:
        label = 'E_IDENTITY_HEAVY'
        next_step = 'redefine_target'
        meaning = 'q* fit on train observables does not transfer — target tied to identity/GT-only'
    elif unpredictable:
        label = 'E_UNPREDICTABLE'
        next_step = 'redefine_target'
        meaning = 'observables barely determine q* even in-sample'
    else:
        label = 'E_MIXED'
        next_step = 'REVIEW'
        meaning = 'weak/partial signal; prefer target redesign unless stronger features appear'

    return dict(
        label=label, next_step=next_step, meaning=meaning,
        predictable=predictable, identity_heavy=identity,
        unpredictable=unpredictable,
        block_probe=dict(train_r2=tr_r2, dev_r2=dv_r2,
                         train_corr=tr_corr, dev_corr=dv_corr),
        image_probe=dict(train_corr=img_tr, dev_corr=img_dv),
        nn_block_spearman=nn_sp,
    )
