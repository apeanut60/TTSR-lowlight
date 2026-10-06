"""V3-A.1 shared runtime: tau target and the two strictly separated losses.

The V3-A implementation had one function used by both arms, so the "naive"
control silently carried the trust and reject terms. Here the two losses are
two functions with no shared code path for the supervision terms.
"""

import numpy as np
import torch
import torch.nn.functional as F

from v3a_runtime import usefulness

# V3-A.1 uses the verifier's native resolution (H/4), not H/8
Q_FACTOR = 4
LAMBDA_TRUST = 0.1
LAMBDA_SAFE = 0.1


def usefulness_at(y0, ref, hr, tau):
    """q_star at the verifier's native H/4 resolution."""
    return usefulness(y0, ref, hr, tau, factor=Q_FACTOR)


def tau_nocancel(y0, ref, hr):
    """Spatially-averaged |d| for one sample (no cancellation).

    The previous definition was ``median_i |mean_xy d_i|``: a field with equal
    positive and negative regions averaged to ~0 and inflated q_star's
    sharpness. This one averages |d| first.
    """
    _, d = usefulness(y0, ref, hr, 1.0, factor=Q_FACTOR)
    return float(d.abs().mean())


def compute_tau(pairs, ds, max_samples=None):
    """tau = median over samples of mean_xy|d| (+1e-6), plan §6.1."""
    vals = []
    n = len(pairs) if max_samples is None else min(max_samples, len(pairs))
    for i in range(n):
        _name, _lr, hr, ref, y0, _m = ds._load(i)
        vals.append(tau_nocancel(y0[None], ref[None], hr[None]))
    return float(np.median(vals)) + 1e-6, vals


def loss_r1_naive_from_out(out, hr, rec_fn, per_fn, smooth_fn, color_fn):
    """The V2.1 recipe: rec 1.0 + per 0.1 + smooth 1.0 + color 0.5."""
    l_rec = rec_fn(out, hr)
    l_per = per_fn(out, hr)
    l_sm = smooth_fn(out)
    l_co = color_fn(out)
    total = 1.0 * l_rec + 0.1 * l_per + 1.0 * l_sm + 0.5 * l_co
    return total, dict(rec=float(l_rec), per=float(l_per),
                       smooth=float(l_sm), color=float(l_co),
                       trust=0.0, safe=0.0)


def loss_r2_state(out, hr, y0, q_v4, q_star4, rec_fn, per_fn, smooth_fn,
                  color_fn, bad=False):
    """Reconstruction-side losses + trust, and for bad references + safety."""
    base, info = loss_r1_naive_from_out(out, hr, rec_fn, per_fn, smooth_fn, color_fn)
    q = q_v4.clamp(1e-6, 1 - 1e-6)
    l_trust = F.binary_cross_entropy(q, q_star4.clamp(1e-6, 1 - 1e-6))
    total = base + LAMBDA_TRUST * l_trust
    info['trust'] = float(l_trust)
    if bad:
        # plan §8: compute the safety term at full resolution, so q_star (H/4)
        # has to be upsampled to the output size first.
        q_full = F.interpolate(q_star4, size=out.shape[-2:], mode='bilinear',
                               align_corners=False)
        l_safe = ((1.0 - q_full) * (out - y0).abs()).mean()
        total = total + LAMBDA_SAFE * l_safe
        info['safe'] = float(l_safe)
    return total, info
