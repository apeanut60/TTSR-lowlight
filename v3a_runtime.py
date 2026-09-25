"""Shared helpers for V3-A: usefulness targets and bad-reference construction.

The verifier is trained against ``q_star``, the **relative usefulness** of the
reference compared with the base output:

    e0 = mean_rgb |D8(Y0) - D8(H)|        error of the base output
    eR = mean_rgb |D8(R)  - D8(H)|        error of the reference
    d  = e0 - eR                          >0 => the reference is closer to the GT
    q_star = sigmoid(d / tau)

``q_star`` is *not* a hallucination probability and is not calibrated: it only
orders samples by "is this reference more useful than what we already have".
"""

import numpy as np
import torch
import torch.nn.functional as F


def d8(x, factor=8):
    """Area downsample to H/factor x W/factor (never nearest)."""
    h, w = x.shape[-2:]
    th, tw = max(1, h // factor), max(1, w // factor)
    if (h, w) == (th, tw):
        return x
    return F.interpolate(x, size=(th, tw), mode='area')


def usefulness(y0, ref, hr, tau, factor=8):
    """-> (q_star, d) with shapes [B,1,h,w] and [B,1,h,w]."""
    if y0.shape != ref.shape or y0.shape != hr.shape:
        raise ValueError('usefulness needs y0/ref/hr of equal shape, got %s %s %s'
                         % (tuple(y0.shape), tuple(ref.shape), tuple(hr.shape)))
    y8, r8, h8 = d8(y0, factor), d8(ref, factor), d8(hr, factor)
    e0 = (y8 - h8).abs().mean(dim=1, keepdim=True)
    eR = (r8 - h8).abs().mean(dim=1, keepdim=True)
    d = e0 - eR
    q = torch.sigmoid(d / tau)
    return q, d


def scalar_d(y0, ref, hr, factor=8):
    """Per-sample mean of d, used to pick tau over the training set."""
    _, d = usefulness(y0, ref, hr, 1.0, factor)
    return d.mean(dim=(1, 2, 3))


def tau_from_samples(ds):
    """tau = median(|d|) over the training samples (+1e-6), per plan §4."""
    vals = np.concatenate([np.abs(ds[i].numpy()) for i in range(len(ds))])
    return float(np.median(vals)) + 1e-6


def mismatch_permutation(rows):
    """Fixed cyclic shift with no fixed points (within each camera block)."""
    perm = list(range(len(rows)))
    for cam in sorted({r['camera'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['camera'] == cam]
        if len(idx) > 1:
            for pos, i in enumerate(idx):
                perm[i] = idx[(pos + 1) % len(idx)]
    assert all(perm[i] != i for i in range(len(rows))), 'fixed point in derangement'
    return perm


def splice_corrupt(ref, donor, rng, min_frac=0.25, max_frac=0.5, n_blocks=2):
    """Replace 1..n_blocks rectangles of ``ref`` with the corresponding area of
    ``donor`` (a different sample's reference). Block edges are a fraction of
    the image side, so the corrupt region is large enough to be visible at H/8.
    """
    out = ref.clone()
    c, h, w = ref.shape
    for _ in range(int(rng.integers(1, n_blocks + 1))):
        fh = float(rng.uniform(min_frac, max_frac))
        fw = float(rng.uniform(min_frac, max_frac))
        bh, bw = max(1, int(round(h * fh))), max(1, int(round(w * fw)))
        y0 = int(rng.integers(0, h - bh + 1))
        x0 = int(rng.integers(0, w - bw + 1))
        out[:, y0:y0 + bh, x0:x0 + bw] = donor[:, y0:y0 + bh, x0:x0 + bw]
    return out


def rescale_reference(ref, gain):
    """Wrong-exposure bad reference: scale around mid-grey, then clip to [-1,1]."""
    return (ref * gain).clamp(-1.0, 1.0)
