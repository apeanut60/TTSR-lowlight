"""Reference-guided query illumination alignment (v1r6/v1r7).

v1r6: global gamma toward Nano Ref mean luma (Query only).
v1r7: + partial luminance histogram matching after gamma (Query only).

Does not use GT. MainNet Low / Ref Value / Key paths are untouched.
"""

from __future__ import annotations

import torch

# Rec.709 luma weights
_LUMA = (0.2126, 0.7152, 0.0722)


def _rgb_luma01(x01: torch.Tensor) -> torch.Tensor:
    """x01: [N,3,H,W] in [0,1] -> [N,1,H,W] luma."""
    w = x01.new_tensor(_LUMA).view(1, 3, 1, 1)
    return (x01 * w).sum(dim=1, keepdim=True)


def _mean_luma(x01: torch.Tensor) -> torch.Tensor:
    """Per-image mean luma → [N]."""
    return _rgb_luma01(x01).flatten(1).mean(dim=1)


def _solve_gamma_binary(
    low01: torch.Tensor,
    target_mean: torch.Tensor,
    gamma_min: float,
    gamma_max: float,
    iters: int = 24,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-image gamma via binary search so mean(low**gamma) ≈ target_mean."""
    n = low01.shape[0]
    device, dtype = low01.device, low01.dtype
    lo = torch.full((n,), float(gamma_min), device=device, dtype=dtype)
    hi = torch.full((n,), float(gamma_max), device=device, dtype=dtype)
    base = low01.clamp(min=eps, max=1.0)
    tgt = target_mean.clamp(min=eps, max=1.0)
    src_mean = _mean_luma(base).clamp(min=eps)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        powered = base.pow(mid.view(n, 1, 1, 1))
        cur = _mean_luma(powered)
        too_bright = cur > tgt
        lo = torch.where(too_bright, mid, lo)
        hi = torch.where(too_bright, hi, mid)

    gamma = 0.5 * (lo + hi)
    both_dark = (src_mean < 1e-4) & (tgt < 1e-4)
    gamma = torch.where(both_dark, torch.ones_like(gamma), gamma)
    return gamma.clamp(gamma_min, gamma_max)


def _apply_gamma_blend(
    low01: torch.Tensor,
    gamma: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return blended query in [0,1]: (1-a)*low + a*low**gamma."""
    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha <= 0.0:
        return low01
    low_g = low01.clamp(min=1e-6).pow(gamma.view(-1, 1, 1, 1)).clamp(0.0, 1.0)
    return ((1.0 - alpha) * low01 + alpha * low_g).clamp(0.0, 1.0)


def _hist_cdf(y01: torch.Tensor, bins: int = 256):
    """Per-image histogram + CDF of luma in [0,1].

    Returns:
        hist: [N, bins]
        cdf: [N, bins]  (right-endpoint cumulative, last=1)
    """
    n = y01.shape[0]
    y = y01.reshape(n, -1).clamp(0.0, 1.0)
    # bucketize into [0, bins-1]
    idx = (y * (bins - 1)).long().clamp(0, bins - 1)
    hist = torch.zeros(n, bins, device=y.device, dtype=y.dtype)
    ones = torch.ones_like(idx, dtype=y.dtype)
    hist.scatter_add_(1, idx, ones)
    hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1.0)
    cdf = hist.cumsum(dim=1)
    # numerical: force last bin to 1
    cdf = cdf / cdf[:, -1:].clamp(min=1e-12)
    return hist, cdf


def build_luma_hist_lut(
    src01: torch.Tensor,
    ref01: torch.Tensor,
    bins: int = 256,
) -> torch.Tensor:
    """Monotonic CDF-matching LUT: src luma bin → ref luma value.

    Args:
        src01, ref01: [N,3,H,W] in [0,1] (sizes may differ; ref resized).
    Returns:
        lut: [N, bins] in [0,1], non-decreasing along bins.
    """
    if ref01.shape[-2:] != src01.shape[-2:]:
        ref01 = torch.nn.functional.interpolate(
            ref01, size=src01.shape[-2:], mode='bilinear', align_corners=False)
    ys = _rgb_luma01(src01)
    yr = _rgb_luma01(ref01)
    _, cdf_s = _hist_cdf(ys, bins)
    _, cdf_r = _hist_cdf(yr, bins)
    n = src01.shape[0]
    device, dtype = src01.device, src01.dtype
    # Bin centers for source; for each src CDF value, invert ref CDF
    # lut[b] = first ref intensity whose CDF >= CDF_src[b]
    ref_levels = torch.linspace(0.0, 1.0, bins, device=device, dtype=dtype)
    lut = torch.empty(n, bins, device=device, dtype=dtype)
    for i in range(n):
        # searchsorted on cdf_r[i]: for each cdf_s value find insertion index
        # cdf_r is non-decreasing
        cr = cdf_r[i]
        cs = cdf_s[i]
        # Ensure strictly increasing for searchsorted stability on plateaus
        cr_eps = cr + torch.arange(bins, device=device, dtype=dtype) * 1e-12
        j = torch.searchsorted(cr_eps, cs, right=False).clamp(0, bins - 1)
        lut[i] = ref_levels[j]
    # Enforce monotonicity (should already hold; belt-and-suspenders)
    lut, _ = torch.cummax(lut, dim=1)
    return lut.clamp(0.0, 1.0)


def apply_luma_hist_lut(
    rgb01: torch.Tensor,
    lut: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Partial hist match on luma; preserve chromaticity via safe scaling.

    Y_new = (1-beta)*Y + beta*H(Y); RGB *= Y_new / Y.
    """
    beta = float(max(0.0, min(1.0, beta)))
    if beta <= 0.0:
        return rgb01
    n, _, h, w = rgb01.shape
    bins = lut.shape[-1]
    y = _rgb_luma01(rgb01).clamp(0.0, 1.0)  # [N,1,H,W]
    # Continuous lookup: index in [0, bins-1]
    pos = y * (bins - 1)
    i0 = pos.long().clamp(0, bins - 2)
    i1 = i0 + 1
    t = (pos - i0.float()).clamp(0.0, 1.0)
    # Gather lut values
    lut_flat = lut  # [N, bins]
    # reshape indices to [N, HW]
    i0f = i0.view(n, -1)
    i1f = i1.view(n, -1)
    v0 = torch.gather(lut_flat, 1, i0f)
    v1 = torch.gather(lut_flat, 1, i1f)
    y_h = (v0 + (v1 - v0) * t.view(n, -1)).view(n, 1, h, w)
    y_new = ((1.0 - beta) * y + beta * y_h).clamp(0.0, 1.0)
    eps = 1e-6
    scale = y_new / y.clamp(min=eps)
    # Near-black: replace with gray of y_new to avoid blow-up
    near_black = y < eps
    out = (rgb01 * scale).clamp(0.0, 1.0)
    if near_black.any():
        gray = y_new.expand_as(rgb01)
        out = torch.where(near_black.expand_as(rgb01), gray, out)
    return out.clamp(0.0, 1.0)


def cdf_l1_distance(a01: torch.Tensor, b01: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Normalized mean |CDF_a - CDF_b| per image → [N]."""
    if b01.shape[-2:] != a01.shape[-2:]:
        b01 = torch.nn.functional.interpolate(
            b01, size=a01.shape[-2:], mode='bilinear', align_corners=False)
    _, ca = _hist_cdf(_rgb_luma01(a01), bins)
    _, cb = _hist_cdf(_rgb_luma01(b01), bins)
    return (ca - cb).abs().mean(dim=1)


def estimate_query_gamma(
    low: torch.Tensor,
    ref: torch.Tensor,
    gamma_min: float = 0.3,
    gamma_max: float = 1.0,
) -> torch.Tensor:
    """Full-image gamma estimate ([N])."""
    low01 = ((low + 1.0) * 0.5).clamp(0.0, 1.0)
    ref01 = ((ref + 1.0) * 0.5).clamp(0.0, 1.0)
    if ref01.shape[-2:] != low01.shape[-2:]:
        ref01 = torch.nn.functional.interpolate(
            ref01, size=low01.shape[-2:], mode='bilinear', align_corners=False)
    return _solve_gamma_binary(low01, _mean_luma(ref01), gamma_min, gamma_max)


def estimate_query_align_maps(
    low: torch.Tensor,
    ref: torch.Tensor,
    alpha: float = 0.75,
    gamma_min: float = 0.3,
    gamma_max: float = 1.0,
    bins: int = 256,
):
    """Full-image gamma + hist LUT for training/tiled reuse.

    Returns:
        gamma: [N]
        lut: [N, bins]  (built on gamma-blended Low vs Ref; identity if alpha<=0)
        info: diagnostic dict
    """
    low01 = ((low + 1.0) * 0.5).clamp(0.0, 1.0)
    ref01 = ((ref + 1.0) * 0.5).clamp(0.0, 1.0)
    if ref01.shape[-2:] != low01.shape[-2:]:
        ref01_stat = torch.nn.functional.interpolate(
            ref01, size=low01.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ref01_stat = ref01

    mean_low = _mean_luma(low01)
    mean_ref = _mean_luma(ref01_stat)
    gamma = _solve_gamma_binary(low01, mean_ref, gamma_min, gamma_max)
    low_g01 = _apply_gamma_blend(low01, gamma, alpha)
    mean_gamma = _mean_luma(low_g01)

    n = low01.shape[0]
    if float(alpha) <= 0.0:
        # Identity LUT (bin centers)
        lut = torch.linspace(0.0, 1.0, bins, device=low.device, dtype=low.dtype)
        lut = lut.view(1, bins).expand(n, -1).contiguous()
        cdf_dist = cdf_l1_distance(low01, ref01_stat, bins)
    else:
        lut = build_luma_hist_lut(low_g01, ref01_stat, bins=bins)
        cdf_dist = cdf_l1_distance(low_g01, ref01_stat, bins)

    info = {
        'gamma': gamma.detach(),
        'mean_low': mean_low.detach(),
        'mean_ref': mean_ref.detach(),
        'mean_gamma': mean_gamma.detach(),
        'cdf_l1_before_hist': cdf_dist.detach(),
        'alpha': float(alpha),
        'bins': bins,
    }
    return gamma, lut, info


def align_query_illumination(
    low: torch.Tensor,
    ref: torch.Tensor,
    alpha: float = 1.0,
    gamma_min: float = 0.3,
    gamma_max: float = 1.0,
    gamma_override: torch.Tensor | None = None,
    beta: float = 0.0,
    hist_bins: int = 256,
    lut_override: torch.Tensor | None = None,
):
    """Align Low Query toward Nano Ref (gamma + optional hist). Never uses GT.

    Args:
        low, ref: [N,3,H,W] in [-1,1].
        alpha: gamma blend strength.
        beta: partial hist strength after gamma (0 = gamma-only / v1r6).
        gamma_override: [N] full-image gamma.
        lut_override: [N, bins] full-image hist LUT (from gamma-aligned full Low).

    Returns:
        low_query: [N,3,H,W] in [-1,1]
        info: diagnostics
    """
    if low.ndim != 4 or ref.ndim != 4:
        raise ValueError('low/ref must be NCHW')
    if low.shape[0] != ref.shape[0] or low.shape[1] != 3 or ref.shape[1] != 3:
        raise ValueError('batch/channel mismatch')

    alpha = float(max(0.0, min(1.0, alpha)))
    beta = float(max(0.0, min(1.0, beta)))
    low01 = ((low + 1.0) * 0.5).clamp(0.0, 1.0)
    ref01 = ((ref + 1.0) * 0.5).clamp(0.0, 1.0)

    if ref01.shape[-2:] != low01.shape[-2:]:
        ref01_stat = torch.nn.functional.interpolate(
            ref01, size=low01.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ref01_stat = ref01

    mean_low = _mean_luma(low01)
    mean_ref = _mean_luma(ref01_stat)

    if gamma_override is not None:
        gamma = gamma_override.to(device=low.device, dtype=low.dtype).view(-1)
        if gamma.numel() != low.shape[0]:
            raise ValueError('gamma_override batch mismatch')
        gamma = gamma.clamp(gamma_min, gamma_max)
    else:
        gamma = _solve_gamma_binary(low01, mean_ref, gamma_min, gamma_max)

    if alpha <= 0.0 and beta <= 0.0:
        low_query = low
        mean_gamma = mean_low
        mean_final = mean_low
        cdf_before = cdf_l1_distance(low01, ref01_stat, hist_bins)
        cdf_after = cdf_before
        lut = None
    else:
        low_g01 = _apply_gamma_blend(low01, gamma, alpha)
        mean_gamma = _mean_luma(low_g01)
        cdf_before = cdf_l1_distance(low_g01, ref01_stat, hist_bins)

        if beta > 0.0:
            if lut_override is not None:
                lut = lut_override.to(device=low.device, dtype=low.dtype)
                if lut.ndim == 1:
                    lut = lut.view(1, -1)
                if lut.shape[0] != low.shape[0]:
                    raise ValueError('lut_override batch mismatch')
            else:
                lut = build_luma_hist_lut(low_g01, ref01_stat, bins=hist_bins)
            low_h01 = apply_luma_hist_lut(low_g01, lut, beta)
            mean_final = _mean_luma(low_h01)
            cdf_after = cdf_l1_distance(low_h01, ref01_stat, hist_bins)
            low_query = (low_h01 * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            lut = None
            mean_final = mean_gamma
            cdf_after = cdf_before
            low_query = (low_g01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    info = {
        'gamma': gamma.detach(),
        'mean_low': mean_low.detach(),
        'mean_ref': mean_ref.detach(),
        'mean_gamma': mean_gamma.detach(),
        'mean_aligned': mean_final.detach(),  # backward-compat alias
        'mean_final': mean_final.detach(),
        'cdf_l1_before_hist': cdf_before.detach(),
        'cdf_l1_after_hist': cdf_after.detach(),
        'alpha': alpha,
        'beta': beta,
        'lut': None if lut is None else lut.detach(),
    }
    return low_query, info
