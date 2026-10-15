"""v2.2 Full-Image DC-aware residual loss.

L = L_spatial + λ_DC * L_DC

When λ_DC=1, L equals MSE(Y0+D, GT) over all RGB×H×W elements
(because Var(D)=E[(D-m)^2] and E[(D-m)^2]+||m-m*||^2 expands to MSE of means
and centered residuals against the target residual).
"""

from __future__ import annotations

import torch


def mean_hw(x: torch.Tensor) -> torch.Tensor:
    """Per-channel spatial mean. x: [B,C,H,W] -> [B,C,1,1]."""
    if x.dim() != 4:
        raise ValueError('expected [B,C,H,W], got %s' % (tuple(x.shape),))
    return x.mean(dim=(-2, -1), keepdim=True)


def split_dc(d: torch.Tensor):
    """Return (d_mean [B,C,1,1], d_spatial [B,C,H,W])."""
    m = mean_hw(d)
    return m, d - m


def dc_aware_residual_loss(d_pred: torch.Tensor, d_tgt: torch.Tensor,
                           lambda_dc: float = 1.0):
    """Compute L_spatial, L_DC, and total L.

    All MSEs are mean over every element of their tensors.
    """
    m_pred, s_pred = split_dc(d_pred)
    m_tgt, s_tgt = split_dc(d_tgt)
    l_sp = torch.mean((s_pred - s_tgt) ** 2)
    l_dc = torch.mean((m_pred - m_tgt) ** 2)
    # Expand m terms to same element count as spatial for fair MSE definition:
    # mean over [B,C,1,1] equals mean over broadcast, but for λDC=1 equivalence
    # to full MSE we need L_DC weighted by the same per-element measure as
    # spatial. Full MSE = mean_HW C ((s_pred+m_pred)-(s_tgt+m_tgt))^2
    #                   = mean((s_pred-s_tgt)^2) + mean((m_pred-m_tgt)^2)
    # because cross terms vanish (spatial zero-mean). Here mean((m_diff)^2)
    # over HxW equals m_diff^2 (constant map), so:
    #   mean_full((m_diff)^2) = mean_C (m_diff^2)   [same as mean over [B,C,1,1]]
    # Yes — averaging a constant map over H,W does not change the value.
    total = l_sp + float(lambda_dc) * l_dc
    return total, l_sp, l_dc, m_pred, m_tgt


def full_mse_from_residual(d_pred: torch.Tensor, d_tgt: torch.Tensor):
    return torch.mean((d_pred - d_tgt) ** 2)
