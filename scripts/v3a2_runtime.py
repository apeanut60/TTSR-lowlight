"""V3-A.2 runtime: action-optimal gate target and its masked loss.

The V3-A.1 target asked "is the reference closer to the GT than Y0 is?", but the
verifier does not choose a reference -- it chooses *how much of a specific
correction to apply*. The right target is therefore the scalar that would make
that correction optimal in local RGB-MSE:

    Y(q) = Y0 + q * D,   D = g_v2 * delta
    q_opt = clamp( <H4-Y04, D4> / (||D4||^2 + eps), 0, 1 )

  * D points the right way      -> q_opt = 1
  * D is twice what is needed   -> q_opt = 0.5
  * D points the wrong way      -> q_opt = 0

Where D is ~0 the target is meaningless, so those locations are masked out
using a threshold measured once on the training split.
"""

import torch
import torch.nn.functional as F


def action_optimal_gate(y0, hr, d_correction, factor=4, eps=1e-8):
    """-> (q_opt [B,1,h,w] in [0,1], energy [B,1,h,w]).

    ``d_correction`` must already include the proposal's own gate
    (``g_v2 * delta``); passing the raw delta defines a different target.
    """
    h, w = y0.shape[-2:]
    th, tw = max(1, h // factor), max(1, w // factor)
    y4 = F.interpolate(y0, size=(th, tw), mode='area')
    h4 = F.interpolate(hr, size=(th, tw), mode='area')
    d4 = F.interpolate(d_correction, size=(th, tw), mode='area')
    num = ((h4 - y4) * d4).sum(dim=1, keepdim=True)
    den = (d4 ** 2).sum(dim=1, keepdim=True) + eps
    q = (num / den).clamp(0.0, 1.0)
    energy = (d4 ** 2).mean(dim=1, keepdim=True)      # mean over RGB
    return q, energy


def energy_threshold(energies):
    """p10 of the per-pixel proposal energy over the training split."""
    e = torch.cat([x.flatten() for x in energies])
    return float(torch.quantile(e.float(), 0.10))


def masked_smooth_l1(q_v, q_opt, mask, beta=0.1):
    """SmoothL1 averaged over the positions where the correction has energy."""
    loss = F.smooth_l1_loss(q_v, q_opt, reduction='none', beta=beta)
    m = mask.to(loss.dtype)
    denom = m.sum().clamp(min=1.0)
    return (loss * m).sum() / denom
