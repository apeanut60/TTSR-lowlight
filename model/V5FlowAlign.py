"""Explicit backward warp by (dx, dy) flow. No soft averaging of texture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

WARP_TYPE = 'backward_grid_sample'


def warp_by_flow(
    x: Tensor,
    flow: Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
) -> Tensor:
    """flow[:,0]=dx, flow[:,1]=dy in source-feature pixels.

    Output (x,y) samples source at (x+dx, y+dy). align_corners=False.
    """
    b, _, h, w = x.shape
    if flow.shape != (b, 2, h, w):
        raise ValueError('flow %s incompatible with x %s'
                         % (tuple(flow.shape), tuple(x.shape)))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing='ij',
    )
    xx = xx[None] + flow[:, 0]
    yy = yy[None] + flow[:, 1]
    gx = (2.0 * xx + 1.0) / w - 1.0
    gy = (2.0 * yy + 1.0) / h - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        x, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
