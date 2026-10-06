"""Flow-guided residual DCN + flow/DCN fusion. C is a fusion feature, not a Y gate."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from model.V5FlowAlign import warp_by_flow

try:
    from torchvision.ops import deform_conv2d
except Exception:
    deform_conv2d = None

DCN_KERNEL = 3
DCN_MASK_BIAS = 8.0


class FlowGuidedDeformAlign(nn.Module):
    """coarse P as base offset; OffsetNet predicts residual ΔP + sigmoid mask."""

    def __init__(self, ch: int, hidden: int | None = None, kernel_size: int = DCN_KERNEL):
        super().__init__()
        if deform_conv2d is None:
            raise ImportError('torchvision.ops.deform_conv2d is required')
        hidden = hidden or ch
        self.ch = int(ch)
        self.ks = int(kernel_size)
        kk = self.ks * self.ks
        self.offset_mask = nn.Sequential(
            nn.Conv2d(ch * 2, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 3 * kk, 3, 1, 1),
        )
        self.weight = nn.Parameter(torch.zeros(ch, ch, self.ks, self.ks))
        self.bias = nn.Parameter(torch.zeros(ch))
        with torch.no_grad():
            idx = torch.arange(ch)
            self.weight[idx, idx, self.ks // 2, self.ks // 2] = 1.0
            nn.init.zeros_(self.offset_mask[-1].weight)
            nn.init.zeros_(self.offset_mask[-1].bias)
            self.offset_mask[-1].bias[2 * kk:] = DCN_MASK_BIAS

    def forward(
        self,
        ref_feat: Tensor,
        target_feat: Tensor,
        coarse_flow: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        b, c, h, w = ref_feat.shape
        if target_feat.shape != ref_feat.shape:
            raise ValueError('target/ref feature shape mismatch')
        if coarse_flow.shape != (b, 2, h, w):
            raise ValueError('coarse_flow shape mismatch')
        flow_warp = warp_by_flow(ref_feat, coarse_flow, mode='bilinear')
        pred = self.offset_mask(torch.cat([target_feat, flow_warp], dim=1))
        kk = self.ks * self.ks
        residual_offset = pred[:, : 2 * kk]
        mask = torch.sigmoid(pred[:, 2 * kk:])
        # torchvision offsets: (dy, dx) per kernel tap
        base_yx = coarse_flow[:, [1, 0]].unsqueeze(1).repeat(1, kk, 1, 1, 1)
        base_offset = base_yx.reshape(b, 2 * kk, h, w)
        offset = base_offset + residual_offset
        deform = deform_conv2d(
            input=ref_feat,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            padding=self.ks // 2,
            mask=mask,
        )
        return deform, residual_offset, mask


class FlowDeformTextureAlign(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.deform = FlowGuidedDeformAlign(ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2 + 1, ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(
        self,
        ref_feat: Tensor,
        target_feat: Tensor,
        flow: Tensor,
        confidence: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        flow_feat = warp_by_flow(ref_feat, flow, mode='bilinear')
        deform_feat, residual_offset, mask = self.deform(
            ref_feat, target_feat, flow)
        aligned = self.fuse(torch.cat([flow_feat, deform_feat, confidence], dim=1))
        aux = dict(
            flow_feat=flow_feat,
            deform_feat=deform_feat,
            residual_offset=residual_offset,
            deform_mask=mask,
        )
        return aligned, aux
