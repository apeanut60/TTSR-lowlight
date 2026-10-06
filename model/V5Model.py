"""V5.0 trainable reference branch. Frozen Base is driven by V5RetinexBridge.

No MHA, RDB, B0 RGB head, or output confidence gate.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.V5Correspondence import MATCH_CHUNK, ChunkedHardMatcher, MatchResult
from model.V5DeformAlign import FlowDeformTextureAlign
from model.V5MatchEncoder import MATCH_CH, V5MatchEncoder
from model.V5RefineBlock import V5RefineBlock
from model.V5RetinexBridge import BASE_H4_CH
from model.V3BFeatureBridge import BASE_H2_CH
from model.V5TextureEncoder import TEXTURE_CHANNELS, V5TextureEncoder

ARCHITECTURE = 'V5_aligned_ref_texture_v0'


def _rgb01(x_m11: Tensor) -> Tensor:
    return (x_m11 + 1.0) * 0.5


class V5Model(nn.Module):
    def __init__(
        self,
        d2_ch: int = BASE_H4_CH,
        d1_ch: int = BASE_H2_CH,
        texture_channels=TEXTURE_CHANNELS,
        match_ch: int = MATCH_CH,
        match_chunk: int = MATCH_CHUNK,
    ):
        super().__init__()
        c0, c1, c2 = texture_channels
        if c2 != d2_ch or c1 != d1_ch:
            raise SystemExit('texture pyramid must match D2/D1 channels')
        self.match_encoder = V5MatchEncoder(match_ch)
        self.texture_encoder = V5TextureEncoder(texture_channels)
        self.matcher = ChunkedHardMatcher(match_chunk)
        self.align_h4 = FlowDeformTextureAlign(c2)
        self.align_h2 = FlowDeformTextureAlign(c1)
        self.refine_h4 = V5RefineBlock(d2_ch, c2)
        self.refine_h2 = V5RefineBlock(d1_ch, c1)

    def match_and_texture(self, y0_m11: Tensor, ref_m11: Tensor):
        q = self.match_encoder(_rgb01(y0_m11))
        k = self.match_encoder(_rgb01(ref_m11))
        match = self.matcher(q, k)
        tex = self.texture_encoder(_rgb01(ref_m11))
        return match, tex

    def refine_at_h4(self, d2: Tensor, y0_m11: Tensor, ref_m11: Tensor):
        match, tex = self.match_and_texture(y0_m11, ref_m11)
        if match.flow.shape[-2:] != d2.shape[-2:]:
            raise SystemExit('H/4 match grid %s != D2 %s'
                             % (tuple(match.flow.shape[-2:]), tuple(d2.shape[-2:])))
        a2, aux2 = self.align_h4(tex['h4'], d2, match.flow, match.confidence)
        d2s, delta = self.refine_h4(d2, a2)
        aux2.update(dict(
            match=match, tex=tex, delta_d2=delta,
            match_flow_h4=match.flow, match_confidence_h4=match.confidence,
            match_index=match.index, sim_max=match.sim_max,
            margin=match.margin, entropy=match.entropy, aligned_h4=a2,
        ))
        return d2s, a2, aux2

    def refine_at_h2(self, d1: Tensor, y0_m11: Tensor, ref_m11: Tensor, h4_aux: Dict):
        match: MatchResult = h4_aux['match']
        tex = h4_aux['tex']
        h2 = tex['h2'].shape[-2:]
        flow_h2 = F.interpolate(match.flow, size=h2, mode='nearest') * 2.0
        conf_h2 = F.interpolate(match.confidence, size=h2, mode='nearest')
        if flow_h2.shape[-2:] != d1.shape[-2:]:
            raise SystemExit('H/2 flow grid %s != D1 %s'
                             % (tuple(flow_h2.shape[-2:]), tuple(d1.shape[-2:])))
        a1, aux1 = self.align_h2(tex['h2'], d1, flow_h2, conf_h2)
        d1s, delta = self.refine_h2(d1, a1)
        aux1.update(dict(
            delta_d1=delta, match_flow_h2=flow_h2,
            match_confidence_h2=conf_h2, aligned_h2=a1,
        ))
        return d1s, a1, aux1


def v5_step0_deltas_zero(model: V5Model) -> bool:
    for blk in (model.refine_h4, model.refine_h2):
        if float(blk.out.weight.abs().max()) > 0 or float(blk.out.bias.abs().max()) > 0:
            return False
    return True
