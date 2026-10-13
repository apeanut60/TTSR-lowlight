"""V7-RIVER: V6 align stack + ReBaIR-inspired AttnFusion at D2/D1."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.LLFormerBridge import D1_CHANNELS, D2_CHANNELS
from model.V5Correspondence import MATCH_CHUNK, ChunkedHardMatcher, MatchResult
from model.V5DeformAlign import FlowDeformTextureAlign
from model.V5MatchEncoder import MATCH_CH, V5MatchEncoder
from model.V6TextureEncoder import TEXTURE_CHANNELS, V6TextureEncoder
from model.V7RiverFusion import V7RiverFusion, v7_fusion_step0_zero

ARCHITECTURE = 'V7_river_attn_fusion_v0'
ARCHITECTURE_WIN3_D2 = 'V7_river_attn_fusion_win3_d2only_v1'

ALIGN_MODULES = (
    'match_encoder', 'texture_encoder', 'matcher', 'align_h4', 'align_h2',
)
FUSION_MODULES = ('refine_h4', 'refine_h2')


def _rgb01(x_m11: Tensor) -> Tensor:
    return (x_m11 + 1.0) * 0.5


class V7Model(nn.Module):
    def __init__(
        self,
        d2_ch: int = D2_CHANNELS,
        d1_ch: int = D1_CHANNELS,
        texture_channels=TEXTURE_CHANNELS,
        match_ch: int = MATCH_CH,
        match_chunk: int = MATCH_CHUNK,
        spatial_window: int = 1,
        d1_enabled: bool = True,
    ):
        super().__init__()
        c0, c1, c2 = texture_channels
        if c2 != d2_ch or c1 != d1_ch:
            raise SystemExit('texture pyramid must match D2/D1 channels')
        self.d2_ch = d2_ch
        self.d1_ch = d1_ch
        self.spatial_window = int(spatial_window)
        self.d1_enabled = bool(d1_enabled)
        # 'align' = Match/Flow/DCN; 'tex_h4' = bypass align, A2 = tex['h4']
        self.a2_mode = 'align'
        self.match_encoder = V5MatchEncoder(match_ch)
        self.texture_encoder = V6TextureEncoder(texture_channels)
        self.matcher = ChunkedHardMatcher(match_chunk)
        self.align_h4 = FlowDeformTextureAlign(c2)
        self.align_h2 = FlowDeformTextureAlign(c1)
        # D2: 64 = 4*16; D1: 32 = 2*16
        self.refine_h4 = V7RiverFusion(
            d2_ch, c2, n_heads=d2_ch // 16, head_dim=16,
            spatial_window=self.spatial_window)
        self.refine_h2 = V7RiverFusion(
            d1_ch, c1, n_heads=d1_ch // 16, head_dim=16,
            spatial_window=self.spatial_window)

    def match_and_texture(self, y0_m11: Tensor, ref_m11: Tensor):
        q = self.match_encoder(_rgb01(y0_m11))
        k = self.match_encoder(_rgb01(ref_m11))
        match = self.matcher(q, k)
        tex = self.texture_encoder(_rgb01(ref_m11))
        return match, tex

    def refine_at_h4(self, d2: Tensor, y0_m11: Tensor, ref_m11: Tensor):
        match, tex = self.match_and_texture(y0_m11, ref_m11)
        if tex['h4'].shape[-2:] != d2.shape[-2:]:
            raise SystemExit('H/4 tex grid %s != D2 %s'
                             % (tuple(tex['h4'].shape[-2:]), tuple(d2.shape[-2:])))
        if self.a2_mode == 'tex_h4':
            # Bypass Match/Flow/DCN: feed co-located texture features as A2.
            a2 = tex['h4']
            aux2 = dict(align_bypassed=True, a2_mode='tex_h4')
        elif self.a2_mode == 'align':
            if match.flow.shape[-2:] != d2.shape[-2:]:
                raise SystemExit('H/4 match grid %s != D2 %s'
                                 % (tuple(match.flow.shape[-2:]),
                                    tuple(d2.shape[-2:])))
            a2, aux2 = self.align_h4(tex['h4'], d2, match.flow, match.confidence)
            aux2['a2_mode'] = 'align'
        else:
            raise SystemExit('unknown a2_mode %r' % self.a2_mode)
        d2s, delta = self.refine_h4(d2, a2)
        aux2.update(dict(
            match=match, tex=tex, delta_d2=delta,
            match_flow_h4=match.flow, match_confidence_h4=match.confidence,
            match_index=match.index, sim_max=match.sim_max,
            margin=match.margin, entropy=match.entropy, aligned_h4=a2,
        ))
        return d2s, a2, aux2

    def refine_at_h2(self, d1: Tensor, y0_m11: Tensor, ref_m11: Tensor, h4_aux: Dict):
        if not self.d1_enabled:
            delta = torch.zeros_like(d1)
            return d1, d1, dict(
                delta_d1=delta, match_flow_h2=None, match_confidence_h2=None,
                aligned_h2=None, d1_disabled=True,
            )
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

    def apply_setting_freeze(self, setting: str) -> List[str]:
        """Setting A: fusion only. Setting B: fusion + align. Returns trainable names."""
        setting = setting.upper()
        if setting == 'A':
            freeze_names = ALIGN_MODULES
        elif setting == 'B':
            freeze_names = ()
        else:
            raise SystemExit('unknown setting %r (use A or B)' % setting)
        for name in ALIGN_MODULES:
            mod = getattr(self, name)
            train = name not in freeze_names
            mod.train(train)
            for p in mod.parameters():
                p.requires_grad_(train)
        for name in FUSION_MODULES:
            mod = getattr(self, name)
            # D2-only: keep refine_h2 frozen / unused
            train_fuse = True if (name != 'refine_h2' or self.d1_enabled) else False
            mod.train(train_fuse)
            for p in mod.parameters():
                p.requires_grad_(train_fuse)
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def trainable_parameters(self) -> Iterable[Tensor]:
        for p in self.parameters():
            if p.requires_grad:
                yield p


def v7_step0_deltas_zero(model: V7Model) -> bool:
    return (v7_fusion_step0_zero(model.refine_h4)
            and v7_fusion_step0_zero(model.refine_h2))
