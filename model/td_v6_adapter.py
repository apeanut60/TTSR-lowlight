"""V7 Task-Decoupled Ref transfer on frozen LLFormer Y0 (paper-guided).

Y = Y0 + ATT(content(Y0), F_tex) gated by C   (Eq.10)
I_tex supervised via L_tex = L1(GT, Y0+I_tex)  (Eq.11)
No D2/D1 injection. Matcher: pretrained C2 ContrasExtractorSep.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from model.V5Correspondence import ChunkedHardMatcher
from model.V5DeformAlign import FlowDeformTextureAlign
from model.V6TextureEncoder import V6TextureEncoder
from model.c2_extractor import C2MatchFeatures, C2_FEATURE_CKPT
from model.td_transfer import TaskDecoupledTransferHead, td_reconstruction_losses


def _rgb01(x_m11: Tensor) -> Tensor:
    return (x_m11 + 1.0) * 0.5


class V7TaskDecoupledModel(nn.Module):
    """Y0 + Ref → C2 match → Flow/DCN → I_tex + ATT → Y."""

    def __init__(self,
                 match_chunk: int = 1024,
                 texture_ch: int = 64,
                 c2_ckpt: str = C2_FEATURE_CKPT,
                 freeze_c2: bool = True):
        super().__init__()
        self.match_features = C2MatchFeatures(ckpt=c2_ckpt, freeze=freeze_c2)
        self.matcher_is_pretrained = True
        self.correspondence = ChunkedHardMatcher(chunk_size=match_chunk)
        self.texture_encoder = V6TextureEncoder((16, 32, texture_ch))
        self.aligner = FlowDeformTextureAlign(texture_ch)
        self.transfer = TaskDecoupledTransferHead(texture_ch)

    def assert_formal_ready(self):
        if not self.matcher_is_pretrained:
            raise RuntimeError(
                'HARD STOP: matcher is unverified. Load C2 feature_extraction.pth.')

    def forward(self, y0_01: Tensor, ref_01: Tensor) -> Dict[str, Tensor]:
        if y0_01.shape != ref_01.shape:
            raise ValueError('1x pilot requires Y0/Ref same BCHW')
        q, k = self.match_features(y0_01, ref_01)
        m = self.correspondence(q, k)
        tex_ref = self.texture_encoder(ref_01)['h4']
        base_feat = self.transfer.content(y0_01)
        if (tex_ref.shape != base_feat.shape
                or tex_ref.shape[-2:] != m.flow.shape[-2:]):
            raise ValueError(
                'H/4 grid mismatch: tex=%s base=%s flow=%s'
                % (tuple(tex_ref.shape), tuple(base_feat.shape),
                   tuple(m.flow.shape)))
        aligned, aux_align = self.aligner(
            tex_ref, base_feat, m.flow.detach(), m.confidence)
        result = self.transfer(y0_01, aligned, m.confidence)
        result['flow_h4'] = m.flow
        result['raw_confidence_h4'] = m.confidence
        result['matched_index'] = m.index
        result['margin_h4'] = m.margin
        result['aligned_h4'] = aligned
        result['q'] = q
        result['k'] = k
        result.update({('align_%s' % kk): vv for kk, vv in aux_align.items()})
        return result

    def forward_m11(self, y0_m11: Tensor, ref_m11: Tensor) -> Dict[str, Tensor]:
        """Convenience: dataset tensors in [-1,1] → internal [0,1] → out [0,1]."""
        out = self.forward(_rgb01(y0_m11), _rgb01(ref_m11))
        return out

    def trainable_named_parameters(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p


def build_v7_model(c2_ckpt: Optional[str] = None, **kw) -> V7TaskDecoupledModel:
    if c2_ckpt is not None:
        kw['c2_ckpt'] = c2_ckpt
    return V7TaskDecoupledModel(**kw)


__all__ = [
    'V7TaskDecoupledModel', 'build_v7_model',
    'td_reconstruction_losses', '_rgb01',
]
