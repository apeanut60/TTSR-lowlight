"""Paper-guided, independent PyTorch implementation for TD-style 1x LLIE.

Paper: Huang et al., CVPR 2022, Task Decoupled Framework for RefSR.
Correspondences: Eq.(10), Eq.(11), supplemental Eq.(1)-(2).

This is NOT authors' official code. The 1x LLIE decoder/output residual are
explicit adaptations; see PLAN.md. This module needs only PyTorch.
"""
from __future__ import annotations

from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                               nn.ReLU(inplace=False),
                               nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.f(x)


class ContentFeatureH4(nn.Module):
    """1x LLIE adaptation: Y0 RGB -> content feature F_base at H/4.

    This is an engineering replacement for the original SISR task feature.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, y0: Tensor) -> Tensor:
        return self.net(y0)


class TextureResidualDecoder(nn.Module):
    """Aligned Ref feature -> RGB I_tex, supervised via Eq.(11).

    I_tex is a residual, not a restored image. No clamp in training.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.ReLU(inplace=False),
                                  ResidualBlock(channels))
        self.out = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tex_h4: Tensor, out_hw: tuple[int, int]) -> Tensor:
        v = self.body(tex_h4)
        v = F.interpolate(v, size=out_hw, mode='bilinear', align_corners=False)
        return self.out(v)


class AdaptiveTextureTransfer(nn.Module):
    """Paper Eq.(10): F_fused = F_base + C * Conv(F_tex).

    8 residual blocks follow as in the paper's supplementary material.
    A residual RGB output y=Y0+delta is a 1x-LLIE adaptation.
    """
    def __init__(self, channels: int = 64, n_blocks: int = 8):
        super().__init__()
        self.tex_projection = nn.Conv2d(channels, channels, 3, padding=1)
        self.restore = nn.Sequential(*(ResidualBlock(channels) for _ in range(n_blocks)))
        self.rgb_out = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.rgb_out.weight)
        nn.init.zeros_(self.rgb_out.bias)

    def forward(self, base_h4: Tensor, tex_h4: Tensor,
                confidence_h4: Tensor, y0: Tensor) -> Dict[str, Tensor]:
        if base_h4.shape != tex_h4.shape:
            raise ValueError(f'feature shape mismatch: {base_h4.shape} / {tex_h4.shape}')
        if confidence_h4.shape != (base_h4.shape[0], 1, *base_h4.shape[-2:]):
            raise ValueError(f'confidence shape mismatch: {confidence_h4.shape}')
        if y0.shape[0] != base_h4.shape[0] or y0.shape[1] != 3:
            raise ValueError('Y0 must be [B,3,H,W]')
        if not torch.isfinite(confidence_h4).all():
            raise ValueError('confidence must be finite')
        # Explicit LLIE engineering decision: cosine can be negative.
        # Do not pretend this clamp is an exact reproduction of authors' code.
        c = confidence_h4.clamp(0.0, 1.0)
        gated_texture = self.tex_projection(tex_h4) * c
        fused = base_h4 + gated_texture
        detail = self.restore(fused)
        detail = F.interpolate(detail, size=y0.shape[-2:],
                               mode='bilinear', align_corners=False)
        delta = self.rgb_out(detail)
        return {'y': y0 + delta, 'delta': delta, 'fused': fused,
                'gated_texture': gated_texture, 'confidence': c}


class TaskDecoupledTransferHead(nn.Module):
    """Two supervised outputs: I_tex and Y_final (paper Eq.(11), supp.).

    Inputs: frozen LLFormer Y0 and externally aligned reference feature F_tex.
    All RGB tensors [0,1] before model (outputs can leave this range).
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.content = ContentFeatureH4(channels)
        self.texture_decoder = TextureResidualDecoder(channels)
        self.att = AdaptiveTextureTransfer(channels, n_blocks=8)

    def forward(self, y0: Tensor, tex_h4: Tensor, c_h4: Tensor) -> Dict[str, Tensor]:
        if y0.ndim != 4 or y0.shape[1] != 3:
            raise ValueError('Y0 must be BCHW RGB')
        if y0.shape[-2] % 4 or y0.shape[-1] % 4:
            raise ValueError('Pad Y0 to a multiple of 4 before TD transfer')
        base_feat = self.content(y0)
        i_tex = self.texture_decoder(tex_h4, y0.shape[-2:])
        out = self.att(base_feat, tex_h4, c_h4, y0)
        out['i_tex'] = i_tex
        out['y0'] = y0
        return out


def td_reconstruction_losses(outputs: Dict[str, Tensor], gt01: Tensor,
                             tex_weight: float = 1.0,
                             rec_weight: float = 1.0) -> Dict[str, Tensor]:
    """L_rec=L1(Y,GT), L_tex=L1(Y0+I_tex,GT).

    This is the paper's reconstruction-loss subset, WITHOUT VGG/GAN losses.
    No pre-clamp; same [0,1] scale for Y0 and GT.
    """
    rec = F.l1_loss(outputs['y'], gt01)
    tex = F.l1_loss(outputs['y0'] + outputs['i_tex'], gt01)
    return {'loss': rec_weight * rec + tex_weight * tex,
            'l_rec': rec, 'l_tex': tex}
