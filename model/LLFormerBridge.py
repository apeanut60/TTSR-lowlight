"""Staged LLFormer bridge for V6 Base training / future Ref injection.

Preserves official LLFormer computation while exposing:
  D2 = decoder_level2_1 output BEFORE up2_1   (64ch @ H/4)
  D1 = decoder_level1_1 output BEFORE up2_0   (32ch @ H/2)
  Y0 = final RGB in [0, 1]

Base-only training uses pass-through staged decode (no Ref refine).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LLFormer import LLFormer

BRIDGE_VERSION = 'llformer_bridge_v1'
PAD_MULTIPLE = 16
D2_CHANNELS = 64
D1_CHANNELS = 32
D2_SCALE = 4
D1_SCALE = 2
D2_TENSOR_NAME = 'out_enc_level2_1_pre_upsample'
D1_TENSOR_NAME = 'out_enc_level1_1_pre_upsample'

OFFICIAL_LOL_CKPT = (
    '/root/projects/TTSR-lowlight/pretrain_model/llformer/model_bestPSNR.pth')

LOL_ARCH = dict(
    inp_channels=3,
    out_channels=3,
    dim=16,
    num_blocks=[2, 4, 8, 16],
    num_refinement_blocks=2,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='WithBias',
    attention=True,
    skip=False,
)


def build_official_llformer(**overrides) -> LLFormer:
    cfg = dict(LOL_ARCH)
    cfg.update(overrides)
    return LLFormer(**cfg)


def strip_module_prefix(state_dict):
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k[7:] if k.startswith('module.') else k] = v
    return out


def load_official_state_dict(path: str) -> OrderedDict:
    blob = torch.load(path, map_location='cpu')
    if isinstance(blob, dict) and 'state_dict' in blob:
        sd = blob['state_dict']
    else:
        sd = blob
    return strip_module_prefix(sd)


def load_into_llformer(net: nn.Module, path: str) -> Dict[str, int]:
    """Strict full load. Returns counts; raises on any mismatch."""
    clean = load_official_state_dict(path)
    msd = net.state_dict()
    miss = [k for k in msd if k not in clean]
    extra = [k for k in clean if k not in msd]
    shape = [k for k in msd if k in clean
             and tuple(msd[k].shape) != tuple(clean[k].shape)]
    if miss or extra or shape:
        raise SystemExit(
            'LLFormer ckpt incomplete: missing=%d unexpected=%d '
            'shape_mismatch=%d first_miss=%s'
            % (len(miss), len(extra), len(shape), (miss or shape or extra)[:4]))
    net.load_state_dict(clean, strict=True)
    return dict(loaded=len(clean), missing=0, unexpected=0, shape_mismatch=0)


def reflect_pad_to_multiple(x: torch.Tensor, multiple: int = PAD_MULTIPLE):
    """Pad H,W with reflect to multiple. Returns padded, h0, w0."""
    _, _, h, w = x.shape
    H = ((h + multiple - 1) // multiple) * multiple
    W = ((w + multiple - 1) // multiple) * multiple
    padh, padw = H - h, W - w
    if padh or padw:
        x = F.pad(x, (0, padw, 0, padh), mode='reflect')
    return x, h, w


def crop_to_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x[..., :h, :w]


class LLFormerBridge(LLFormer):
    """Official LOL arch + staged encode/decode for future Ref refine."""

    def __init__(self, **overrides):
        cfg = dict(LOL_ARCH)
        cfg.update(overrides)
        super().__init__(**cfg)
        self.bridge_version = BRIDGE_VERSION

    def encode_to_d2(self, x: torch.Tensor) -> Dict[str, Any]:
        """X [0,1] (already padded) → state with D2 @ H/4, 64ch."""
        inp_enc_encoder1 = self.patch_embed(x)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2)

        inp_fusion_123 = torch.cat(
            [out_enc_encoder1.unsqueeze(1),
             out_enc_encoder2.unsqueeze(1),
             out_enc_encoder3.unsqueeze(1)], dim=1)
        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        inp_enc_level1_0 = self.down_1(out_fusion_123)
        out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)

        inp_enc_level2_0 = self.down_2(out_enc_level1_0)
        out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)

        inp_enc_level3_0 = self.down_3(out_enc_level2_0)
        out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0)

        inp_enc_level4_0 = self.down_4(out_enc_level3_0)
        out_enc_level4_0 = self.decoder_level4(inp_enc_level4_0)

        out_enc_level4_0 = self.up4_3(out_enc_level4_0)
        inp_enc_level3_1 = (
            self.coefficient_4_3[0, :][None, :, None, None] * out_enc_level3_0
            + self.coefficient_4_3[1, :][None, :, None, None] * out_enc_level4_0)
        inp_enc_level3_1 = self.skip_4_3(inp_enc_level3_1)
        out_enc_level3_1 = self.decoder_level3_1(inp_enc_level3_1)

        out_enc_level3_1 = self.up3_2(out_enc_level3_1)
        inp_enc_level2_1 = (
            self.coefficient_3_2[0, :][None, :, None, None] * out_enc_level2_0
            + self.coefficient_3_2[1, :][None, :, None, None] * out_enc_level3_1)
        inp_enc_level2_1 = self.skip_3_2(inp_enc_level2_1)
        d2 = self.decoder_level2_1(inp_enc_level2_1)

        if int(d2.shape[1]) != D2_CHANNELS:
            raise SystemExit('D2 channels %d != %d' % (int(d2.shape[1]), D2_CHANNELS))

        return dict(
            inp_img=x,
            out_fusion_123=out_fusion_123,
            out_enc_level1_0=out_enc_level1_0,
            d2=d2,
        )

    def decode_d2_to_d1(self, state: Dict[str, Any],
                        d2: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """D2* → D1 @ H/2, 32ch (pre full-res upsample)."""
        if d2 is None:
            d2 = state['d2']
        up = self.up2_1(d2)
        inp = (
            self.coefficient_2_1[0, :][None, :, None, None] * state['out_enc_level1_0']
            + self.coefficient_2_1[1, :][None, :, None, None] * up)
        inp = self.skip_1_0(inp)
        d1 = self.decoder_level1_1(inp)
        if int(d1.shape[1]) != D1_CHANNELS:
            raise SystemExit('D1 channels %d != %d' % (int(d1.shape[1]), D1_CHANNELS))
        out = dict(state)
        out['d1'] = d1
        return out

    def decode_d1_to_rgb(self, state: Dict[str, Any],
                         d1: Optional[torch.Tensor] = None) -> torch.Tensor:
        """D1* → Y RGB [0,1] (same domain as official; no clamp here)."""
        if d1 is None:
            d1 = state['d1']
        up = self.up2_0(d1)
        fusion = self.latent(state['out_fusion_123'])
        out = (
            self.coefficient_1_0[0, :][None, :, None, None] * fusion
            + self.coefficient_1_0[1, :][None, :, None, None] * up)
        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)
        inp_fusion = torch.cat(
            [out_1.unsqueeze(1), out_2.unsqueeze(1), out_3.unsqueeze(1)], dim=1)
        fused = self.layer_fussion_2(inp_fusion)
        feat = self.conv_fuss_2(fused)
        if self.skip:
            return self.output(feat) + state['inp_img']
        return self.output(feat)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return D2, D1, optional F0, Y0 for a padded [0,1] tensor."""
        st = self.encode_to_d2(x)
        st = self.decode_d2_to_d1(st, st['d2'])
        # F0 = full-res 16ch before output conv
        up = self.up2_0(st['d1'])
        fusion = self.latent(st['out_fusion_123'])
        out = (
            self.coefficient_1_0[0, :][None, :, None, None] * fusion
            + self.coefficient_1_0[1, :][None, :, None, None] * up)
        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)
        inp_fusion = torch.cat(
            [out_1.unsqueeze(1), out_2.unsqueeze(1), out_3.unsqueeze(1)], dim=1)
        fused = self.layer_fussion_2(inp_fusion)
        f0 = self.conv_fuss_2(fused)
        if self.skip:
            y0 = self.output(f0) + x
        else:
            y0 = self.output(f0)
        return dict(d2=st['d2'], d1=st['d1'], f0=f0, y0=y0)

    def forward_staged(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through staged decode (identity refine)."""
        st = self.encode_to_d2(x)
        st = self.decode_d2_to_d1(st, st['d2'])
        return self.decode_d1_to_rgb(st, st['d1'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically equivalent to official LLFormer.forward via staged path."""
        return self.forward_staged(x)

    def forward_official_path(self, x: torch.Tensor) -> torch.Tensor:
        """Call parent (byte-level same ops as vendored LLFormer.forward)."""
        return LLFormer.forward(self, x)


def infer_rgb(bridge: LLFormerBridge, x01: torch.Tensor,
              clamp: bool = True) -> torch.Tensor:
    """Eval helper: pad→forward→crop→optional clamp. x01 in [0,1]."""
    xp, h0, w0 = reflect_pad_to_multiple(x01, PAD_MULTIPLE)
    y = bridge(xp)
    y = crop_to_hw(y, h0, w0)
    if clamp:
        y = torch.clamp(y, 0.0, 1.0)
    return y


def load_llformer_bridge(ckpt_path: str = OFFICIAL_LOL_CKPT,
                         device: str = 'cuda',
                         train: bool = False) -> Tuple[LLFormerBridge, Dict]:
    net = LLFormerBridge()
    info = load_into_llformer(net, ckpt_path)
    net.to(device)
    if train:
        net.train()
    else:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    print('[LLFormerBridge] loaded %d tensors from %s (%s)'
          % (info['loaded'], ckpt_path, 'train' if train else 'eval'),
          flush=True)
    return net, info
