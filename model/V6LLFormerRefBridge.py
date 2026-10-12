"""Frozen LLFormerBridge + V6 Ref refine at D2 then D1.

Dataset tensors are [-1,1]; LLFormer internals are [0,1].
Zero ΔD ⇒ Y equals frozen Base (after m11↔01 conversion).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.LLFormerBridge import (D1_CHANNELS, D2_CHANNELS, LLFormerBridge,
                                  PAD_MULTIPLE, crop_to_hw,
                                  reflect_pad_to_multiple)
from model.V3BFeatureBridge import TILE_OVERLAP, TILE_SIZE, _feather
from trainer import tile_starts

INJECTION_POINT = 'llformer_d2_then_d1'
REFINE_SCALES = ('H/4', 'H/2')
CANONICAL_BASE = (
    '/root/data/experiments/llformer_lolv2real/canonical/'
    'llformer_lolv2real_base_E110.pth')


def m11_to_01(x):
    return (x + 1.0) * 0.5


def o1_to_m11(x):
    return x * 2.0 - 1.0


def load_frozen_llformer_base(ckpt_path: str = CANONICAL_BASE,
                              device: str = 'cuda') -> LLFormerBridge:
    blob = torch.load(ckpt_path, map_location='cpu')
    sd = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
    net = LLFormerBridge()
    net.load_state_dict(sd, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print('[V6] frozen LLFormer Base from %s' % ckpt_path, flush=True)
    return net


def crop_img(img, y0, x0, th, tw):
    return img[:, :, y0:y0 + th, x0:x0 + tw]


def base_decode_tile(base: LLFormerBridge, x_m11: torch.Tensor) -> torch.Tensor:
    """One tile, Base only (no Ref). [-1,1] → [-1,1]."""
    x01 = m11_to_01(x_m11)
    xp, h0, w0 = reflect_pad_to_multiple(x01, PAD_MULTIPLE)
    y01 = base(xp)
    return o1_to_m11(crop_to_hw(y01, h0, w0))


def base_forward_m11(base: LLFormerBridge, x_m11: torch.Tensor,
                     tile_size=TILE_SIZE, overlap=TILE_OVERLAP) -> torch.Tensor:
    """Frozen Base with the same tiling/feather as V6 Ref (step0 identity)."""
    _, _, H, W = x_m11.shape
    if H <= tile_size and W <= tile_size:
        return base_decode_tile(base, x_m11)

    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    ov_h = max(0, min(overlap, tile_h // 2))
    ov_w = max(0, min(overlap, tile_w // 2))
    starts_h = tile_starts(H, tile_h, max(1, tile_h - ov_h))
    starts_w = tile_starts(W, tile_w, max(1, tile_w - ov_w))
    accum = x_m11.new_zeros(1, 3, H, W)
    wacc = x_m11.new_zeros(1, 1, H, W)
    n_h, n_w = len(starts_h), len(starts_w)
    for i_h, ys in enumerate(starts_h):
        for i_w, xs in enumerate(starts_w):
            feather = _feather(
                tile_h, tile_w, ov_h, ov_w,
                i_h == 0, i_h == n_h - 1, i_w == 0, i_w == n_w - 1,
                x_m11.device)
            xt = crop_img(x_m11, ys, xs, tile_h, tile_w)
            sr = base_decode_tile(base, xt)
            accum[:, :, ys:ys + tile_h, xs:xs + tile_w] += sr * feather
            wacc[:, :, ys:ys + tile_h, xs:xs + tile_w] += feather
    if float(wacc.min()) <= 0:
        raise SystemExit('tiled Base uncovered pixels')
    return accum / wacc


def v6_decode_tile(base: LLFormerBridge, branch, x_m11, y0_m11, r_m11):
    """One tile. x/y0/r in [-1,1], same H×W. Returns Y in [-1,1]."""
    x01 = m11_to_01(x_m11)
    xp, h0, w0 = reflect_pad_to_multiple(x01, PAD_MULTIPLE)
    # pad Y0/R with same spatial pad so match grids align with padded D2/D1
    y01 = m11_to_01(y0_m11)
    r01 = m11_to_01(r_m11)
    yp, _, _ = reflect_pad_to_multiple(y01, PAD_MULTIPLE)
    rp, _, _ = reflect_pad_to_multiple(r01, PAD_MULTIPLE)
    # MatchEncoder expects m11 via V6Model._rgb01
    y_pad_m11 = o1_to_m11(yp)
    r_pad_m11 = o1_to_m11(rp)

    st = base.encode_to_d2(xp)
    d2 = st['d2']
    if int(d2.shape[1]) != D2_CHANNELS:
        raise SystemExit('D2 ch %d' % int(d2.shape[1]))
    d2s, a2, aux2 = branch.refine_at_h4(d2, y_pad_m11, r_pad_m11)
    st = base.decode_d2_to_d1(st, d2s)
    d1 = st['d1']
    if int(d1.shape[1]) != D1_CHANNELS:
        raise SystemExit('D1 ch %d' % int(d1.shape[1]))
    d1s, a1, aux1 = branch.refine_at_h2(d1, y_pad_m11, r_pad_m11, aux2)
    y01 = base.decode_d1_to_rgb(st, d1s)
    y01 = crop_to_hw(y01, h0, w0)
    aux = dict(d2=d2, d2s=d2s, d1=d1, d1s=d1s, a2=a2, a1=a1)
    aux.update(aux2)
    aux.update({('h2_%s' % k): v for k, v in aux1.items()})
    return o1_to_m11(y01), aux


def tiled_v6_forward(base, branch, x_m11, y0_m11, r_m11,
                     tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
                     collect_aux=False):
    """Match Base tiling; LLFormer pad×16 inside each tile."""
    _, _, H, W = x_m11.shape
    if y0_m11.shape[-2:] != (H, W) or r_m11.shape[-2:] != (H, W):
        raise SystemExit('Y0/R spatial mismatch with X')
    if H <= tile_size and W <= tile_size:
        y, aux = v6_decode_tile(base, branch, x_m11, y0_m11, r_m11)
        return (y, aux) if collect_aux else y

    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    ov_h = max(0, min(overlap, tile_h // 2))
    ov_w = max(0, min(overlap, tile_w // 2))
    starts_h = tile_starts(H, tile_h, max(1, tile_h - ov_h))
    starts_w = tile_starts(W, tile_w, max(1, tile_w - ov_w))
    accum = x_m11.new_zeros(1, 3, H, W)
    wacc = x_m11.new_zeros(1, 1, H, W)
    n_h, n_w = len(starts_h), len(starts_w)
    last_aux = None
    for i_h, ys in enumerate(starts_h):
        for i_w, xs in enumerate(starts_w):
            feather = _feather(
                tile_h, tile_w, ov_h, ov_w,
                i_h == 0, i_h == n_h - 1, i_w == 0, i_w == n_w - 1,
                x_m11.device)
            xt = crop_img(x_m11, ys, xs, tile_h, tile_w)
            yt = crop_img(y0_m11, ys, xs, tile_h, tile_w)
            rt = crop_img(r_m11, ys, xs, tile_h, tile_w)
            sr, aux = v6_decode_tile(base, branch, xt, yt, rt)
            last_aux = aux
            accum[:, :, ys:ys + tile_h, xs:xs + tile_w] += sr * feather
            wacc[:, :, ys:ys + tile_h, xs:xs + tile_w] += feather
    if float(wacc.min()) <= 0:
        raise SystemExit('tiled V6 uncovered pixels')
    y = accum / wacc
    return (y, last_aux) if collect_aux else y
