"""Shared tiled inference for TTSREnhance (CSFI and MainNet).

Does **not** mutate model.args / stage_mode / query_illum / texture_fusion.
Used by trainer validation and independent eval entrypoints.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from model.CSFIUNet import CSFIUNet


def pad_to_multiple(x, multiple=4):
    if x is None:
        return None, (0, 0)
    h, w = x.shape[-2:]
    th = int(math.ceil(h / float(multiple)) * multiple)
    tw = int(math.ceil(w / float(multiple)) * multiple)
    ph, pw = th - h, tw - w
    if ph == 0 and pw == 0:
        return x, (h, w)
    mode = 'reflect' if ph < h and pw < w else 'replicate'
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)


def tile_spans(length, tile, overlap):
    """Inclusive-exclusive spans. No negative origins.

    Supports length < tile (single span of true length), and last-tile shift.
    """
    length = int(length)
    tile = int(tile)
    if length <= 0 or tile <= 0:
        raise ValueError('length and tile must be positive')
    if length <= tile:
        return [(0, length)]
    overlap = max(0, min(int(overlap), tile - 1))
    stride = tile - overlap
    spans = []
    start = 0
    while True:
        if start + tile >= length:
            last = (length - tile, length)
            if not spans or spans[-1] != last:
                spans.append(last)
            break
        spans.append((start, start + tile))
        start += stride
    return spans


def cosine_ramp(overlap, device, dtype):
    overlap = int(overlap)
    if overlap <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    ang = torch.linspace(0, math.pi, overlap, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(ang)


def tile_blend_weight(y0, y1, x0, x1, height, width, overlap, device, dtype):
    """Feather only edges that touch another tile. Outer borders stay 1."""
    th = int(y1 - y0)
    tw = int(x1 - x0)
    if th <= 0 or tw <= 0:
        raise ValueError('empty tile')
    wy = torch.ones(th, device=device, dtype=dtype)
    wx = torch.ones(tw, device=device, dtype=dtype)
    ov_h = min(int(overlap), th)
    ov_w = min(int(overlap), tw)
    if ov_h > 0:
        ramp_h = cosine_ramp(ov_h, device, dtype)
        if y0 > 0:
            wy[:ov_h] = wy[:ov_h] * ramp_h
        if y1 < height:
            wy[-ov_h:] = wy[-ov_h:] * ramp_h.flip(0)
    if ov_w > 0:
        ramp_w = cosine_ramp(ov_w, device, dtype)
        if x0 > 0:
            wx[:ov_w] = wx[:ov_w] * ramp_w
        if x1 < width:
            wx[-ov_w:] = wx[-ov_w:] * ramp_w.flip(0)
    return wy[:, None] * wx[None, :]


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def tiled_forward(
    model,
    lr,
    lrsr,
    ref,
    refsr,
    tile_size=256,
    overlap=96,
    enable_grad=False,
    apply_illum=True,
    apply_ref_illum=True,
    fwd_extra=None,
):
    """Tiled backbone + optional photometric postprocess.

    Args:
        lr/lrsr/ref/refsr: ``[B,3,H,W]``. ``B!=1`` raises.
        Does not rewrite ``model.args``.
    Returns:
        ``sr`` float32 in model native ``[-1,1]`` range, original H/W.
    """
    if lr.dim() != 4:
        raise ValueError('lr must be [B,C,H,W]')
    if int(lr.shape[0]) != 1:
        raise ValueError(
            'tiled_forward currently supports batch_size=1 only, got %d'
            % int(lr.shape[0]))

    net = _unwrap(model)
    use_csfi = isinstance(getattr(net, 'MainNet', None), CSFIUNet)
    tile = int(tile_size)
    overlap = max(0, min(int(overlap), max(tile // 2, 0)))
    fwd_extra = dict(fwd_extra or {})

    # Snapshot args to prove we do not mutate config.
    args_before = None
    if hasattr(net, 'args'):
        args_before = dict(vars(net.args))

    ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with ctx:
        orig_hw = tuple(lr.shape[-2:])
        stem_full = None
        lr_w, lrsr_w, ref_w, refsr_w = lr, lrsr, ref, refsr
        if use_csfi:
            lr_w, _ = pad_to_multiple(lr, 4)
            lrsr_w, _ = pad_to_multiple(lrsr, 4)
            if ref is not None:
                ref_w, _ = pad_to_multiple(ref, 4)
            if refsr is not None:
                refsr_w, _ = pad_to_multiple(refsr, 4)
            tile = int(math.ceil(tile / 4.0) * 4)
            overlap = int(math.floor(overlap / 4.0) * 4)
            stem_full = net.MainNet.encode_low_stem(lr_w)

        _, _, height, width = lr_w.shape
        spans_y = tile_spans(height, tile, overlap)
        spans_x = tile_spans(width, tile, overlap)

        def one(y0, y1, x0, x1):
            kw = dict(fwd_extra)
            kw.update(apply_illum=False, apply_ref_illum=False)
            if stem_full is not None:
                kw['stem'] = CSFIUNet.crop_stem(stem_full, y0, y1, x0, x1)
            out = model(
                lr=lr_w[:, :, y0:y1, x0:x1],
                lrsr=lrsr_w[:, :, y0:y1, x0:x1],
                ref=None if ref_w is None else ref_w[:, :, y0:y1, x0:x1],
                refsr=None if refsr_w is None else refsr_w[:, :, y0:y1, x0:x1],
                **kw)
            return out[0]

        if len(spans_y) == 1 and len(spans_x) == 1:
            y0, y1 = spans_y[0]
            x0, x1 = spans_x[0]
            sr = one(y0, y1, x0, x1)
            if (y0, x0) != (0, 0) or (y1, x1) != (height, width):
                acc = torch.zeros(
                    1, 3, height, width, device=lr.device, dtype=torch.float32)
                acc[:, :, y0:y1, x0:x1] = sr.float()
                sr = acc
        else:
            acc = torch.zeros(
                1, 3, height, width, device=lr.device, dtype=torch.float32)
            wt = torch.zeros(
                1, 1, height, width, device=lr.device, dtype=torch.float32)
            for y0, y1 in spans_y:
                for x0, x1 in spans_x:
                    pred = one(y0, y1, x0, x1).float()
                    if pred.shape[-2:] != (y1 - y0, x1 - x0):
                        raise RuntimeError(
                            'tile pred shape %s != span %s'
                            % (tuple(pred.shape[-2:]), (y1 - y0, x1 - x0)))
                    weight = tile_blend_weight(
                        y0, y1, x0, x1, height, width, overlap,
                        lr.device, torch.float32)
                    acc[:, :, y0:y1, x0:x1] += pred * weight
                    wt[:, :, y0:y1, x0:x1] += weight
            if int((wt <= 0).sum()) != 0:
                raise RuntimeError(
                    'tiled forward left pixels with weight_sum <= 0')
            sr = acc / wt

        sr = sr[..., :orig_hw[0], :orig_hw[1]]
        if not torch.isfinite(sr).all():
            raise RuntimeError('tiled forward produced non-finite values')

        # Photometric postprocess once on the full image (same RefCorrection).
        if apply_ref_illum or apply_illum:
            low = lr
            ref_eff = None
            if ref is not None and hasattr(net, 'get_effective_reference'):
                ref_eff = net.get_effective_reference(ref)
            if apply_ref_illum and hasattr(net, 'apply_ref_illum'):
                sr = net.apply_ref_illum(sr, low, ref_eff)
            if apply_illum and hasattr(net, 'apply_illum_head'):
                sr = net.apply_illum_head(sr, low)

    if args_before is not None:
        after = dict(vars(net.args))
        if after != args_before:
            raise RuntimeError(
                'tiled_forward mutated model.args; before/after differ')

    return sr
