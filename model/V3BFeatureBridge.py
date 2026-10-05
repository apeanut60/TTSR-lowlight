"""V3-B.4 frozen Retinexformer prefix/tail at decoder H/2.

Injection: after decoder block 0 (H/4→H/2 + skip + IGAB), before H/2→H upsample.
No RefTextureAdapter, no H/4, no forward hooks.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from trainer import tile_starts

INJECTION_POINT = 'decoder_h2_after_block0'
BASE_H2_CH = 80
TILE_SIZE = 256
TILE_OVERLAP = 96


def _pad_to_4(x):
    h0, w0 = x.shape[-2:]
    ph, pw = (-h0) % 4, (-w0) % 4
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
    return x, h0, w0


def prefix_to_h2(mainnet, x_m11):
    """X in [-1,1] → F_h2 [B,80,H/2,W/2] (padded) + tail_ctx.

    Does **not** run RefTextureAdapter. Matches N0 with use_texture=False
    through decoder stage 0 inclusive.
    """
    x_pad, h0, w0 = _pad_to_4(x_m11)
    x01 = (x_pad + 1.0) * 0.5
    illu_fea, illu_map = mainnet.estimator(x01)
    x_c = x01 * illu_map + x01
    den = mainnet.denoiser
    fea = den.embedding(x_c)
    fea_encoder = []
    illu_fea_list = []
    for (igab, fea_down, illu_down) in den.encoder_layers:
        fea = igab(fea, illu_fea)
        illu_fea_list.append(illu_fea)
        fea_encoder.append(fea)
        fea = fea_down(fea)
        illu_fea = illu_down(illu_fea)
    fea = den.bottleneck(fea, illu_fea)
    up0, fusion0, block0 = den.decoder_layers[0]
    fea = up0(fea)
    fea = fusion0(torch.cat([fea, fea_encoder[1]], dim=1))
    fea = block0(fea, illu_fea_list[1])
    if int(fea.shape[1]) != BASE_H2_CH:
        raise SystemExit('F_h2 channels %d != %d' % (int(fea.shape[1]), BASE_H2_CH))
    ctx = dict(
        fea_encoder0=fea_encoder[0],
        illu_fea0=illu_fea_list[0],
        x_c=x_c,
        h0=h0, w0=w0,
        hp=int(x_pad.shape[-2]), wp=int(x_pad.shape[-1]),
    )
    return fea, ctx


def decode_from_h2(mainnet, f_h2, ctx):
    """F_h2 + tail_ctx → RGB [-1,1] cropped to original H0,W0."""
    den = mainnet.denoiser
    up1, fusion1, block1 = den.decoder_layers[1]
    fea = up1(f_h2)
    fea = fusion1(torch.cat([fea, ctx['fea_encoder0']], dim=1))
    fea = block1(fea, ctx['illu_fea0'])
    out01 = den.mapping(fea) + ctx['x_c']
    sr = out01 * 2.0 - 1.0
    return sr[..., :ctx['h0'], :ctx['w0']]


def _feather(tile_h, tile_w, ov_h, ov_w, first_h, last_h, first_w, last_w, device):
    def mask(n, ov, first, last):
        m = torch.ones(n, device=device)
        k = min(int(ov), n)
        if k <= 0:
            return m
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, k, device=device))
        if k == 1:
            ramp = torch.full((1,), 0.5, device=device)
        if not first:
            m[:k] = ramp
        if not last:
            m[-k:] = ramp.flip(0)
        return m
    return (mask(tile_h, ov_h, first_h, last_h)[:, None]
            * mask(tile_w, ov_w, first_w, last_w)[None, :]).view(1, 1, tile_h, tile_w)


def crop_ft(f0, t, y0, x0, th, tw):
    """Crop matcher F0/T (H/2 of full image) to a pixel tile."""
    ys, xs = y0 // 2, x0 // 2
    ye, xe = (y0 + th) // 2, (x0 + tw) // 2
    return f0[:, :, ys:ye, xs:xe], t[:, :, ys:ye, xs:xe]


def original_base_forward(n0_model, x_m11):
    """Frozen N0 RGB [-1,1], no reference (matches cache generation flags)."""
    return n0_model(
        lr=x_m11, lrsr=x_m11, ref=x_m11, refsr=x_m11,
        use_reference=False, apply_illum=False, apply_ref_illum=False)[0]


def bridge_zero_delta(mainnet, x_m11):
    f_h2, ctx = prefix_to_h2(mainnet, x_m11)
    return decode_from_h2(mainnet, f_h2, ctx)


def tiled_bridge_decode(mainnet, x_m11, delta_fn=None,
                        tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Match N0 tiled inference; delta_fn(y0,x0,th,tw,F_h2)->ΔF or None."""
    _, _, H, W = x_m11.shape
    if H <= tile_size and W <= tile_size:
        f_h2, ctx = prefix_to_h2(mainnet, x_m11)
        dlt = None if delta_fn is None else delta_fn(0, 0, H, W, f_h2)
        if dlt is not None:
            if dlt.shape != f_h2.shape:
                raise SystemExit('ΔF shape %s != F_h2 %s'
                                 % (tuple(dlt.shape), tuple(f_h2.shape)))
            f_h2 = f_h2 + dlt
        return decode_from_h2(mainnet, f_h2, ctx)

    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    ov_h = max(0, min(overlap, tile_h // 2))
    ov_w = max(0, min(overlap, tile_w // 2))
    starts_h = tile_starts(H, tile_h, max(1, tile_h - ov_h))
    starts_w = tile_starts(W, tile_w, max(1, tile_w - ov_w))
    accum = x_m11.new_zeros(1, 3, H, W)
    wacc = x_m11.new_zeros(1, 1, H, W)
    n_h, n_w = len(starts_h), len(starts_w)
    for i_h, y0 in enumerate(starts_h):
        y1 = y0 + tile_h
        for i_w, x0 in enumerate(starts_w):
            x1 = x0 + tile_w
            feather = _feather(
                tile_h, tile_w, ov_h, ov_w,
                i_h == 0, i_h == n_h - 1, i_w == 0, i_w == n_w - 1,
                x_m11.device)
            tile = x_m11[:, :, y0:y1, x0:x1]
            f_h2, ctx = prefix_to_h2(mainnet, tile)
            dlt = None if delta_fn is None else delta_fn(y0, x0, tile_h, tile_w, f_h2)
            if dlt is not None:
                if dlt.shape != f_h2.shape:
                    raise SystemExit('ΔF tile shape %s != F_h2 %s'
                                     % (tuple(dlt.shape), tuple(f_h2.shape)))
                f_h2 = f_h2 + dlt
            sr = decode_from_h2(mainnet, f_h2, ctx)
            accum[:, :, y0:y1, x0:x1] += sr * feather
            wacc[:, :, y0:y1, x0:x1] += feather
    if float(wacc.min()) <= 0:
        raise SystemExit('tiled bridge uncovered pixels')
    return accum / wacc
