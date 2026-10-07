"""Frozen Retinexformer prefix/tail with staged H/4 then H/2 injection.

D2 = bottleneck 160ch @ H/4 (no RefTextureAdapter).
D1 = decoder block 0 output 80ch @ H/2.
Zero ΔD ⇒ original Base RGB (tiled, same as B4).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.RetinexRefMainNet import RetinexRefMainNet
from model.V3BFeatureBridge import (
    BASE_H2_CH, TILE_OVERLAP, TILE_SIZE, _feather, _pad_to_4, decode_from_h2,
    original_base_forward, tiled_bridge_decode,
)
from trainer import tile_starts

INJECTION_POINT = 'decoder_h4_bottleneck_then_h2_block0'
BASE_H4_CH = 160
REFINE_SCALES = ('H/4', 'H/2')


def load_frozen_retinex_mainnet(base_ckpt, base_run_dir, device='cuda'):
    """Load only Retinexformer MainNet. No LTE / VGG19 / Trainer.

    V5 never uses TTSR texture transfer or perceptual VGG. The old
    ``load_frozen_n0`` helper constructed TTSREnhance (LTE inits from
    torchvision VGG19) and Trainer (a second VGG19 for unused loss).
    """
    from local_refine_runtime import load_args_from_run

    cfg = load_args_from_run(base_run_dir)
    n_blocks = [int(x) for x in
                str(getattr(cfg, 'retinex_num_blocks', '1,2,2')).split(',')]
    net = RetinexRefMainNet(
        n_feat=getattr(cfg, 'retinex_n_feat', 40),
        num_blocks=n_blocks,
        ref_illum_pool=getattr(cfg, 'ref_illum_pool', 8),
        use_global_illum=False)
    blob = torch.load(base_ckpt, map_location='cpu')
    msd = net.state_dict()
    loaded = 0
    missing = []
    for k in msd:
        pk = 'MainNet.' + k
        if pk in blob and tuple(blob[pk].shape) == tuple(msd[k].shape):
            msd[k] = blob[pk]
            loaded += 1
        elif k.startswith('estimator.') or k.startswith('denoiser.'):
            missing.append(k)
    if missing:
        raise SystemExit('frozen MainNet missing %d keys (first %s)'
                         % (len(missing), missing[:6]))
    net.load_state_dict(msd, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print('[V5] loaded %d MainNet tensors from %s (no VGG/LTE)'
          % (loaded, base_ckpt), flush=True)
    return net


def prefix_to_h4(mainnet, x_m11):
    """X [-1,1] → D2 [B,160,H/4,W/4] (padded) + ctx through bottleneck, no adapter."""
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
    if int(fea.shape[1]) != BASE_H4_CH:
        raise SystemExit('F_h4 channels %d != %d' % (int(fea.shape[1]), BASE_H4_CH))
    ctx = dict(
        fea_encoder0=fea_encoder[0],
        fea_encoder1=fea_encoder[1],
        illu_fea0=illu_fea_list[0],
        illu_fea1=illu_fea_list[1],
        x_c=x_c,
        h0=h0, w0=w0,
        hp=int(x_pad.shape[-2]), wp=int(x_pad.shape[-1]),
    )
    return fea, ctx


def h4_to_h2(mainnet, f_h4, ctx):
    """D2 → D1 after decoder stage 0 (upsample + skip + IGAB)."""
    den = mainnet.denoiser
    up0, fusion0, block0 = den.decoder_layers[0]
    fea = up0(f_h4)
    fea = fusion0(torch.cat([fea, ctx['fea_encoder1']], dim=1))
    fea = block0(fea, ctx['illu_fea1'])
    if int(fea.shape[1]) != BASE_H2_CH:
        raise SystemExit('F_h2 channels %d != %d' % (int(fea.shape[1]), BASE_H2_CH))
    return fea


def decode_from_h4(mainnet, f_h4, ctx):
    """Zero-refine path: D2 → D1 → RGB."""
    f_h2 = h4_to_h2(mainnet, f_h4, ctx)
    return decode_from_h2(mainnet, f_h2, ctx)


def bridge_zero_delta(mainnet, x_m11):
    f_h4, ctx = prefix_to_h4(mainnet, x_m11)
    return decode_from_h4(mainnet, f_h4, ctx)


def crop_img(img, y0, x0, th, tw):
    return img[:, :, y0:y0 + th, x0:x0 + tw]


def v5_decode_tile(mainnet, branch, x_tile, y0_tile, r_tile):
    """One spatial tile. branch.refine_* may be identity at step0."""
    x_pad, _, _ = _pad_to_4(x_tile)
    y_pad, _, _ = _pad_to_4(y0_tile)
    r_pad, _, _ = _pad_to_4(r_tile)
    d2, ctx = prefix_to_h4(mainnet, x_tile)
    d2s, a2, aux2 = branch.refine_at_h4(d2, y_pad, r_pad)
    d1 = h4_to_h2(mainnet, d2s, ctx)
    d1s, a1, aux1 = branch.refine_at_h2(d1, y_pad, r_pad, aux2)
    y = decode_from_h2(mainnet, d1s, ctx)
    aux = dict(d2=d2, d2s=d2s, d1=d1, d1s=d1s, a2=a2, a1=a1)
    aux.update(aux2)
    aux.update({('h2_%s' % k): v for k, v in aux1.items()})
    return y, aux


def tiled_v5_forward(mainnet, branch, x_m11, y0_m11, r_m11,
                     tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
                     collect_aux=False):
    """Match B4 tiled inference; refine H/4 then frozen transition then H/2."""
    _, _, H, W = x_m11.shape
    if y0_m11.shape[-2:] != (H, W) or r_m11.shape[-2:] != (H, W):
        raise SystemExit('Y0/R spatial mismatch with X')
    if H <= tile_size and W <= tile_size:
        y, aux = v5_decode_tile(mainnet, branch, x_m11, y0_m11, r_m11)
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
            sr, aux = v5_decode_tile(mainnet, branch, xt, yt, rt)
            last_aux = aux
            accum[:, :, ys:ys + tile_h, xs:xs + tile_w] += sr * feather
            wacc[:, :, ys:ys + tile_h, xs:xs + tile_w] += feather
    if float(wacc.min()) <= 0:
        raise SystemExit('tiled V5 uncovered pixels')
    y = accum / wacc
    return (y, last_aux) if collect_aux else y
