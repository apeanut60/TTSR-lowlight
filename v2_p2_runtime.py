"""Frozen TTSR P2 inference for v2 Y0 cache generation.

P2 = M1 texture fusion + lv2 branch_bypass (v1r13) on legacy MainNetEnhance.
Outputs stay in [-1,1] float32 with no GT mean-align / clamp / round.

block-amend CSFI-U-Net uses ``build_args_csfi`` / ``forward_csfi_tiled_backbone``:
pad to /4, run IlluminationEstimator+light-up once on the full image, default
``stage_mode_lv2=normal`` (inject T_lv2 at H/2). Old P2 ckpts still load via
``backbone=mainnet``.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model import TTSREnhance
from model.CSFIUNet import CSFIUNet
from model.query_illum_align import estimate_query_align_maps

CKPT = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt')

P2_CONFIG = dict(
    query_illum_align=True,
    query_illum_alpha=0.75,
    query_hist_beta=0.50,
    texture_fusion=True,
    texture_fusion_lambda=0.25,
    texture_fusion_s_source='a0',
    stage_mode_lv3='normal',
    stage_mode_lv2='branch_bypass',
    stage_mode_lv1='normal',
    tile_size=256,
    tile_overlap=96,
    ref_degrade=False,
)

# Default runtime knobs for the new CSFI Stage-A backbone (not the frozen P2 ckpt).
CSFI_CONFIG = dict(
    query_illum_align=True,
    query_illum_alpha=0.75,
    query_hist_beta=0.50,
    texture_fusion=True,
    texture_fusion_lambda=0.25,
    texture_fusion_s_source='a0',
    stage_mode_lv3='normal',
    stage_mode_lv2='normal',
    stage_mode_lv1='normal',
    tile_size=256,
    tile_overlap=96,
    ref_degrade=False,
)


def build_args():
    """Args for loading the frozen legacy P2 (MainNetEnhance) checkpoint."""
    return SimpleNamespace(
        dataset='data1', dataset_dir='/root/data/datasets/data1',
        data1_camera='Huawei',
        ref_dir='/root/data/datasets/data1/Eval/Huawei/nanobanana_ref',
        ref_degrade=False, eval_ref_degrade=None, ref_gamma=1.0,
        eval_degrade_seed=1234, enhance_mode=True,
        backbone='mainnet',
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
        restoration_blocks='1,1', restoration_heads=1, enable_lightup=True,
        use_global_illum=False, use_ref_illum=False,
        freeze_lte=False, ref_correction=True, ref_correction_feats=16,
        ref_illum_pool=8,
        query_illum_align=True, query_illum_alpha=0.75, query_hist_beta=0.50,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=True, texture_fusion_lambda=0.25,
        texture_fusion_s_source='a0',
        gate_mode=None, gate_shift='', gate_smooth_kernel=31,
        collect_gate_stats=False, collect_residual_decomp=False,
        disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
        stage_mode_lv3='normal', stage_mode_lv2='branch_bypass',
        stage_mode_lv1='normal',
        tile_size=256, tile_overlap=96, tile_window='cosine',
        cpu=False, num_gpu=1, num_workers=0, batch_size=1,
        eval_chroma_gain=1.0, eval_tta=False,
    )


def build_args_csfi():
    """Args for the block-amend CSFI-U-Net Stage-A backbone."""
    args = build_args()
    args.backbone = 'csfi_unet'
    args.num_res_blocks = '8+8+4+2'
    args.stage_mode_lv2 = 'normal'
    args.use_global_illum = False
    args.use_ref_illum = False
    args.enable_lightup = True
    return args


def _pad_to_multiple(x, multiple=4):
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


def load_frozen_p2(device='cuda'):
    args = build_args()
    model = TTSREnhance.TTSREnhance(args).to(device).eval()
    state = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    n = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            n += 1
    model.load_state_dict(cur)
    for p in model.parameters():
        p.requires_grad_(False)
    print('Loaded P2 base weights: %d tensors from %s' % (n, CKPT))
    return model, args


def apply_p2_cfg(model):
    """Apply historical P2 knobs. Call only when explicitly rebuilding a P2 experiment.

    Must **not** be invoked from generic tiled / trainer inference paths.
    """
    for k, v in P2_CONFIG.items():
        if k in ('tile_size', 'tile_overlap', 'ref_degrade'):
            continue
        setattr(model.args, k, v)
    model.args.gate_mode = None
    model.args.disable_T_lv3 = False
    model.args.disable_T_lv2 = False
    model.args.disable_T_lv1 = False


def apply_csfi_cfg(model):
    """Apply CSFI_CONFIG. Explicit historical/helper use only — not during inference."""
    for k, v in CSFI_CONFIG.items():
        if k in ('tile_size', 'tile_overlap', 'ref_degrade'):
            continue
        setattr(model.args, k, v)
    model.args.gate_mode = None
    model.args.disable_T_lv3 = False
    model.args.disable_T_lv2 = False
    model.args.disable_T_lv1 = False
    model.args.stage_mode_lv2 = 'normal'


def tile_spans(length, tile, overlap):
    """Inclusive-exclusive spans along one axis. No negative origins.

    Images shorter than ``tile`` are a single span of the true length.
    The last span is shifted so it stays inside the image when the length
    is not a multiple of the stride.
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
    """Feather only edges that touch another tile. Outer image borders stay 1."""
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


def forward_p2_tiled_backbone(model, args, lr, lrsr, ref, refsr, enable_grad=False):
    """Tiled backbone. No RefIllumTransfer and no GlobalIllumHead.

    Does **not** call ``apply_p2_cfg`` / ``apply_csfi_cfg`` (no silent config
    rewrite). Uses the shared ``model.tiled_infer.tiled_forward`` path.
    Photometric heads are left off here; callers apply them explicitly.
    """
    from model.tiled_infer import tiled_forward

    fwd = {}
    # Only estimate query maps when the *current* args already enable it.
    if (ref is not None and bool(getattr(args, 'query_illum_align', False))
            and float(getattr(args, 'query_illum_alpha', 0.0)) > 0.0):
        g, lut, _ = estimate_query_align_maps(
            lrsr, ref, alpha=args.query_illum_alpha,
            gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
            bins=args.query_hist_bins)
        fwd['query_gamma_override'] = g
        if float(getattr(args, 'query_hist_beta', 0.0)) > 0.0:
            fwd['query_hist_lut_override'] = lut
    # Honour existing stage_mode on args without rewriting them.
    for k in ('stage_mode_lv3', 'stage_mode_lv2', 'stage_mode_lv1'):
        if getattr(args, k, None) is not None:
            fwd[k] = getattr(args, k)

    return tiled_forward(
        model, lr, lrsr, ref, refsr,
        tile_size=int(getattr(args, 'tile_size', 256)),
        overlap=int(getattr(args, 'tile_overlap', 96)),
        enable_grad=enable_grad,
        apply_illum=False,
        apply_ref_illum=False,
        fwd_extra=fwd,
    )


def apply_p2_photometric_postprocess(model, sr, low, ref):
    """RefIllumTransfer, then GlobalIllumHead. One call each.

    The reference is the full-image effective reference (RefCorrection once
    when that head exists). Gradients of these two modules are not blocked.
    """
    ref_eff = model.get_effective_reference(ref)
    y = model.apply_ref_illum(sr, low, ref_eff)
    y = model.apply_illum_head(y, low)
    if not torch.isfinite(y).all():
        raise RuntimeError('photometric postprocess produced a non-finite value')
    return y


@torch.no_grad()
def tiled_p2(model, args, lr, lrsr, ref, refsr):
    """Full-image tiled P2 forward; returns [1,3,H,W] float32 in [-1,1]."""
    sr = forward_p2_tiled_backbone(
        model, args, lr, lrsr, ref, refsr, enable_grad=False)
    return apply_p2_photometric_postprocess(model, sr, lr, ref)


def as_batch(path, device):
    from imageio import imread
    import numpy as np
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    a = im.astype(np.float32) / 127.5 - 1.
    return torch.from_numpy(a.transpose(2, 0, 1))[None].to(device)


def resize_to(t, hw):
    if tuple(t.shape[-2:]) == tuple(hw):
        return t
    return torch.nn.functional.interpolate(
        t, size=tuple(hw), mode='bilinear', align_corners=False)


@torch.no_grad()
def infer_p2_pair(model, args, low_path, nano_path, device):
    """Run P2 on one Low/Nano pair. Nano resized to Low spatial size."""
    lr = as_batch(low_path, device)
    ref = as_batch(nano_path, device)
    ref = resize_to(ref, lr.shape[-2:])
    # TestSet uses LR_sr / Ref_sr = same resolution as LR/Ref for enhance_mode
    return tiled_p2(model, args, lr, lr, ref, ref)
