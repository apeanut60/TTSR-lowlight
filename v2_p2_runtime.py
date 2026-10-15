"""Frozen TTSR P2 inference for v2 Y0 cache generation.

P2 = M1 texture fusion + lv2 branch_bypass (v1r13). Outputs stay in [-1,1]
float32 with no GT mean-align / clamp / round.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch

from model import TTSREnhance
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


def build_args():
    return SimpleNamespace(
        dataset='data1', dataset_dir='/root/data/datasets/data1',
        data1_camera='Huawei',
        ref_dir='/root/data/datasets/data1/Eval/Huawei/nanobanana_ref',
        ref_degrade=False, eval_ref_degrade=None, ref_gamma=1.0,
        eval_degrade_seed=1234, enhance_mode=True,
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
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
    for k, v in P2_CONFIG.items():
        if k in ('tile_size', 'tile_overlap', 'ref_degrade'):
            continue
        setattr(model.args, k, v)
    model.args.gate_mode = None
    model.args.disable_T_lv3 = False
    model.args.disable_T_lv2 = False
    model.args.disable_T_lv1 = False


@torch.no_grad()
def tiled_p2(model, args, lr, lrsr, ref, refsr):
    """Full-image tiled P2 forward; returns [1,3,H,W] float32 in [-1,1]."""
    apply_p2_cfg(model)
    tile = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile // 2))
    g, lut, _ = estimate_query_align_maps(
        lrsr, ref, alpha=args.query_illum_alpha,
        gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
        bins=args.query_hist_bins)
    fwd = dict(
        query_gamma_override=g, query_hist_lut_override=lut,
        stage_mode_lv3=args.stage_mode_lv3,
        stage_mode_lv2=args.stage_mode_lv2,
        stage_mode_lv1=args.stage_mode_lv1,
    )
    _, _, H, W = lr.shape

    def one(y0, y1, x0, x1):
        return model(
            lr=lr[:, :, y0:y1, x0:x1],
            lrsr=lrsr[:, :, y0:y1, x0:x1],
            ref=ref[:, :, y0:y1, x0:x1],
            refsr=refsr[:, :, y0:y1, x0:x1],
            apply_illum=False, apply_ref_illum=False, **fwd)[0]

    if H <= tile and W <= tile:
        sr = one(0, H, 0, W)
    else:
        stride = max(1, tile - overlap)
        nh = max(1, (H - overlap + stride - 1) // stride)
        nw = max(1, (W - overlap + stride - 1) // stride)
        acc = torch.zeros(1, 3, H, W, device=lr.device)
        wt = torch.zeros(1, 1, H, W, device=lr.device)
        my = torch.ones(tile, device=lr.device)
        mx = torch.ones(tile, device=lr.device)
        if overlap > 0:
            ramp = 0.5 - 0.5 * torch.cos(
                torch.linspace(0, math.pi, overlap, device=lr.device))
            my[:overlap] = ramp
            my[-overlap:] = ramp.flip(0)
            mx[:overlap] = ramp
            mx[-overlap:] = ramp.flip(0)
        feather = (my[:, None] * mx[None, :]).view(1, 1, tile, tile)
        for ih in range(nh):
            y0 = min(ih * stride, H - tile)
            for iw in range(nw):
                x0 = min(iw * stride, W - tile)
                t = one(y0, y0 + tile, x0, x0 + tile)
                acc[:, :, y0:y0 + tile, x0:x0 + tile] += t * feather
                wt[:, :, y0:y0 + tile, x0:x0 + tile] += feather
        sr = acc / wt.clamp(min=1e-8)
    sr = model.apply_ref_illum(sr, lr, ref)
    sr = model.apply_illum_head(sr, lr)
    return sr


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
