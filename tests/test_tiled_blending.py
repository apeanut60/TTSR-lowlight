"""Tile spans and blend weights. No GT, no network."""
from __future__ import annotations

import os
import sys

import torch
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from v2_p2_runtime import (  # noqa: E402
    P2_CONFIG, forward_p2_tiled_backbone, tile_blend_weight, tile_spans,
)


def _args(tile, overlap):
    return SimpleNamespace(
        tile_size=tile, tile_overlap=overlap,
        query_illum_alpha=0.75, query_gamma_min=0.3, query_gamma_max=1.0,
        query_hist_bins=32,
        stage_mode_lv3='normal', stage_mode_lv2='branch_bypass',
        stage_mode_lv1='normal',
    )


class _Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, **kwargs):
        return lr, None

    def get_effective_reference(self, ref):
        return ref

    def apply_ref_illum(self, x, low, ref):
        return x

    def apply_illum_head(self, x, low):
        return x


def _legacy_outer_taper(height, width, tile, overlap):
    """Old feather: cosine ramp on every tile edge, including the image border."""
    if height <= tile and width <= tile:
        return torch.ones(1, 1, height, width)
    stride = max(1, tile - overlap)
    nh = max(1, (height - overlap + stride - 1) // stride)
    nw = max(1, (width - overlap + stride - 1) // stride)
    wt = torch.zeros(1, 1, height, width)
    my = torch.ones(tile)
    mx = torch.ones(tile)
    if overlap > 0:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap))
        my[:overlap] = ramp
        my[-overlap:] = ramp.flip(0)
        mx[:overlap] = ramp
        mx[-overlap:] = ramp.flip(0)
    feather = (my[:, None] * mx[None, :]).view(1, 1, tile, tile)
    for ih in range(nh):
        y0 = min(ih * stride, height - tile)
        for iw in range(nw):
            x0 = min(iw * stride, width - tile)
            wt[:, :, y0:y0 + tile, x0:x0 + tile] += feather
    return wt


def test_p2_config_unchanged():
    assert P2_CONFIG == dict(
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


def test_spans_cover_edges_and_stay_inside():
    cases = [
        (10, 32, 8),
        (32, 32, 0),
        (32, 32, 8),
        (40, 32, 8),
        (100, 32, 0),
        (100, 32, 8),
        (97, 32, 7),
        (255, 64, 16),
        (257, 64, 16),
    ]
    for length, tile, overlap in cases:
        spans = tile_spans(length, tile, overlap)
        assert spans[0][0] == 0
        assert spans[-1][1] == length
        for y0, y1 in spans:
            assert 0 <= y0 < y1 <= length
            assert y1 - y0 == min(tile, length)


def test_border_weight_positive_and_legacy_border_is_zero():
    height, width, tile, overlap = 80, 96, 32, 8
    spans_y = tile_spans(height, tile, overlap)
    spans_x = tile_spans(width, tile, overlap)
    wt = torch.zeros(1, 1, height, width)
    for y0, y1 in spans_y:
        for x0, x1 in spans_x:
            w = tile_blend_weight(y0, y1, x0, x1, height, width, overlap, 'cpu', torch.float32)
            wt[:, :, y0:y1, x0:x1] += w
    assert int((wt <= 0).sum()) == 0
    assert float(wt[0, 0, 0, 0]) > 0
    assert float(wt[0, 0, -1, -1]) > 0
    legacy = _legacy_outer_taper(height, width, tile, overlap)
    assert float(legacy[0, 0, 0, 0]) == 0.0


def test_constant_image_survives_blend():
    model = _Identity()
    shapes = [
        (8, 8, 32, 8),
        (40, 18, 32, 8),
        (18, 50, 32, 0),
        (70, 65, 32, 8),
        (31, 33, 16, 5),
        (64, 64, 64, 0),
        (63, 65, 32, 9),
    ]
    for h, w, tile, overlap in shapes:
        img = torch.full((1, 3, h, w), 0.25)
        out = forward_p2_tiled_backbone(
            model, _args(tile, overlap), img, img, img, img, enable_grad=False)
        assert out.shape == img.shape
        assert torch.isfinite(out).all()
        assert torch.allclose(out, img, atol=1e-6)


def test_overlap_zero_and_partial_axis():
    model = _Identity()
    img = torch.arange(3 * 20 * 50, dtype=torch.float32).reshape(1, 3, 20, 50) / 1000 - 0.2
    out = forward_p2_tiled_backbone(
        model, _args(16, 0), img, img, img, img)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
