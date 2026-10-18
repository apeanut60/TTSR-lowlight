"""Photometric postprocess shares one path with tiled eval."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from model.MainNetEnhance import GlobalIllumHead, RefIllumTransfer  # noqa: E402
from model.TTSREnhance import RefCorrectionHead  # noqa: E402
from train_lolv2syn_illum import configure_illum_only, illum_forward  # noqa: E402
from v2_p2_runtime import (  # noqa: E402
    apply_p2_photometric_postprocess, tiled_p2,
)


def _args():
    return SimpleNamespace(
        tile_size=32, tile_overlap=8,
        query_illum_alpha=0.75, query_gamma_min=0.3, query_gamma_max=1.0,
        query_hist_bins=32,
        stage_mode_lv3='normal', stage_mode_lv2='branch_bypass',
        stage_mode_lv1='normal',
    )


class _Heads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.global_illum_head = GlobalIllumHead()
        self.ref_illum = RefIllumTransfer(pool=4, init_scale=0.05)


class _Wrap(torch.nn.Module):
    def __init__(self, correct=True):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 3, 1)
        self.MainNet = _Heads()
        self.RefCorrection = RefCorrectionHead(4) if correct else None
        self.calls = []
        self.seen_ref = None
        self.seen_low = None
        self.args = SimpleNamespace()
        if self.RefCorrection is not None:
            with torch.no_grad():
                self.RefCorrection.global_conv.bias.fill_(0.2)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, **kwargs):
        return lr, None

    def get_effective_reference(self, ref):
        if ref is None:
            return None
        if self.RefCorrection is None:
            return ref.detach()
        return self.RefCorrection(ref)

    def apply_ref_illum(self, x, low, ref):
        self.calls.append('ref')
        self.seen_ref = None if ref is None else ref.detach().clone()
        self.seen_low = low.detach().clone()
        return self.MainNet.ref_illum(x, low, ref if ref is not None else x)

    def apply_illum_head(self, x, low):
        self.calls.append('illum')
        self.seen_low = low.detach().clone()
        gain, bias, gamma = self.MainNet.global_illum_head(low)
        base = torch.clamp((x + 1.) / 2., 1e-2, 1.)
        y = torch.pow(base, gamma) * 2. - 1.
        return torch.clamp(y * gain + bias, -1, 1)


def test_order_and_single_call_and_corrected_ref():
    torch.manual_seed(0)
    model = _Wrap(correct=True)
    low = torch.rand(1, 3, 48, 40) * 2 - 1
    ref = torch.rand(1, 3, 48, 40) * 2 - 1
    y = apply_p2_photometric_postprocess(model, low, low, ref)
    assert model.calls == ['ref', 'illum']
    assert y.shape == low.shape
    corrected = model.RefCorrection(ref)
    assert torch.allclose(model.seen_ref, corrected)
    assert not torch.allclose(model.seen_ref, ref)
    assert model.seen_low.shape[-2:] == (48, 40)


def test_ref_correction_off_uses_raw_ref():
    model = _Wrap(correct=False)
    low = torch.zeros(1, 3, 16, 16)
    ref = torch.full((1, 3, 16, 16), 0.3)
    apply_p2_photometric_postprocess(model, low, low, ref)
    assert model.calls == ['ref', 'illum']
    assert torch.allclose(model.seen_ref, ref)


def test_ref_none():
    model = _Wrap(correct=True)
    low = torch.zeros(1, 3, 16, 16)
    y = apply_p2_photometric_postprocess(model, low, low, None)
    assert model.seen_ref is None
    assert y.shape == low.shape


def test_train_and_eval_forward_match_and_grads():
    torch.manual_seed(1)
    model = _Wrap(correct=True)
    args = _args()
    low = torch.rand(1, 3, 48, 36) * 2 - 1
    ref = torch.rand(1, 3, 48, 36) * 2 - 1
    model.eval()
    with torch.no_grad():
        y_eval = tiled_p2(model, args, low, low, ref, ref)
    configure_illum_only(model)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters()
              if 'global_illum_head' not in n and 'ref_illum' not in n}
    y_train, _ = illum_forward(model, args, low, ref, live=True)
    assert torch.allclose(y_eval, y_train, atol=1e-6)
    assert model.calls.count('ref') == 2
    assert model.calls.count('illum') == 2
    assert model.seen_low.shape[-2:] == low.shape[-2:]
    loss = F.mse_loss(y_train, torch.zeros_like(y_train))
    loss.backward()
    saw = False
    for name, param in model.named_parameters():
        if 'global_illum_head' in name or 'ref_illum' in name:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            if float(param.grad.abs().sum()) > 0:
                saw = True
        else:
            assert param.grad is None
            assert torch.equal(param, frozen[name])
    assert saw
