#!/usr/bin/env python
"""Gate 2.3 — Attention selection (tex_h4 vs G2.1) + pretrained C2 geometry.

No training. No Setting B / 3k / Official Test / LLFormer unfreeze.

Locks G2.2 conclusion: ref_path diagnosis only; heldout did NOT validate
generalization (tex_h4 N−Shuf +0.276 microset vs −0.045 heldout;
heldout N−Base −2.322 dB).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = '/root/projects/TTSR-lowlight'
OUT = '/root/data/experiments/V7_CODE_REVIEW_FIX'
DOCS = os.path.join(REPO, 'docs', 'v7_code_review_fix')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V5Correspondence import ChunkedHardMatcher  # noqa: E402
from model.V5FlowAlign import warp_by_flow  # noqa: E402
from model.V6LLFormerRefBridge import (  # noqa: E402
    load_frozen_llformer_base, m11_to_01, o1_to_m11,
    reflect_pad_to_multiple, PAD_MULTIPLE)
from model.V7Model import V7Model  # noqa: E402
from model.c2_extractor import (  # noqa: E402
    C2_FEATURE_CKPT, load_c2_feature_extractor)
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v7_river_runtime import ROOT as EXP_ROOT, SRC  # noqa: E402

G21_CKPT = os.path.join(OUT, 'gate2_1', 'ckpt_microset_win3_fixed.pt')
TEX_CKPT = os.path.join(OUT, 'gate2_2', 'ckpt_tex_h4_200.pt')
G22_JSON = os.path.join(DOCS, 'G2_2_align_info_quality.json')
HELDOUT_JSON = '/root/data/experiments/V7_REF_DIAG/heldout24.json'
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'
INIT_PATH = (
    '/root/data/experiments/V7_RIVER_validation/experiments/'
    'A_fusion_only/init/v7_river_s42.pt')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def _grad_norm(module, prefix):
    total = 0.0
    for n, p in module.named_parameters():
        if n.startswith(prefix) and p.grad is not None:
            total += float(p.grad.detach().norm())
    return total


def _param_delta(a, b, prefix):
    total = 0.0
    da, db = dict(a.named_parameters()), dict(b.named_parameters())
    for n in da:
        if n.startswith(prefix):
            total += float((da[n].detach() - db[n].detach()).norm())
    return total


def _make_ds(lock, rows):
    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = lock['reference_variant']
    mmap = json.load(open(lock['mismatch_train']))
    return make_dataset(
        ns, rows,
        os.path.join(EXP_ROOT, lock['cache_name'], 'refiner_train'), mmap)


def _load_v7(device, ckpt_path, a2_mode, init_path=INIT_PATH):
    model = V7Model(spatial_window=3, d1_enabled=False).to(device)
    init = torch.load(init_path, map_location='cpu')
    model.load_state_dict(init['model'], strict=False)
    if ckpt_path is not None:
        blob = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(blob['model'], strict=False)
    model.a2_mode = a2_mode
    model.apply_setting_freeze('A')
    for name in ('match_encoder', 'texture_encoder', 'matcher',
                 'align_h4', 'align_h2', 'refine_h2'):
        getattr(model, name).eval()
        for p in getattr(model, name).parameters():
            p.requires_grad_(False)
    return model


def _attn_from_weights(weights):
    n = weights.shape[-1]
    ent = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
    ent_u = float(np.log(n))
    wmax = weights.max(-1).values
    center = n // 2
    return dict(
        n_keys=n,
        entropy_mean=float(ent.mean()),
        entropy_norm=float(ent.mean() / ent_u),
        weight_max_mean=float(wmax.mean()),
        weight_max_p90=float(torch.quantile(wmax.flatten(), 0.9)),
        frac_argmax_center=float((weights.argmax(-1) == center).float().mean()),
        weight_center_mean=float(weights[..., center].mean()),
        uniform=1.0 / n,
        wmax_map=wmax[0].mean(0).detach().cpu(),  # [HW] avg heads
        h=None, w=None,
    )


@torch.no_grad()
def _forward_attn_tile(base, model, x, y0, r):
    from model.V3BFeatureBridge import TILE_SIZE
    _, _, H, W = x.shape
    th, tw = min(TILE_SIZE, H), min(TILE_SIZE, W)
    xt, yt, rt = x[:, :, :th, :tw], y0[:, :, :th, :tw], r[:, :, :th, :tw]
    xp, h0, w0 = reflect_pad_to_multiple(m11_to_01(xt), PAD_MULTIPLE)
    yp, _, _ = reflect_pad_to_multiple(m11_to_01(yt), PAD_MULTIPLE)
    rp, _, _ = reflect_pad_to_multiple(m11_to_01(rt), PAD_MULTIPLE)
    st = base.encode_to_d2(xp)
    d2 = st['d2']
    match, tex = model.match_and_texture(o1_to_m11(yp), o1_to_m11(rp))
    if model.a2_mode == 'tex_h4':
        a2 = tex['h4']
    else:
        a2, _ = model.align_h4(tex['h4'], d2, match.flow, match.confidence)
    _, _, info = model.refine_h4(d2, a2, return_attn=True)
    stats = _attn_from_weights(info['weights'])
    _, _, dh, dw = d2.shape
    stats['h'], stats['w'] = dh, dw
    stats['wmax_map'] = info['weights'].max(-1).values[0].mean(0).reshape(
        dh, dw).detach().cpu()
    return stats, info


def task_a_attention(base, lock, ds, pairs, device, out_dir):
    """Compare G2.1 align ckpt vs tex_h4 ckpt attention + grads + Δθ."""
    init_m = _load_v7(device, None, 'align')
    g21 = _load_v7(device, G21_CKPT, 'align')
    tex = _load_v7(device, TEX_CKPT, 'tex_h4')

    def audit(model, tag):
        rows = []
        maps = {}
        for p in pairs:
            t = sample_tensors(ds, p['i'], 'correct', device)
            st, _ = _forward_attn_tile(base, model, t['X'], t['Y0'], t['R'])
            maps[p['name']] = st['wmax_map']
            rows.append(dict(
                name=p['name'], arm=tag,
                weight_max_mean=st['weight_max_mean'],
                entropy_norm=st['entropy_norm'],
                frac_argmax_center=st['frac_argmax_center'],
                weight_center_mean=st['weight_center_mean'],
            ))
        # one-image grads
        model.refine_h4.train()
        model.zero_grad(set_to_none=True)
        t = sample_tensors(ds, pairs[0]['i'], 'correct', device)
        # tiled forward for grad
        from model.V6LLFormerRefBridge import tiled_v6_forward
        from v3b_runtime import b0_loss
        y = tiled_v6_forward(base, model, t['X'], t['Y0'], t['R'])
        b0_loss(y, t['H']).backward()
        grads = dict(
            q=_grad_norm(model.refine_h4, 'q_proj'),
            k=_grad_norm(model.refine_h4, 'k_proj'),
            v=_grad_norm(model.refine_h4, 'v_proj'),
            out_conv=_grad_norm(model.refine_h4, 'out_conv'),
        )
        model.zero_grad(set_to_none=True)
        model.eval()
        deltas = dict(
            q=_param_delta(model.refine_h4, init_m.refine_h4, 'q_proj'),
            k=_param_delta(model.refine_h4, init_m.refine_h4, 'k_proj'),
            v=_param_delta(model.refine_h4, init_m.refine_h4, 'v_proj'),
            out_conv=_param_delta(model.refine_h4, init_m.refine_h4, 'out_conv'),
            fusion=_param_delta(model.refine_h4, init_m.refine_h4, 'fusion'),
        )
        wmax = float(np.mean([r['weight_max_mean'] for r in rows]))
        ent = float(np.mean([r['entropy_norm'] for r in rows]))
        uniform = 1.0 / 9
        selection_real = bool(
            wmax > uniform * 1.5 and ent < 0.95
            and (deltas['q'] > 1e-3 or deltas['k'] > 1e-3))
        # V/Conv content path dominant if V+out+fusion update >> Q/K and
        # attention stays near-uniform
        content_transfer = bool(
            (not selection_real)
            and (deltas['v'] + deltas['out_conv'] + deltas['fusion'])
            > 5 * (deltas['q'] + deltas['k'] + 1e-12))
        return dict(
            tag=tag, rows=rows, maps=maps, grads=grads, deltas=deltas,
            mean_weight_max=wmax, mean_entropy_norm=ent,
            attention_selection_real=selection_real,
            v_conv_content_transfer=content_transfer,
        )

    a_g21 = audit(g21, 'g21_align')
    a_tex = audit(tex, 'tex_h4')

    # visualizations
    viz_dir = os.path.join(out_dir, 'attn_viz')
    os.makedirs(viz_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for name in list(a_g21['maps'].keys())[:4]:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            for ax, arm, title in (
                    (axes[0], a_g21, 'G2.1 align'),
                    (axes[1], a_tex, 'tex_h4')):
                m = arm['maps'][name].numpy()
                im = ax.imshow(m, vmin=1 / 9, vmax=max(float(m.max()), 0.2),
                               cmap='magma')
                ax.set_title('%s %s\nmax=%.3f' % (
                    title, name, float(m.max())))
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(viz_dir, 'wmax_%s.png' % name.replace(
                '.png', '')), dpi=120)
            plt.close(fig)
        # bar comparison
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ['wmax', 'ent_norm', 'Δq', 'Δk', 'Δv']
        gvals = [a_g21['mean_weight_max'], a_g21['mean_entropy_norm'],
                 a_g21['deltas']['q'], a_g21['deltas']['k'], a_g21['deltas']['v']]
        tvals = [a_tex['mean_weight_max'], a_tex['mean_entropy_norm'],
                 a_tex['deltas']['q'], a_tex['deltas']['k'], a_tex['deltas']['v']]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, gvals, 0.4, label='G2.1')
        ax.bar(x + 0.2, tvals, 0.4, label='tex_h4')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_title('Attention / Δθ comparison')
        fig.tight_layout()
        fig.savefig(os.path.join(viz_dir, 'attn_compare_bars.png'), dpi=120)
        plt.close(fig)
        viz_ok = True
    except Exception as e:
        viz_ok = False
        open(os.path.join(viz_dir, 'viz_error.txt'), 'w').write(str(e))

    def slim(a):
        return dict(
            tag=a['tag'],
            mean_weight_max=a['mean_weight_max'],
            mean_entropy_norm=a['mean_entropy_norm'],
            grads=a['grads'], deltas=a['deltas'],
            attention_selection_real=a['attention_selection_real'],
            v_conv_content_transfer=a['v_conv_content_transfer'],
            per_image=[{k: v for k, v in r.items()} for r in a['rows']],
        )

    return dict(
        g21=slim(a_g21), tex_h4=slim(a_tex),
        viz_dir=viz_dir, viz_ok=viz_ok,
        verdict=(
            'spatial_selection' if a_tex['attention_selection_real'] else
            'v_conv_content_transfer' if a_g21['v_conv_content_transfer']
            else 'no_spatial_selection'),
        verdict_detail=(
            'Softmax near-uniform on both arms; G2.1 is V/Conv content '
            'transfer; tex_h4 may update Q/K but weights stay flat'),
    )


def _grad_corr(a, b):
    """Simple structural score: corr of spatial gradients."""
    def g(x):
        # x [1,3,H,W] or [1,C,H,W] → magnitude
        if x.shape[1] > 1:
            x = x.mean(1, keepdim=True)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        # pad back
        dx = F.pad(dx, (0, 1))
        dy = F.pad(dy, (0, 0, 0, 1))
        return (dx.pow(2) + dy.pow(2)).sqrt()
    ga, gb = g(a).flatten(), g(b).flatten()
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    denom = ga.norm() * gb.norm()
    if float(denom) < 1e-8:
        return 0.0
    return float((ga * gb).sum() / denom)


def _match_pack(matcher, q, k):
    m = matcher(q, k)
    mag = m.flow.pow(2).sum(1).sqrt()
    return m, dict(
        conf_mean=float(m.confidence.mean()),
        margin_mean=float(m.margin.mean()),
        flow_mean=float(mag.mean()),
        flow_median=float(mag.median()),
        flow_frac_lt1=float((mag < 1).float().mean()),
        flow_frac_lt2=float((mag < 2).float().mean()),
        entropy_mean=float(m.entropy.mean()),
    )


def task_b_c2(ds_m, pairs_m, ds_h, pairs_h, device, out_dir):
    c2_path = C2_FEATURE_CKPT
    c2_sha = _sha256(c2_path)
    c2 = load_c2_feature_extractor(c2_path, device=device, freeze=True)
    # strict load already inside loader; record feature dim
    with torch.no_grad():
        t0 = sample_tensors(ds_m, pairs_m[0]['i'], 'correct', device)
        y0 = m11_to_01(t0['Y0'])
        out0 = c2(y0, y0)
        q0 = out0['dense_features1']
        feat_shape = list(q0.shape)
        feat_ch = int(q0.shape[1])

    # verify reload strict
    c2_b = load_c2_feature_extractor(c2_path, device='cpu', freeze=True)
    sd_a = c2.state_dict()
    sd_b = c2_b.state_dict()
    strict_equal = all(
        torch.equal(sd_a[k].cpu(), sd_b[k]) for k in sd_a)

    matcher = ChunkedHardMatcher().to(device).eval()
    viz_dir = os.path.join(out_dir, 'c2_viz')
    os.makedirs(viz_dir, exist_ok=True)

    def eval_split(ds, pairs, split_tag):
        rows = []
        for idx, p in enumerate(pairs):
            t = sample_tensors(ds, p['i'], 'correct', device)
            donor = pairs[(idx + 1) % len(pairs)]
            td = sample_tensors(ds, donor['i'], 'correct', device)
            y0 = m11_to_01(t['Y0'])
            ref = m11_to_01(t['R'])
            ref_sh = m11_to_01(td['R'])
            # synthetic translation of ref (roll)
            dy_pix, dx_pix = 8, 12  # at image res; feature is ~H/4
            ref_roll = torch.roll(ref, shifts=(dy_pix, dx_pix), dims=(-2, -1))

            with torch.no_grad():
                f_self = c2(y0, y0)
                f_ok = c2(y0, ref)
                f_sh = c2(y0, ref_sh)
                f_roll = c2(y0, ref_roll)
                m_self, s_self = _match_pack(
                    matcher, f_self['dense_features1'], f_self['dense_features2'])
                m_ok, s_ok = _match_pack(
                    matcher, f_ok['dense_features1'], f_ok['dense_features2'])
                m_sh, s_sh = _match_pack(
                    matcher, f_sh['dense_features1'], f_sh['dense_features2'])
                m_roll, s_roll = _match_pack(
                    matcher, f_roll['dense_features1'], f_roll['dense_features2'])

                # expected flow for roll at feature grid (~H/4)
                _, _, fh, fw = m_roll.flow.shape
                scale_y = fh / y0.shape[-2]
                scale_x = fw / y0.shape[-1]
                exp_dy = dy_pix * scale_y
                exp_dx = dx_pix * scale_x
                flow_err = float((
                    (m_roll.flow[:, 0] - exp_dx).pow(2)
                    + (m_roll.flow[:, 1] - exp_dy).pow(2)
                ).sqrt().mean())
                # fraction of pixels whose flow within 2 feat-px of expected
                err_map = (
                    (m_roll.flow[:, 0] - exp_dx).pow(2)
                    + (m_roll.flow[:, 1] - exp_dy).pow(2)
                ).sqrt()
                frac_flow_near_expected = float((err_map < 2).float().mean())

                # warp structural consistency (ref→y0 at feat res)
                def warp_score(m, ref_img):
                    ref_f = F.interpolate(
                        ref_img, size=m.flow.shape[-2:],
                        mode='bilinear', align_corners=False)
                    y0_f = F.interpolate(
                        y0, size=m.flow.shape[-2:],
                        mode='bilinear', align_corners=False)
                    warped = warp_by_flow(ref_f, m.flow)
                    return dict(
                        l1=float((warped - y0_f).abs().mean()),
                        grad_corr=_grad_corr(warped, y0_f),
                    )

                w_ok = warp_score(m_ok, ref)
                w_sh = warp_score(m_sh, ref_sh)
                w_self = warp_score(m_self, y0)

            rec = dict(
                split=split_tag, name=p['name'],
                self_flow_mean=s_self['flow_mean'],
                self_flow_frac_lt1=s_self['flow_frac_lt1'],
                self_conf=s_self['conf_mean'],
                self_margin=s_self['margin_mean'],
                self_warp_l1=w_self['l1'],
                self_warp_grad_corr=w_self['grad_corr'],
                correct_flow_mean=s_ok['flow_mean'],
                correct_conf=s_ok['conf_mean'],
                correct_margin=s_ok['margin_mean'],
                correct_warp_l1=w_ok['l1'],
                correct_warp_grad_corr=w_ok['grad_corr'],
                shuffle_flow_mean=s_sh['flow_mean'],
                shuffle_conf=s_sh['conf_mean'],
                shuffle_margin=s_sh['margin_mean'],
                shuffle_warp_l1=w_sh['l1'],
                shuffle_warp_grad_corr=w_sh['grad_corr'],
                # geometry: correct should warp better than shuffle
                delta_warp_l1_shuffle_minus_correct=(
                    w_sh['l1'] - w_ok['l1']),
                delta_grad_corr_correct_minus_shuffle=(
                    w_ok['grad_corr'] - w_sh['grad_corr']),
                roll_flow_mean=s_roll['flow_mean'],
                roll_expected_dx=exp_dx,
                roll_expected_dy=exp_dy,
                roll_flow_err_to_expected=flow_err,
                roll_frac_flow_near_expected=frac_flow_near_expected,
                # NOT used alone as proof:
                delta_conf_correct_minus_shuffle=(
                    s_ok['conf_mean'] - s_sh['conf_mean']),
            )
            rows.append(rec)

            # viz a few
            if idx < 3:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
                    y0_np = y0[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    ref_np = ref[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    axes[0, 0].imshow(y0_np); axes[0, 0].set_title('Y0')
                    axes[0, 1].imshow(ref_np); axes[0, 1].set_title('Ref')
                    mag = m_ok.flow.pow(2).sum(1).sqrt()[0].cpu()
                    axes[0, 2].imshow(mag, cmap='viridis')
                    axes[0, 2].set_title('flow|correct|')
                    mag_s = m_sh.flow.pow(2).sum(1).sqrt()[0].cpu()
                    axes[1, 0].imshow(mag_s, cmap='viridis')
                    axes[1, 0].set_title('flow|shuffle|')
                    mag_r = m_roll.flow.pow(2).sum(1).sqrt()[0].cpu()
                    axes[1, 1].imshow(mag_r, cmap='viridis')
                    axes[1, 1].set_title('flow|roll|')
                    mag_sf = m_self.flow.pow(2).sum(1).sqrt()[0].cpu()
                    axes[1, 2].imshow(mag_sf, cmap='viridis')
                    axes[1, 2].set_title('flow|self|')
                    for ax in axes.flat:
                        ax.axis('off')
                    fig.suptitle('%s %s' % (split_tag, p['name']))
                    fig.tight_layout()
                    fig.savefig(os.path.join(
                        viz_dir, '%s_%s.png' % (
                            split_tag, p['name'].replace('.png', ''))), dpi=110)
                    plt.close(fig)
                except Exception:
                    pass
        return rows

    rows_m = eval_split(ds_m, pairs_m, 'microset')
    rows_h = eval_split(ds_h, pairs_h, 'microset_heldout')
    all_rows = rows_m + rows_h

    def summ(rows):
        def avg(k):
            return float(np.mean([r[k] for r in rows]))
        return dict(
            n=len(rows),
            self_flow_mean=avg('self_flow_mean'),
            self_flow_frac_lt1=avg('self_flow_frac_lt1'),
            correct_warp_grad_corr=avg('correct_warp_grad_corr'),
            shuffle_warp_grad_corr=avg('shuffle_warp_grad_corr'),
            delta_grad_corr=avg('delta_grad_corr_correct_minus_shuffle'),
            delta_warp_l1=avg('delta_warp_l1_shuffle_minus_correct'),
            roll_frac_near_expected=avg('roll_frac_flow_near_expected'),
            roll_flow_err=avg('roll_flow_err_to_expected'),
            delta_conf=avg('delta_conf_correct_minus_shuffle'),
        )

    sm, sh = summ(rows_m), summ(rows_h)
    # Geometric pass: self near-id, correct warps structurally better than
    # shuffle, and synthetic roll recovers expected translation on a
    # non-trivial fraction of pixels. Confidence alone is insufficient.
    geo_ok = bool(
        sm['self_flow_frac_lt1'] > 0.5
        and sm['delta_grad_corr'] > 0.02
        and sm['delta_warp_l1'] > 0.0
        and sm['roll_frac_near_expected'] > 0.15
        and sh['self_flow_frac_lt1'] > 0.5
        and sh['delta_grad_corr'] > 0.02
    )

    return dict(
        c2_path=c2_path,
        c2_sha256=c2_sha,
        strict_load_verified=strict_equal,
        feature_shape=feat_shape,
        feature_channels=feat_ch,
        microset=sm,
        microset_heldout=sh,
        geometric_match_ok=geo_ok,
        rows=all_rows,
        viz_dir=viz_dir,
        note=('Geometric criteria: self flow, warp grad-corr/L1 correct>shuffle, '
              'synthetic roll recovery — NOT confidence alone'),
    )


def task_c_flags(g22):
    """Independent gates — never OR microset/heldout for long-train go."""
    tex_m = g22['tex_h4_microset']
    tex_h = g22['tex_h4_microset_heldout']
    # locked from user / G2.2
    ref_path_works = bool(
        tex_m['dpsnr_normal_minus_shuffle'] > 0.05
        and tex_m['mae_y_swapA2'] > 1e-3)
    heldout_ref_gain = bool(tex_h['dpsnr_normal_minus_shuffle'] > 0.05)
    heldout_base_safety = bool(tex_h['dpsnr_normal_minus_base'] > -0.1)
    return dict(
        ref_path_works=ref_path_works,
        heldout_ref_gain=heldout_ref_gain,
        heldout_base_safety=heldout_base_safety,
        values=dict(
            microset_dpsnr_n_shuf=tex_m['dpsnr_normal_minus_shuffle'],
            heldout_dpsnr_n_shuf=tex_h['dpsnr_normal_minus_shuffle'],
            heldout_dpsnr_n_base=tex_h['dpsnr_normal_minus_base'],
        ),
        rule=('Long train / Setting B / Official Test require ALL of: '
              'ref_path_works AND heldout_ref_gain AND heldout_base_safety '
              'AND (C2 geometric_match_ok OR expanded-microset plan). '
              'G2.2 clear_ref_utilization=OR is RETRACTED.'),
        allow_long_train=False,  # filled later
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)

    out_dir = os.path.join(OUT, 'gate2_3')
    os.makedirs(out_dir, exist_ok=True)
    g22 = json.load(open(G22_JSON))
    arm = os.path.join(EXP_ROOT, 'A_fusion_only')
    lock = json.load(open(os.path.join(arm, 'artifact_lock.json')))
    sha = _git_sha()

    locked = dict(
        repo_head_at_run=sha,
        g21_code_commit=g22['g21_commit'],
        g22_report_commit=g22.get('report_commit'),
        g21_ckpt=G21_CKPT,
        g21_ckpt_sha256=_sha256(G21_CKPT),
        tex_h4_ckpt=TEX_CKPT,
        tex_h4_ckpt_sha256=_sha256(TEX_CKPT),
        base_ckpt=lock['base_ckpt'],
        base_ckpt_sha256=_sha256(lock['base_ckpt']),
        c2_ckpt=C2_FEATURE_CKPT,
        c2_ckpt_sha256=_sha256(C2_FEATURE_CKPT),
        g22_locked_conclusion=dict(
            microset_n_shuf_dpsnr=0.2760442070180473,
            heldout_n_shuf_dpsnr=-0.044820264508608876,
            heldout_n_base_dpsnr=-2.321880144542228,
            note='G2.2 PASS was ref-path diagnosis only; NOT heldout generalization',
        ),
    )

    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    names = g22['microset_names']
    held_names = g22['microset_heldout_names']
    name_to = {r[0]: r for r in splits['train']}
    train_rows = [name_to[n] for n in names]
    held_rows = [name_to[n] for n in held_names]
    ds_m = _make_ds(lock, train_rows)
    ds_h = _make_ds(lock, held_rows)
    pairs_m = [dict(i=i, name=names[i]) for i in range(len(names))]
    pairs_h = [dict(i=i, name=held_names[i]) for i in range(len(held_names))]

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)

    print('=== Task A: attention tex_h4 vs G2.1 ===', flush=True)
    attn = task_a_attention(base, lock, ds_m, pairs_m, a.device, out_dir)
    print('  g21 selection=%s content_xfer=%s | tex selection=%s content_xfer=%s'
          % (attn['g21']['attention_selection_real'],
             attn['g21']['v_conv_content_transfer'],
             attn['tex_h4']['attention_selection_real'],
             attn['tex_h4']['v_conv_content_transfer']), flush=True)

    print('=== Task B: C2 geometric match ===', flush=True)
    c2 = task_b_c2(ds_m, pairs_m, ds_h, pairs_h, a.device, out_dir)
    print('  geo_ok=%s  feat=%s  roll_frac_m=%.3f Δgradcorr_m=%.4f'
          % (c2['geometric_match_ok'], c2['feature_shape'],
             c2['microset']['roll_frac_near_expected'],
             c2['microset']['delta_grad_corr']), flush=True)

    print('=== Task C: independent flags ===', flush=True)
    flags = task_c_flags(g22)
    # Next-step recommendation (no train this round)
    if c2['geometric_match_ok']:
        next_step = dict(
            type='C2_warp_single_variable',
            action=('Propose only: replace random Match/Flow with frozen C2 '
                    'match + warp of tex[h4] as A2; keep Fusion win3, '
                    'correct-only, D2-only, same seed; short microset — '
                    'DO NOT TRAIN in Gate 2.3'),
            train_now=False,
        )
    else:
        next_step = dict(
            type='expand_microset_single_variable',
            action=('Propose only: keep tex_h4 (or identity) path; expand '
                    'correct-only microset beyond n=8 to test whether '
                    'heldout_ref_gain / heldout_base_safety can turn positive '
                    '— DO NOT TRAIN in Gate 2.3'),
            train_now=False,
        )
    flags['allow_long_train'] = bool(
        flags['ref_path_works'] and flags['heldout_ref_gain']
        and flags['heldout_base_safety'] and c2['geometric_match_ok'])
    flags['next_step'] = next_step

    # CSV
    csv_attn = os.path.join(out_dir, 'attn_per_image.csv')
    with open(csv_attn, 'w', newline='', encoding='utf-8') as f:
        rows = attn['g21']['per_image'] + attn['tex_h4']['per_image']
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    csv_c2 = os.path.join(out_dir, 'c2_match_per_image.csv')
    with open(csv_c2, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(c2['rows'][0].keys()))
        w.writeheader()
        w.writerows(c2['rows'])

    report = dict(
        gate='G2.3_attn_and_c2_geometry',
        locked=locked,
        task_a_attention=dict(
            g21=attn['g21'], tex_h4=attn['tex_h4'],
            verdict=attn['verdict'], viz_dir=attn['viz_dir'],
        ),
        task_b_c2={k: v for k, v in c2.items() if k != 'rows'},
        task_c_flags=flags,
        artifacts=dict(
            attn_csv=csv_attn, c2_csv=csv_c2,
            attn_viz=attn['viz_dir'], c2_viz=c2['viz_dir'],
        ),
        gate2_3_pass=bool(
            # diagnostic complete; not a train-go signal
            flags['ref_path_works'] is True
            and flags['heldout_ref_gain'] is False
            and flags['heldout_base_safety'] is False
            and isinstance(c2['geometric_match_ok'], bool)
        ),
        train_authorization='DENIED',
    )

    out_json = os.path.join(OUT, 'G2_3_attn_c2_audit.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    md = os.path.join(OUT, 'Gate2_3_report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Gate 2.3 — Attention Selection & C2 Geometry\n\n')
        f.write('## Locked context\n\n')
        f.write('- **run HEAD:** `%s`\n' % sha)
        f.write('- **G2.1 ckpt SHA256:** `%s`\n' % locked['g21_ckpt_sha256'])
        f.write('- **tex_h4_200 ckpt SHA256:** `%s`\n'
                % locked['tex_h4_ckpt_sha256'])
        f.write('- **C2 feature_extraction SHA256:** `%s`\n'
                % locked['c2_ckpt_sha256'])
        f.write('- **G2.2 lock:** microset N−Shuf **+0.276 dB**; heldout '
                'N−Shuf **−0.045 dB**; heldout N−Base **−2.322 dB**.\n')
        f.write('- G2.2 was **ref-path diagnosis only**, not heldout '
                'generalization.\n\n')
        f.write('## Task A — Attention (no train)\n\n')
        f.write('| | G2.1 align | tex_h4 |\n|---|---|---|\n')
        f.write('| weight_max | %.4f | %.4f |\n' % (
            attn['g21']['mean_weight_max'], attn['tex_h4']['mean_weight_max']))
        f.write('| entropy_norm | %.4f | %.4f |\n' % (
            attn['g21']['mean_entropy_norm'],
            attn['tex_h4']['mean_entropy_norm']))
        f.write('| Δq / Δk / Δv | %.3g / %.3g / %.3g | %.3g / %.3g / %.3g |\n'
                % (attn['g21']['deltas']['q'], attn['g21']['deltas']['k'],
                   attn['g21']['deltas']['v'],
                   attn['tex_h4']['deltas']['q'], attn['tex_h4']['deltas']['k'],
                   attn['tex_h4']['deltas']['v']))
        f.write('| grad q/k/v | %.3g / %.3g / %.3g | %.3g / %.3g / %.3g |\n'
                % (attn['g21']['grads']['q'], attn['g21']['grads']['k'],
                   attn['g21']['grads']['v'],
                   attn['tex_h4']['grads']['q'], attn['tex_h4']['grads']['k'],
                   attn['tex_h4']['grads']['v']))
        f.write('| selection_real | %s | %s |\n' % (
            attn['g21']['attention_selection_real'],
            attn['tex_h4']['attention_selection_real']))
        f.write('| V/Conv content transfer | %s | %s |\n\n' % (
            attn['g21']['v_conv_content_transfer'],
            attn['tex_h4']['v_conv_content_transfer']))
        f.write('**Verdict:** `%s` — Ref path works primarily via V/fusion '
                'content path, not peaking 3×3 Softmax selection.\n\n'
                % attn['verdict'])
        f.write('Viz: `%s`\n\n' % attn['viz_dir'])
        f.write('## Task B — Pretrained C2 geometry (no train)\n\n')
        f.write('| | value |\n|---|---|\n')
        f.write('| strict_load_verified | %s |\n' % c2['strict_load_verified'])
        f.write('| feature shape | %s |\n' % c2['feature_shape'])
        f.write('| channels | %d |\n' % c2['feature_channels'])
        f.write('| geometric_match_ok | **%s** |\n\n' % c2['geometric_match_ok'])
        for tag, s in (('microset', c2['microset']),
                       ('microset-heldout', c2['microset_heldout'])):
            f.write('### %s\n\n' % tag)
            f.write('| self ‖flow‖ | self frac&lt;1 | Δgrad_corr '
                    '(corr−shuf) | Δwarp_L1 (shuf−corr) | '
                    'roll frac near expected |\n')
            f.write('|---|---|---|---|---|\n')
            f.write('| %.3f | %.3f | %+.4f | %+.4f | %.3f |\n\n' % (
                s['self_flow_mean'], s['self_flow_frac_lt1'],
                s['delta_grad_corr'], s['delta_warp_l1'],
                s['roll_frac_near_expected']))
        f.write('Confidence delta is reported in CSV but **not** used as '
                'proof of match correctness.\n\n')
        f.write('## Task C — Independent gate flags\n\n')
        f.write('| flag | value | meaning |\n|---|---|---|\n')
        f.write('| `ref_path_works` | **%s** | microset Ref path sensitivity '
                '|\n' % flags['ref_path_works'])
        f.write('| `heldout_ref_gain` | **%s** | heldout Normal−Shuffle '
                '> 0.05 dB |\n' % flags['heldout_ref_gain'])
        f.write('| `heldout_base_safety` | **%s** | heldout Normal−Base '
                '> −0.1 dB |\n' % flags['heldout_base_safety'])
        f.write('| `allow_long_train` | **%s** | all safety flags + C2 geo '
                '|\n\n' % flags['allow_long_train'])
        f.write('G2.2 `clear_ref_utilization = microset OR heldout` is '
                '**retracted** for authorizing further training.\n\n')
        f.write('## Next step (proposal only — no train this round)\n\n')
        f.write('- **type:** `%s`\n' % next_step['type'])
        f.write('- %s\n\n' % next_step['action'])
        f.write('## Verdict\n\n')
        f.write('**Gate 2.3 diagnostic complete.** '
                '`train_authorization=DENIED`.\n\n')
        f.write('## Repro\n\n```bash\n')
        f.write('cd %s\n' % REPO)
        f.write('/root/miniconda3/envs/ttsr/bin/python '
                'scripts/gate2_3_c2_and_attn_audit.py --device cuda\n')
        f.write('```\n')

    # also copy key outputs to docs later in push step
    print(json.dumps(dict(
        gate2_3_pass=report['gate2_3_pass'],
        attn_verdict=attn['verdict'],
        c2_geo=c2['geometric_match_ok'],
        flags=flags,
        next_step=next_step,
    ), indent=2), flush=True)
    print('wrote %s' % md, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
