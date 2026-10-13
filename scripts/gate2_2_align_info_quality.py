#!/usr/bin/env python
"""Gate 2.2 — Alignment information quality.

1) Fix overfit metric: same 8-image mean MSE at init vs G2.1 ckpt (not
   losses[-1] vs losses[0] across different images). Per-image Base/Normal/
   Shuffle PSNR. Label train-split extras as microset-heldout (not formal dev).
2) No-train attention audit on G2.1 ckpt (entropy, max weight, Q/K/V grad + Δθ).
3) Single-variable: A2 = tex['h4'] (bypass Match/Flow/DCN); same Fusion/data/
   init/opt/200/seed. Microset + heldout sensitivity.
4) If still no clear Ref use → audit pretrained C2 match only (no Setting B).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

REPO = '/root/projects/TTSR-lowlight'
OUT = '/root/data/experiments/V7_CODE_REVIEW_FIX'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics  # noqa: E402
from model.V6LLFormerRefBridge import (  # noqa: E402
    load_frozen_llformer_base, tiled_v6_forward, base_forward_m11,
    m11_to_01, o1_to_m11, reflect_pad_to_multiple, crop_to_hw, PAD_MULTIPLE)
from model.V7Model import V7Model, ARCHITECTURE_WIN3_D2  # noqa: E402
from model.V7RiverFusion import V7RiverFusion  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3b_runtime import b0_loss  # noqa: E402
from v7_river_runtime import ROOT as EXP_ROOT, SRC  # noqa: E402

N_IMAGES = 8
UPDATES = 200
LR = 1e-4
SEED = 42
G21_CKPT = os.path.join(OUT, 'gate2_1', 'ckpt_microset_win3_fixed.pt')
G21_JSON = os.path.join(OUT, 'G2_1_window_layout_fix.json')
HELDOUT_JSON = '/root/data/experiments/V7_REF_DIAG/heldout24.json'
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'
C2_CKPT = (
    '/root/projects/TTSR-lowlight/pretrain_model/C2matching/feature_extraction.pth')


def _git_sha():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def _grad_norm(module, prefix):
    total = 0.0
    for n, p in module.named_parameters():
        if n.startswith(prefix) and p.grad is not None:
            total += float(p.grad.detach().norm())
    return total


def _param_norm(module, prefix):
    total = 0.0
    for n, p in module.named_parameters():
        if n.startswith(prefix):
            total += float(p.detach().norm())
    return total


def _param_delta(a, b, prefix):
    """L2 of (a-b) for params under prefix."""
    total = 0.0
    da = dict(a.named_parameters())
    db = dict(b.named_parameters())
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


def _forward_tiled(base, model, x, y0, r, roll_a2=False):
    if not roll_a2:
        return tiled_v6_forward(base, model, x, y0, r)
    orig = model.refine_h4.forward

    def wrapped(d, a, return_attn=False):
        a = torch.roll(a, shifts=(3, 5), dims=(-2, -1))
        return orig(d, a, return_attn=return_attn)

    model.refine_h4.forward = wrapped
    try:
        return tiled_v6_forward(base, model, x, y0, r)
    finally:
        model.refine_h4.forward = orig


def _load_v7(lock, device, ckpt_path=None, a2_mode='align'):
    model = V7Model(spatial_window=3, d1_enabled=False).to(device)
    init = torch.load(lock['v7_init_path'], map_location='cpu')
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


def _eval_batch_mse(base, model, ds, pairs, device):
    """Mean MSE over the fixed pair list (same images every call)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for p in pairs:
            t = sample_tensors(ds, p['i'], 'correct', device)
            y = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
            losses.append(float(b0_loss(y, t['H'])))
    return float(sum(losses) / len(losses)), losses


def _eval_per_image_psnr(base, model, ds, pairs, device, split_tag):
    """Per-image Base / Normal / Shuffle PSNR (+ sensitivities)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for idx, p in enumerate(pairs):
            t = sample_tensors(ds, p['i'], 'correct', device)
            donor = pairs[(idx + 1) % len(pairs)]
            td = sample_tensors(ds, donor['i'], 'correct', device)
            y_base = base_forward_m11(base, t['X'])
            y_n = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
            y_shuf = _forward_tiled(base, model, t['X'], t['Y0'], td['R'])
            y_swap = _forward_tiled(
                base, model, t['X'], t['Y0'], t['R'], roll_a2=True)
            pb, _, _ = _metrics(y_base, t['H'])
            pn, _, _ = _metrics(y_n, t['H'])
            ps, _, _ = _metrics(y_shuf, t['H'])
            rows.append(dict(
                split=split_tag,
                name=p['name'],
                psnr_base=float(pb),
                psnr_normal=float(pn),
                psnr_shuffle=float(ps),
                dpsnr_normal_minus_base=float(pn - pb),
                dpsnr_normal_minus_shuffle=float(pn - ps),
                mae_y_swapA2=float((y_n - y_swap).abs().mean()),
                mae_y_shuffleRef=float((y_n - y_shuf).abs().mean()),
                mse_normal=float(b0_loss(y_n, t['H'])),
            ))
    return rows


def _attn_stats_from_weights(weights):
    # weights [B,Nh,HW,N]
    ent = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
    # uniform entropy for N keys
    n = weights.shape[-1]
    ent_uniform = float(torch.log(torch.tensor(float(n))))
    wmax = weights.max(-1).values
    # fraction where argmax is center (index N//2)
    center = n // 2
    argmax = weights.argmax(-1)
    frac_center = float((argmax == center).float().mean())
    return dict(
        n_keys=n,
        entropy_mean=float(ent.mean()),
        entropy_uniform=ent_uniform,
        entropy_frac_of_uniform=float(ent.mean() / ent_uniform),
        weight_max_mean=float(wmax.mean()),
        weight_max_p90=float(torch.quantile(wmax.flatten(), 0.9)),
        frac_argmax_center=frac_center,
        weight_center_mean=float(weights[..., center].mean()),
    )


@torch.no_grad()
def _collect_attn_on_tile(base, model, x, y0, r):
    """Run one (possibly tiled) image; gather attn from first tile via hook."""
    captured = {}

    def hook(mod, args, kwargs=None):
        pass

    # Direct single-tile path if image fits; else first crop
    _, _, H, W = x.shape
    from model.V3BFeatureBridge import TILE_SIZE
    th = min(TILE_SIZE, H)
    tw = min(TILE_SIZE, W)
    xt, yt, rt = x[:, :, :th, :tw], y0[:, :, :th, :tw], r[:, :, :th, :tw]

    x01 = m11_to_01(xt)
    xp, h0, w0 = reflect_pad_to_multiple(x01, PAD_MULTIPLE)
    yp, _, _ = reflect_pad_to_multiple(m11_to_01(yt), PAD_MULTIPLE)
    rp, _, _ = reflect_pad_to_multiple(m11_to_01(rt), PAD_MULTIPLE)
    y_pad, r_pad = o1_to_m11(yp), o1_to_m11(rp)
    st = base.encode_to_d2(xp)
    d2 = st['d2']
    match, tex = model.match_and_texture(y_pad, r_pad)
    if model.a2_mode == 'tex_h4':
        a2 = tex['h4']
    else:
        a2, _ = model.align_h4(tex['h4'], d2, match.flow, match.confidence)
    _, _, info = model.refine_h4(d2, a2, return_attn=True)
    return _attn_stats_from_weights(info['weights']), info


def audit_attention(base, model_init, model_g21, ds, pairs, device):
    """No-train: attn selection + Q/K/V grad + param update vs init."""
    model_g21.eval()
    attn_rows = []
    for p in pairs:
        t = sample_tensors(ds, p['i'], 'correct', device)
        stats, _ = _collect_attn_on_tile(
            base, model_g21, t['X'], t['Y0'], t['R'])
        stats['name'] = p['name']
        attn_rows.append(stats)

    # grads on G2.1 with one microset image
    model_g21.refine_h4.train()
    model_g21.zero_grad(set_to_none=True)
    t = sample_tensors(ds, pairs[0]['i'], 'correct', device)
    y = _forward_tiled(base, model_g21, t['X'], t['Y0'], t['R'])
    b0_loss(y, t['H']).backward()
    grads = dict(
        q=_grad_norm(model_g21.refine_h4, 'q_proj'),
        k=_grad_norm(model_g21.refine_h4, 'k_proj'),
        v=_grad_norm(model_g21.refine_h4, 'v_proj'),
        out_conv=_grad_norm(model_g21.refine_h4, 'out_conv'),
    )
    model_g21.zero_grad(set_to_none=True)
    model_g21.eval()

    deltas = dict(
        q=_param_delta(model_g21.refine_h4, model_init.refine_h4, 'q_proj'),
        k=_param_delta(model_g21.refine_h4, model_init.refine_h4, 'k_proj'),
        v=_param_delta(model_g21.refine_h4, model_init.refine_h4, 'v_proj'),
        out_conv=_param_delta(
            model_g21.refine_h4, model_init.refine_h4, 'out_conv'),
        fusion=_param_delta(
            model_g21.refine_h4, model_init.refine_h4, 'fusion'),
    )
    norms = dict(
        q_init=_param_norm(model_init.refine_h4, 'q_proj'),
        q_g21=_param_norm(model_g21.refine_h4, 'q_proj'),
        k_init=_param_norm(model_init.refine_h4, 'k_proj'),
        k_g21=_param_norm(model_g21.refine_h4, 'k_proj'),
        v_init=_param_norm(model_init.refine_h4, 'v_proj'),
        v_g21=_param_norm(model_g21.refine_h4, 'v_proj'),
    )

    ent = [r['entropy_frac_of_uniform'] for r in attn_rows]
    wmax = [r['weight_max_mean'] for r in attn_rows]
    # Selection "real" if mean max-weight clearly above 1/N and entropy
    # clearly below uniform (not near-uniform), AND Q or K updated.
    n_keys = attn_rows[0]['n_keys']
    uniform_w = 1.0 / n_keys
    mean_wmax = float(sum(wmax) / len(wmax))
    mean_ent_frac = float(sum(ent) / len(ent))
    qk_updated = deltas['q'] > 1e-4 or deltas['k'] > 1e-4
    selection_real = bool(
        mean_wmax > uniform_w * 1.5 and mean_ent_frac < 0.95 and qk_updated)

    return dict(
        per_image=attn_rows,
        grad_g21=grads,
        param_delta_init_to_g21=deltas,
        param_norms=norms,
        mean_weight_max=mean_wmax,
        mean_entropy_frac_of_uniform=mean_ent_frac,
        uniform_weight=uniform_w,
        qk_param_updated=qk_updated,
        attention_selection_real=selection_real,
    )


def train_tex_h4(base, lock, ds, pairs, device):
    torch.manual_seed(SEED)
    model = _load_v7(lock, device, ckpt_path=None, a2_mode='tex_h4')
    # step0 identity
    t0 = sample_tensors(ds, pairs[0]['i'], 'correct', device)
    with torch.no_grad():
        y0p = _forward_tiled(base, model, t0['X'], t0['Y0'], t0['R'])
    step0 = float((y0p - t0['Y0']).abs().max())
    if step0 > 1e-3:
        raise SystemExit('tex_h4 step0 not identity: %s' % step0)

    opt = torch.optim.Adam(
        [p for p in model.refine_h4.parameters() if p.requires_grad], lr=LR)
    # batch-mean MSE at checkpoints
    mse0, _ = _eval_batch_mse(base, model, ds, pairs, device)
    log_path = os.path.join(OUT, 'gate2_2', 'train_tex_h4.log')
    t_wall = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        def log(msg):
            print(msg, flush=True)
            logf.write(msg + '\n')
            logf.flush()

        log('G2.2 tex_h4 bypass-align n=%d updates=%d seed=%d' % (
            len(pairs), UPDATES, SEED))
        log('  batch_mse@0 = %.6f' % mse0)
        for step in range(1, UPDATES + 1):
            model.refine_h4.train()
            opt.zero_grad(set_to_none=True)
            p = pairs[(step - 1) % len(pairs)]
            t = sample_tensors(ds, p['i'], 'correct', device)
            y = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
            loss = b0_loss(y, t['H'])
            loss.backward()
            for p_b in base.parameters():
                if p_b.grad is not None:
                    raise SystemExit('Base got grad')
            opt.step()
            if step == 1 or step % 20 == 0 or step == UPDATES:
                log('  step %d  instant_loss=%.5f  (%.0fs)'
                    % (step, float(loss.detach()), time.time() - t_wall))
    mse200, per = _eval_batch_mse(base, model, ds, pairs, device)
    ckpt = os.path.join(OUT, 'gate2_2', 'ckpt_tex_h4_200.pt')
    torch.save(dict(
        model=model.state_dict(), step=UPDATES, a2_mode='tex_h4',
        spatial_window=3, d1_enabled=False, seed=SEED, lr=LR,
        architecture=ARCHITECTURE_WIN3_D2,
        batch_mse_0=mse0, batch_mse_200=mse200,
    ), ckpt)
    return model, dict(
        batch_mse_0=mse0, batch_mse_200=mse200,
        overfit_batch_mse=(mse200 < mse0 * 0.95),
        per_image_mse_200=per, ckpt=ckpt, step0_max_abs=step0,
    )


def audit_c2(ds, pairs, device, split_tag):
    """Frozen pretrained C2: self/correct/shuffle flow+conf (no train)."""
    if not os.path.isfile(C2_CKPT):
        return dict(available=False, path=C2_CKPT)
    from model.c2_extractor import load_c2_feature_extractor
    from model.V5Correspondence import ChunkedHardMatcher

    c2 = load_c2_feature_extractor(C2_CKPT, device=device, freeze=True)
    matcher = ChunkedHardMatcher().to(device).eval()
    rows = []
    with torch.no_grad():
        for idx, p in enumerate(pairs):
            t = sample_tensors(ds, p['i'], 'correct', device)
            donor = pairs[(idx + 1) % len(pairs)]
            td = sample_tensors(ds, donor['i'], 'correct', device)
            y0 = m11_to_01(t['Y0'])
            ref = m11_to_01(t['R'])
            ref_s = m11_to_01(td['R'])
            # C2 expects ImageNet-normalized inside extractor
            f_self = c2(y0, y0)
            f_ok = c2(y0, ref)
            f_sh = c2(y0, ref_s)
            # dense_features are at reduced res; match q=image1, k=image2
            def _match(feat):
                q = feat['dense_features1']
                k = feat['dense_features2']
                # channel may differ from V5; matcher only needs same C
                if q.shape[1] != k.shape[1]:
                    raise SystemExit('C2 q/k ch mismatch')
                # ChunkedHardMatcher expects same spatial — already
                m = matcher(q, k)
                mag = m.flow.pow(2).sum(1).sqrt()
                return dict(
                    conf=float(m.confidence.mean()),
                    flow_mean=float(mag.mean()),
                    flow_frac_lt1=float((mag < 1).float().mean()),
                    margin=float(m.margin.mean()),
                )

            s_self = _match(f_self)
            s_ok = _match(f_ok)
            s_sh = _match(f_sh)
            rows.append(dict(
                split=split_tag, name=p['name'],
                self_conf=s_self['conf'], self_flow_mean=s_self['flow_mean'],
                self_flow_frac_lt1=s_self['flow_frac_lt1'],
                correct_conf=s_ok['conf'], correct_flow_mean=s_ok['flow_mean'],
                shuffle_conf=s_sh['conf'], shuffle_flow_mean=s_sh['flow_mean'],
                delta_conf_correct_minus_shuffle=s_ok['conf'] - s_sh['conf'],
            ))
    mean_dc = float(sum(
        r['delta_conf_correct_minus_shuffle'] for r in rows) / len(rows))
    mean_self_f = float(sum(r['self_flow_mean'] for r in rows) / len(rows))
    return dict(
        available=True, path=C2_CKPT, n=len(rows),
        mean_delta_conf_correct_minus_shuffle=mean_dc,
        mean_self_flow=mean_self_f,
        match_quality_ok=bool(mean_dc > 0.02 and mean_self_f < 2.0),
        rows=rows,
    )


def _summarize_rows(rows):
    def avg(k):
        return float(sum(r[k] for r in rows) / len(rows))
    return dict(
        n=len(rows),
        psnr_base=avg('psnr_base'),
        psnr_normal=avg('psnr_normal'),
        psnr_shuffle=avg('psnr_shuffle'),
        dpsnr_normal_minus_base=avg('dpsnr_normal_minus_base'),
        dpsnr_normal_minus_shuffle=avg('dpsnr_normal_minus_shuffle'),
        mae_y_swapA2=avg('mae_y_swapA2'),
        mae_y_shuffleRef=avg('mae_y_shuffleRef'),
        mse_normal=avg('mse_normal'),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)

    os.makedirs(os.path.join(OUT, 'gate2_2'), exist_ok=True)
    g21 = json.load(open(G21_JSON))
    arm = os.path.join(EXP_ROOT, 'A_fusion_only')
    lock = json.load(open(os.path.join(arm, 'artifact_lock.json')))
    sha = _git_sha()

    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    train_rows = splits['train'][:N_IMAGES]
    names = [r[0] for r in train_rows]
    if names != g21['image_names']:
        raise SystemExit('microset image drift vs G2.1')

    held = json.load(open(HELDOUT_JSON))
    name_to_row = {r[0]: r for r in splits['train']}
    micro_set = set(names)
    held_names = [n for n in held['names']
                  if n in name_to_row and n not in micro_set][:16]
    held_rows = [name_to_row[n] for n in held_names]

    ds_m = _make_ds(lock, train_rows)
    ds_h = _make_ds(lock, held_rows)
    pairs_m = [dict(i=i, name=train_rows[i][0]) for i in range(len(train_rows))]
    pairs_h = [dict(i=i, name=held_rows[i][0]) for i in range(len(held_rows))]

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)
    model_init = _load_v7(lock, a.device, ckpt_path=None, a2_mode='align')
    model_g21 = _load_v7(lock, a.device, ckpt_path=G21_CKPT, a2_mode='align')

    report = dict(
        gate='G2.2_alignment_info_quality',
        g21_commit=g21['actual_commit'],
        actual_commit=sha,
        g21_ckpt=G21_CKPT,
        base_ckpt=lock['base_ckpt'],
        seed=SEED, lr=LR, updates=UPDATES, n_microset=N_IMAGES,
        optimizer='Adam', loss='MSE',
        microset_names=names,
        microset_heldout_names=held_names,
        note_heldout=('microset-heldout from train split (NOT formal split.dev); '
                      'same protocol as G2.1 16-image set'),
    )

    # ---- 1) Fixed-batch overfit + per-image PSNR for G2.1 ----
    print('=== G2.1 fixed-batch MSE ===', flush=True)
    mse0, mse0_list = _eval_batch_mse(base, model_init, ds_m, pairs_m, a.device)
    mse200, mse200_list = _eval_batch_mse(
        base, model_g21, ds_m, pairs_m, a.device)
    report['g21_batch_eval'] = dict(
        batch_mse_init=mse0,
        batch_mse_g21=mse200,
        overfit_ok=bool(mse200 < mse0 * 0.95),
        bogus_step_loss_comparison={
            'old_loss_first': g21.get('loss_first'),
            'old_loss_last': g21.get('loss_last'),
            'old_overfit_ok_claim': g21.get('overfit_ok'),
            'note': 'old compared per-step losses on different images; superseded',
        },
        per_image_mse_init=mse0_list,
        per_image_mse_g21=mse200_list,
    )
    print('  batch_mse init=%.6f  g21=%.6f  overfit=%s'
          % (mse0, mse200, report['g21_batch_eval']['overfit_ok']), flush=True)

    rows_g21_m = _eval_per_image_psnr(
        base, model_g21, ds_m, pairs_m, a.device, 'microset')
    rows_g21_h = _eval_per_image_psnr(
        base, model_g21, ds_h, pairs_h, a.device, 'microset_heldout')
    report['g21_microset'] = _summarize_rows(rows_g21_m)
    report['g21_microset_heldout'] = _summarize_rows(rows_g21_h)

    # ---- 2) Attention audit (no train) ----
    print('=== Attention audit (G2.1, no train) ===', flush=True)
    attn = audit_attention(
        base, model_init, model_g21, ds_m, pairs_m, a.device)
    report['attention_audit'] = {k: v for k, v in attn.items() if k != 'per_image'}
    report['attention_audit']['per_image_summary_n'] = len(attn['per_image'])
    # keep full per-image in separate json later
    attn_path = os.path.join(OUT, 'gate2_2', 'attention_per_image.json')
    with open(attn_path, 'w', encoding='utf-8') as f:
        json.dump(attn['per_image'], f, indent=2)
    report['attention_per_image_json'] = attn_path
    print('  selection_real=%s  wmax=%.4f  ent_frac=%.4f  Δq=%.4g Δk=%.4g Δv=%.4g'
          % (attn['attention_selection_real'], attn['mean_weight_max'],
             attn['mean_entropy_frac_of_uniform'],
             attn['param_delta_init_to_g21']['q'],
             attn['param_delta_init_to_g21']['k'],
             attn['param_delta_init_to_g21']['v']), flush=True)

    # ---- 3) tex_h4 single-variable train ----
    print('=== tex_h4 bypass-align train ===', flush=True)
    model_tex, tex_train = train_tex_h4(base, lock, ds_m, pairs_m, a.device)
    report['tex_h4_train'] = tex_train
    rows_tex_m = _eval_per_image_psnr(
        base, model_tex, ds_m, pairs_m, a.device, 'microset')
    rows_tex_h = _eval_per_image_psnr(
        base, model_tex, ds_h, pairs_h, a.device, 'microset_heldout')
    report['tex_h4_microset'] = _summarize_rows(rows_tex_m)
    report['tex_h4_microset_heldout'] = _summarize_rows(rows_tex_h)

    # Clear Ref utilization?
    def _ref_used(summary):
        return bool(
            summary['mae_y_swapA2'] > 1e-3
            and summary['dpsnr_normal_minus_shuffle'] > 0.05
            and summary['dpsnr_normal_minus_base'] > 0.05)

    report['tex_h4_ref_util_microset'] = _ref_used(report['tex_h4_microset'])
    report['tex_h4_ref_util_heldout'] = _ref_used(report['tex_h4_microset_heldout'])
    report['clear_ref_utilization'] = bool(
        report['tex_h4_ref_util_microset'] or report['tex_h4_ref_util_heldout'])

    # ---- 4) C2 audit if needed ----
    if not report['clear_ref_utilization']:
        print('=== C2 pretrained match audit (no train) ===', flush=True)
        c2_m = audit_c2(ds_m, pairs_m, a.device, 'microset')
        c2_h = audit_c2(ds_h, pairs_h, a.device, 'microset_heldout')
        report['c2_audit_microset'] = {
            k: v for k, v in c2_m.items() if k != 'rows'}
        report['c2_audit_microset_heldout'] = {
            k: v for k, v in c2_h.items() if k != 'rows'}
        c2_rows_path = os.path.join(OUT, 'gate2_2', 'c2_match_per_image.csv')
        with open(c2_rows_path, 'w', newline='', encoding='utf-8') as f:
            if c2_m.get('rows'):
                w = csv.DictWriter(f, fieldnames=list(c2_m['rows'][0].keys()))
                w.writeheader()
                w.writerows(c2_m['rows'])
                w.writerows(c2_h.get('rows') or [])
        report['c2_match_csv'] = c2_rows_path
    else:
        report['c2_audit_microset'] = dict(skipped=True, reason='ref_util_ok')

    # ---- CSV all PSNR rows ----
    csv_path = os.path.join(OUT, 'gate2_2', 'per_image_psnr.csv')
    all_rows = []
    for r in rows_g21_m:
        r2 = dict(r); r2['arm'] = 'g21_align'; all_rows.append(r2)
    for r in rows_g21_h:
        r2 = dict(r); r2['arm'] = 'g21_align'; all_rows.append(r2)
    for r in rows_tex_m:
        r2 = dict(r); r2['arm'] = 'tex_h4'; all_rows.append(r2)
    for r in rows_tex_h:
        r2 = dict(r); r2['arm'] = 'tex_h4'; all_rows.append(r2)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    report['per_image_csv'] = csv_path

    report['gate2_2_pass'] = bool(
        report['g21_batch_eval']['overfit_ok']
        and report['clear_ref_utilization'])
    report['stop_reason'] = (
        None if report['gate2_2_pass'] else
        'no_clear_ref_utilization_after_tex_h4; C2 audited; no Setting B / long train')

    out_json = os.path.join(OUT, 'G2_2_align_info_quality.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    md = os.path.join(OUT, 'Gate2_2_report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Gate 2.2 — Alignment Information Quality\n\n')
        f.write('- **actual commit:** `%s`\n' % sha)
        f.write('- **G2.1 code commit:** `%s`\n' % g21['actual_commit'])
        f.write('- **G2.1 ckpt:** `%s`\n' % G21_CKPT)
        f.write('- **seed/lr/updates/n:** %d / %s / %d / %d\n\n'
                % (SEED, LR, UPDATES, N_IMAGES))
        f.write('## 1. Fixed-batch overfit (supersedes step-loss claim)\n\n')
        f.write('| | init | G2.1 |\n|---|---|---|\n')
        f.write('| mean MSE (8 imgs) | %.6f | %.6f |\n'
                % (mse0, mse200))
        f.write('| overfit_ok | | %s |\n\n'
                % report['g21_batch_eval']['overfit_ok'])
        f.write('Old G2.1 compared `loss_first`/`loss_last` on **different** '
                'images in the cycle — invalid for overfit.\n\n')
        f.write('## 2. Attention audit (no train)\n\n')
        aa = report['attention_audit']
        f.write('| metric | value |\n|---|---|\n')
        f.write('| mean weight_max | %.4f (uniform=%.4f) |\n'
                % (aa['mean_weight_max'], aa['uniform_weight']))
        f.write('| mean entropy / uniform | %.4f |\n'
                % aa['mean_entropy_frac_of_uniform'])
        f.write('| Δ‖q‖ / Δ‖k‖ / Δ‖v‖ | %.4g / %.4g / %.4g |\n'
                % (aa['param_delta_init_to_g21']['q'],
                   aa['param_delta_init_to_g21']['k'],
                   aa['param_delta_init_to_g21']['v']))
        f.write('| grad q/k/v @G2.1 | %.4g / %.4g / %.4g |\n'
                % (aa['grad_g21']['q'], aa['grad_g21']['k'],
                   aa['grad_g21']['v']))
        f.write('| **attention_selection_real** | **%s** |\n\n'
                % aa['attention_selection_real'])
        f.write('## 3. G2.1 align arm — PSNR / sensitivity\n\n')
        for tag, key in (('microset', 'g21_microset'),
                         ('microset-heldout', 'g21_microset_heldout')):
            s = report[key]
            f.write('### %s (n=%d)\n\n' % (tag, s['n']))
            f.write('| PSNR Base | Normal | Shuffle | N−Base | N−Shuf | '
                    'mae_swap | mae_shuf |\n')
            f.write('|---|---|---|---|---|---|---|\n')
            f.write('| %.3f | %.3f | %.3f | %+.4f | %+.4f | %.3e | %.3e |\n\n'
                    % (s['psnr_base'], s['psnr_normal'], s['psnr_shuffle'],
                       s['dpsnr_normal_minus_base'],
                       s['dpsnr_normal_minus_shuffle'],
                       s['mae_y_swapA2'], s['mae_y_shuffleRef']))
        f.write('## 4. Single-variable: A2 = tex[\'h4\'] (bypass align)\n\n')
        f.write('| batch MSE @0 | @200 | overfit |\n|---|---|---|\n')
        f.write('| %.6f | %.6f | %s |\n\n' % (
            tex_train['batch_mse_0'], tex_train['batch_mse_200'],
            tex_train['overfit_batch_mse']))
        for tag, key in (('microset', 'tex_h4_microset'),
                         ('microset-heldout', 'tex_h4_microset_heldout')):
            s = report[key]
            f.write('### %s (n=%d)\n\n' % (tag, s['n']))
            f.write('| PSNR Base | Normal | Shuffle | N−Base | N−Shuf | '
                    'mae_swap | mae_shuf |\n')
            f.write('|---|---|---|---|---|---|---|\n')
            f.write('| %.3f | %.3f | %.3f | %+.4f | %+.4f | %.3e | %.3e |\n\n'
                    % (s['psnr_base'], s['psnr_normal'], s['psnr_shuffle'],
                       s['dpsnr_normal_minus_base'],
                       s['dpsnr_normal_minus_shuffle'],
                       s['mae_y_swapA2'], s['mae_y_shuffleRef']))
        f.write('clear_ref_utilization: **%s**\n\n'
                % report['clear_ref_utilization'])
        if 'c2_audit_microset' in report and not report['c2_audit_microset'].get(
                'skipped'):
            f.write('## 5. Pretrained C2 match audit (no train / no Setting B)\n\n')
            for tag, key in (('microset', 'c2_audit_microset'),
                             ('microset-heldout', 'c2_audit_microset_heldout')):
                c = report[key]
                if not c.get('available', True):
                    f.write('%s: C2 ckpt missing at `%s`\n\n' % (tag, c.get('path')))
                    continue
                f.write('### %s\n\n' % tag)
                f.write('| mean Δconf (correct−shuffle) | mean self‖flow‖ | ok |\n')
                f.write('|---|---|---|\n')
                f.write('| %.4f | %.4f | %s |\n\n' % (
                    c['mean_delta_conf_correct_minus_shuffle'],
                    c['mean_self_flow'], c['match_quality_ok']))
        f.write('## Verdict\n\n')
        f.write('**Gate 2.2 %s**\n\n' % (
            'PASS' if report['gate2_2_pass'] else 'FAIL'))
        if report.get('stop_reason'):
            f.write('Stop: %s\n\n' % report['stop_reason'])
        f.write('Artifacts: `%s`, `%s`, `%s`\n' % (out_json, csv_path, md))
        f.write('\n## Repro\n\n```bash\n')
        f.write('cd %s && /root/miniconda3/envs/ttsr/bin/python '
                'scripts/gate2_2_align_info_quality.py --device cuda\n'
                % REPO)
        f.write('```\n')

    print(json.dumps({
        'gate2_2_pass': report['gate2_2_pass'],
        'g21_batch': report['g21_batch_eval'],
        'attention_selection_real': attn['attention_selection_real'],
        'tex_h4_microset': report['tex_h4_microset'],
        'clear_ref_utilization': report['clear_ref_utilization'],
        'c2': report.get('c2_audit_microset'),
    }, indent=2), flush=True)
    print('wrote %s' % md, flush=True)
    return 0 if report['gate2_2_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
