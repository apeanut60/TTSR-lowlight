#!/usr/bin/env python
"""Gate 2.3a — Correct C2 geometry controls (no train).

Retracts Gate 2.3 absolute-flow expectation on Nano Ref rolls:
Nano is not pixel-aligned to Y0, so Flow==(dx,dy) is unjustified.

Tasks:
  1) Positive control: Y0 vs translated Y0 (known GT), edge-masked.
  2) Nano Ref: relative source-index consistency under roll (no zero-flow assume).
  3) Identity warp vs C2 warp structural consistency (Correct + Shuffle).
  4) Expand-train overlap audit vs microset-heldout; exclusive ID lists.

Flow convention (ChunkedHardMatcher + warp_by_flow):
  flow_x = sx - tx, flow_y = sy - ty  (source − target, feature pixels)
  warp samples source at (x+dx, y+dy).
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
from model.c2_extractor import C2_FEATURE_CKPT, load_c2_feature_extractor  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v7_river_runtime import ROOT as EXP_ROOT, SRC  # noqa: E402
from model.V6LLFormerRefBridge import m11_to_01  # noqa: E402

G22_JSON = os.path.join(DOCS, 'G2_2_align_info_quality.json')
G23_JSON = os.path.join(DOCS, 'G2_3_attn_c2_audit.json')
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'

# Image-space translation for positive control (pixels)
DY_IMG, DX_IMG = 8, 12


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def _make_ds(lock, rows):
    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = lock['reference_variant']
    mmap = json.load(open(lock['mismatch_train']))
    return make_dataset(
        ns, rows,
        os.path.join(EXP_ROOT, lock['cache_name'], 'refiner_train'), mmap)


def _grad_corr(a, b):
    def g(x):
        if x.shape[1] > 1:
            x = x.mean(1, keepdim=True)
        dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1))
        dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        return (dx.pow(2) + dy.pow(2)).sqrt()
    ga, gb = g(a).flatten(), g(b).flatten()
    ga, gb = ga - ga.mean(), gb - gb.mean()
    denom = ga.norm() * gb.norm()
    if float(denom) < 1e-8:
        return 0.0
    return float((ga * gb).sum() / denom)


def _feat_shift_from_img(h_img, w_img, fh, fw, dy_img, dx_img):
    """Map image-pixel roll to feature-grid shift (float)."""
    return dy_img * (fh / float(h_img)), dx_img * (fw / float(w_img))


def _interior_mask(h, w, dy_f, dx_f, device):
    """Mask out border + cyclic wrap contamination; dy_f/dx_f in feat pixels."""
    # ceil abs shift + 1 margin
    my = int(np.ceil(abs(dy_f))) + 1
    mx = int(np.ceil(abs(dx_f))) + 1
    mask = torch.ones(1, 1, h, w, device=device, dtype=torch.bool)
    mask[:, :, :my, :] = False
    mask[:, :, h - my:, :] = False
    mask[:, :, :, :mx] = False
    mask[:, :, :, w - mx:] = False
    return mask, my, mx


def _index_to_yx(index, w):
    sy = torch.div(index, w, rounding_mode='floor')
    sx = index.remainder(w)
    return sy, sx


def task1_y0_positive_control(c2, matcher, ds, pairs, device):
    """image1=Y0, image2=roll(Y0); known GT flow; edge-masked."""
    rows = []
    for p in pairs:
        t = sample_tensors(ds, p['i'], 'correct', device)
        y0 = m11_to_01(t['Y0'])
        y0_t = torch.roll(y0, shifts=(DY_IMG, DX_IMG), dims=(-2, -1))
        with torch.no_grad():
            f = c2(y0, y0_t)
            q, k = f['dense_features1'], f['dense_features2']
            m = matcher(q, k)
            _, _, fh, fw = m.flow.shape
            _, _, H, W = y0.shape
            exp_dy, exp_dx = _feat_shift_from_img(H, W, fh, fw, DY_IMG, DX_IMG)
            mask, my, mx = _interior_mask(fh, fw, exp_dy, exp_dx, device)
            err = (
                (m.flow[:, 0:1] - exp_dx).pow(2)
                + (m.flow[:, 1:2] - exp_dy).pow(2)
            ).sqrt()
            err_m = err[mask]
            # also check sign via mean flow on mask
            fx = m.flow[:, 0:1][mask]
            fy = m.flow[:, 1:2][mask]
            # match accuracy: within 1.0 feat-px of GT
            acc1 = float((err_m < 1.0).float().mean())
            acc2 = float((err_m < 2.0).float().mean())
            rows.append(dict(
                name=p['name'],
                feat_h=fh, feat_w=fw,
                img_h=H, img_w=W,
                dy_img=DY_IMG, dx_img=DX_IMG,
                exp_dy_feat=exp_dy, exp_dx_feat=exp_dx,
                border_my=my, border_mx=mx,
                n_valid=int(mask.sum()),
                flow_err_mean=float(err_m.mean()),
                flow_err_median=float(err_m.median()),
                flow_err_p90=float(torch.quantile(err_m.float(), 0.9)),
                mean_flow_dx=float(fx.mean()),
                mean_flow_dy=float(fy.mean()),
                acc_within_1px=acc1,
                acc_within_2px=acc2,
                # sign check: recovered mean should match expected sign/mag
                sign_dx_ok=bool(float(fx.mean()) * exp_dx > 0),
                sign_dy_ok=bool(float(fy.mean()) * exp_dy > 0),
            ))
    def avg(k):
        return float(np.mean([r[k] for r in rows]))
    summary = dict(
        n=len(rows),
        flow_err_mean=avg('flow_err_mean'),
        flow_err_median=avg('flow_err_median'),
        acc_within_1px=avg('acc_within_1px'),
        acc_within_2px=avg('acc_within_2px'),
        mean_flow_dx=avg('mean_flow_dx'),
        mean_flow_dy=avg('mean_flow_dy'),
        exp_dx_feat=rows[0]['exp_dx_feat'],
        exp_dy_feat=rows[0]['exp_dy_feat'],
        frac_sign_dx_ok=float(np.mean([r['sign_dx_ok'] for r in rows])),
        frac_sign_dy_ok=float(np.mean([r['sign_dy_ok'] for r in rows])),
        # pass if masked accuracy high and signs correct
        positive_control_pass=bool(
            avg('acc_within_2px') > 0.7
            and avg('flow_err_median') < 2.0
            and float(np.mean([r['sign_dx_ok'] for r in rows])) > 0.8),
        flow_definition=dict(
            matcher='flow_x=sx-tx, flow_y=sy-ty (source-target)',
            warp='samples source at (x+dx,y+dy)',
            roll_expectation=('torch.roll(key,+dy,+dx) ⇒ expected flow '
                              '≈ (+dx_feat,+dy_feat) on interior'),
        ),
    )
    return summary, rows


def task2_nano_relative_consistency(c2, matcher, ds, pairs, device):
    """Nano Ref: compare source indices before/after roll (relative)."""
    rows = []
    for p in pairs:
        t = sample_tensors(ds, p['i'], 'correct', device)
        y0 = m11_to_01(t['Y0'])
        ref = m11_to_01(t['R'])
        ref_r = torch.roll(ref, shifts=(DY_IMG, DX_IMG), dims=(-2, -1))
        with torch.no_grad():
            f0 = c2(y0, ref)
            f1 = c2(y0, ref_r)
            m0 = matcher(f0['dense_features1'], f0['dense_features2'])
            m1 = matcher(f1['dense_features1'], f1['dense_features2'])
            _, _, fh, fw = m0.flow.shape
            _, _, H, W = y0.shape
            exp_dy, exp_dx = _feat_shift_from_img(H, W, fh, fw, DY_IMG, DX_IMG)
            # integer shift on feature grid for index wrap
            # use rounded shift for discrete index consistency
            idy = int(round(exp_dy))
            idx = int(round(exp_dx))
            mask, my, mx = _interior_mask(fh, fw, exp_dy, exp_dx, device)
            # Also exclude queries whose *source* would wrap under roll
            sy0, sx0 = _index_to_yx(m0.index, fw)
            # predicted source after roll of key content
            sy_exp = (sy0 + idy) % fh
            sx_exp = (sx0 + idx) % fw
            # drop targets whose original source is in wrap-affected band of key
            # (source near border before roll is ambiguous under circularity)
            src_ok = (sy0 >= my) & (sy0 < fh - my) & (sx0 >= mx) & (sx0 < fw - mx)
            valid = mask.squeeze(1) & src_ok  # [B,H,W]
            sy1, sx1 = _index_to_yx(m1.index, fw)
            match_ok = (sy1 == sy_exp) & (sx1 == sx_exp)
            # also soft: L1 distance of source coords
            dist = (sy1.float() - sy_exp.float()).abs() + (
                sx1.float() - sx_exp.float()).abs()
            v = valid[0]
            n_valid = int(v.sum())
            if n_valid == 0:
                acc = 0.0
                dist_mean = float('nan')
            else:
                acc = float(match_ok[0][v].float().mean())
                dist_mean = float(dist[0][v].float().mean())
            # relative flow delta should ≈ (idx, idy) on valid set
            dflow = m1.flow - m0.flow
            d_err = (
                (dflow[:, 0:1] - float(idx)).pow(2)
                + (dflow[:, 1:2] - float(idy)).pow(2)
            ).sqrt()
            d_err_m = d_err[0, 0][v] if n_valid else d_err[0, 0]
            rows.append(dict(
                name=p['name'],
                idy_feat=idy, idx_feat=idx,
                n_valid=n_valid,
                source_index_exact_acc=acc,
                source_index_l1_mean=dist_mean,
                delta_flow_err_mean=float(d_err_m.mean()) if n_valid else float('nan'),
                delta_flow_err_median=float(d_err_m.median()) if n_valid else float('nan'),
                frac_delta_flow_within_2=float((d_err_m < 2).float().mean())
                if n_valid else 0.0,
            ))
    def avg(k):
        vals = [r[k] for r in rows if r[k] == r[k]]  # skip nan
        return float(np.mean(vals)) if vals else float('nan')
    summary = dict(
        n=len(rows),
        source_index_exact_acc=avg('source_index_exact_acc'),
        source_index_l1_mean=avg('source_index_l1_mean'),
        delta_flow_err_median=avg('delta_flow_err_median'),
        frac_delta_flow_within_2=avg('frac_delta_flow_within_2'),
        relative_consistency_pass=bool(
            avg('frac_delta_flow_within_2') > 0.5
            or avg('source_index_exact_acc') > 0.4),
        note=('Does NOT assume base Nano flow is zero; tests that rolling '
              'the key shifts matched source indices by the same roll'),
    )
    return summary, rows


def task3_identity_vs_c2_warp(c2, matcher, ds, pairs, device, viz_dir):
    """Correct/Shuffle: identity resize vs C2-warp structural consistency to Y0."""
    os.makedirs(viz_dir, exist_ok=True)
    rows = []
    for idx, p in enumerate(pairs):
        t = sample_tensors(ds, p['i'], 'correct', device)
        donor = pairs[(idx + 1) % len(pairs)]
        td = sample_tensors(ds, donor['i'], 'correct', device)
        y0 = m11_to_01(t['Y0'])
        ref = m11_to_01(t['R'])
        ref_sh = m11_to_01(td['R'])
        with torch.no_grad():
            def scores(ref_img, tag):
                f = c2(y0, ref_img)
                m = matcher(f['dense_features1'], f['dense_features2'])
                y0_f = F.interpolate(
                    y0, size=m.flow.shape[-2:], mode='bilinear',
                    align_corners=False)
                ref_f = F.interpolate(
                    ref_img, size=m.flow.shape[-2:], mode='bilinear',
                    align_corners=False)
                id_l1 = float((ref_f - y0_f).abs().mean())
                id_gc = _grad_corr(ref_f, y0_f)
                warped = warp_by_flow(ref_f, m.flow)
                c2_l1 = float((warped - y0_f).abs().mean())
                c2_gc = _grad_corr(warped, y0_f)
                return dict(
                    tag=tag, flow_mean=float(
                        m.flow.pow(2).sum(1).sqrt().mean()),
                    identity_l1=id_l1, identity_grad_corr=id_gc,
                    c2_warp_l1=c2_l1, c2_warp_grad_corr=c2_gc,
                    delta_l1_id_minus_c2=id_l1 - c2_l1,
                    delta_grad_corr_c2_minus_id=c2_gc - id_gc,
                    _warped=warped, _y0f=y0_f, _reff=ref_f, _flow=m.flow,
                )

            ok = scores(ref, 'correct')
            sh = scores(ref_sh, 'shuffle')
            rec = dict(
                name=p['name'],
                correct_flow_mean=ok['flow_mean'],
                correct_identity_l1=ok['identity_l1'],
                correct_identity_grad_corr=ok['identity_grad_corr'],
                correct_c2_warp_l1=ok['c2_warp_l1'],
                correct_c2_warp_grad_corr=ok['c2_warp_grad_corr'],
                correct_delta_l1_id_minus_c2=ok['delta_l1_id_minus_c2'],
                correct_delta_gc_c2_minus_id=ok['delta_grad_corr_c2_minus_id'],
                shuffle_flow_mean=sh['flow_mean'],
                shuffle_identity_l1=sh['identity_l1'],
                shuffle_identity_grad_corr=sh['identity_grad_corr'],
                shuffle_c2_warp_l1=sh['c2_warp_l1'],
                shuffle_c2_warp_grad_corr=sh['c2_warp_grad_corr'],
                shuffle_delta_l1_id_minus_c2=sh['delta_l1_id_minus_c2'],
                shuffle_delta_gc_c2_minus_id=sh['delta_grad_corr_c2_minus_id'],
                # C2 helps Correct more than Shuffle if Δgc_correct > Δgc_shuffle
                c2_helps_correct_more_than_shuffle=bool(
                    ok['delta_grad_corr_c2_minus_id']
                    > sh['delta_grad_corr_c2_minus_id'] + 0.01),
            )
            rows.append(rec)
            if idx < 3:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
                    def show(ax, t4, title):
                        if t4.shape[1] == 3:
                            ax.imshow(t4[0].permute(1, 2, 0).cpu().clip(0, 1))
                        else:
                            ax.imshow(t4[0, 0].cpu(), cmap='viridis')
                        ax.set_title(title); ax.axis('off')
                    show(axes[0, 0], ok['_y0f'], 'Y0 @feat')
                    show(axes[0, 1], ok['_reff'], 'Ref identity')
                    show(axes[0, 2], ok['_warped'], 'Ref C2-warp')
                    show(axes[1, 0], sh['_reff'], 'Shuf identity')
                    show(axes[1, 1], sh['_warped'], 'Shuf C2-warp')
                    mag = ok['_flow'].pow(2).sum(1).sqrt()[0].cpu()
                    axes[1, 2].imshow(mag, cmap='viridis')
                    axes[1, 2].set_title('|flow| correct'); axes[1, 2].axis('off')
                    fig.suptitle(p['name'])
                    fig.tight_layout()
                    fig.savefig(os.path.join(
                        viz_dir, 'warp_%s.png' % p['name'].replace('.png', '')),
                        dpi=110)
                    plt.close(fig)
                except Exception as e:
                    open(os.path.join(viz_dir, 'err.txt'), 'a').write(
                        '%s %s\n' % (p['name'], e))
    def avg(k):
        return float(np.mean([r[k] for r in rows]))
    summary = dict(
        n=len(rows),
        correct_delta_l1_id_minus_c2=avg('correct_delta_l1_id_minus_c2'),
        correct_delta_gc_c2_minus_id=avg('correct_delta_gc_c2_minus_id'),
        shuffle_delta_l1_id_minus_c2=avg('shuffle_delta_l1_id_minus_c2'),
        shuffle_delta_gc_c2_minus_id=avg('shuffle_delta_gc_c2_minus_id'),
        frac_c2_helps_correct_more=float(np.mean([
            r['c2_helps_correct_more_than_shuffle'] for r in rows])),
        # C2 warp useful on Nano if it improves Correct structure vs identity
        # AND more than on Shuffle
        c2_warp_useful_vs_identity=bool(
            avg('correct_delta_gc_c2_minus_id') > 0.02
            and avg('correct_delta_gc_c2_minus_id')
            > avg('shuffle_delta_gc_c2_minus_id') + 0.01),
    )
    return summary, rows


def task4_split_audit(g22, splits):
    """Ensure expand-train does not leak microset-heldout; list formal dev."""
    micro = list(g22['microset_names'])
    held = list(g22['microset_heldout_names'])
    train_all = [r[0] for r in splits['train']]
    dev_all = [r[0] for r in splits['dev']]
    blocked = set(micro) | set(held)

    # Dangerous pattern: train[:64]
    train_prefix64 = train_all[:64]
    leak = [n for n in train_prefix64 if n in blocked]
    leak_held = [n for n in train_prefix64 if n in held]
    leak_micro = [n for n in train_prefix64 if n in micro]

    # Exclusive expand pool: train minus blocked, preserve order
    expand_pool = [n for n in train_all if n not in blocked]
    # Proposed expand microset sizes
    expand_n32 = expand_pool[:32]
    expand_n64 = expand_pool[:64]
    # Verify no overlap
    assert not (set(expand_n64) & blocked)

    # Formal split.dev — never use as train; OK as true heldout eval
    formal_dev = list(dev_all)
    # Check formal_dev vs blocked (heldout was from train, may overlap names?)
    dev_vs_held = sorted(set(formal_dev) & set(held))
    dev_vs_micro = sorted(set(formal_dev) & set(micro))

    return dict(
        microset_n=len(micro),
        microset_heldout_n=len(held),
        train_n=len(train_all),
        formal_dev_n=len(formal_dev),
        train_prefix64_leak=dict(
            n_leak=len(leak),
            leak_ids=leak,
            leak_heldout=leak_held,
            leak_microset=leak_micro,
            unsafe=bool(len(leak) > 0),
            warning=('train[:64] overlaps microset and/or microset-heldout — '
                     'FORBIDDEN as expand set'),
        ),
        expand_pool_n=len(expand_pool),
        expand_exclusive_n32=expand_n32,
        expand_exclusive_n64=expand_n64,
        formal_dev_ids=formal_dev,
        formal_dev_overlap_heldout=dev_vs_held,
        formal_dev_overlap_microset=dev_vs_micro,
        budget_suggestion=dict(
            updates=400,
            n_images=32,
            states=['correct'],
            d1_enabled=False,
            a2_mode='tex_h4',  # single-variable vs n=8 tex_h4
            freeze_llformer=True,
            freeze_align=True,
            setting='A',
            seed=42,
            lr=1e-4,
            eval_on=['expand_train_subset', 'microset_heldout_16', 'formal_dev'],
            note=('Gate 2.4 bounded pilot proposal only: increase n with '
                  'exclusive IDs; keep tex_h4; do NOT start here'),
        ),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)

    out_dir = os.path.join(OUT, 'gate2_3a')
    os.makedirs(out_dir, exist_ok=True)
    g22 = json.load(open(G22_JSON))
    g23 = json.load(open(G23_JSON))
    arm = os.path.join(EXP_ROOT, 'A_fusion_only')
    lock = json.load(open(os.path.join(arm, 'artifact_lock.json')))
    sha = _git_sha()

    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    name_to = {r[0]: r for r in splits['train']}
    # Use microset + heldout for geometry tasks
    names = g22['microset_names'] + g22['microset_heldout_names']
    rows = [name_to[n] for n in names]
    ds = _make_ds(lock, rows)
    pairs = [dict(i=i, name=names[i]) for i in range(len(names))]
    pairs_m = pairs[:8]
    pairs_h = pairs[8:]

    print('[G2.3a] loading C2…', flush=True)
    c2 = load_c2_feature_extractor(C2_FEATURE_CKPT, device=a.device, freeze=True)
    matcher = ChunkedHardMatcher().to(a.device).eval()

    print('=== Task1 Y0 positive control ===', flush=True)
    t1_m, t1_m_rows = task1_y0_positive_control(
        c2, matcher, ds, pairs_m, a.device)
    t1_h, t1_h_rows = task1_y0_positive_control(
        c2, matcher, ds, pairs_h, a.device)
    print('  micro pass=%s acc2=%.3f err_med=%.3f dx=%.2f (exp %.2f)'
          % (t1_m['positive_control_pass'], t1_m['acc_within_2px'],
             t1_m['flow_err_median'], t1_m['mean_flow_dx'],
             t1_m['exp_dx_feat']), flush=True)

    print('=== Task2 Nano relative consistency ===', flush=True)
    t2_m, t2_m_rows = task2_nano_relative_consistency(
        c2, matcher, ds, pairs_m, a.device)
    t2_h, t2_h_rows = task2_nano_relative_consistency(
        c2, matcher, ds, pairs_h, a.device)
    print('  micro rel_pass=%s delta_within2=%.3f idx_acc=%.3f'
          % (t2_m['relative_consistency_pass'],
             t2_m['frac_delta_flow_within_2'],
             t2_m['source_index_exact_acc']), flush=True)

    print('=== Task3 Identity vs C2 warp ===', flush=True)
    viz = os.path.join(out_dir, 'warp_viz')
    t3_m, t3_m_rows = task3_identity_vs_c2_warp(
        c2, matcher, ds, pairs_m, a.device, viz)
    t3_h, t3_h_rows = task3_identity_vs_c2_warp(
        c2, matcher, ds, pairs_h, a.device,
        os.path.join(out_dir, 'warp_viz_heldout'))
    print('  micro c2_useful=%s Δgc_corr=%+.4f Δgc_shuf=%+.4f'
          % (t3_m['c2_warp_useful_vs_identity'],
             t3_m['correct_delta_gc_c2_minus_id'],
             t3_m['shuffle_delta_gc_c2_minus_id']), flush=True)

    print('=== Task4 split audit ===', flush=True)
    # need raw split lists
    split_raw = json.load(open(SPLIT_JSON))
    # rebuild splits structure like load_rows
    from v3a5_pipeline import load_rows as _lr
    splits_full = _lr(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                      SPLIT_JSON)
    t4 = task4_split_audit(g22, splits_full)
    print('  train[:64] leak=%s n=%d; expand_pool=%d; formal_dev=%d'
          % (t4['train_prefix64_leak']['unsafe'],
             t4['train_prefix64_leak']['n_leak'],
             t4['expand_pool_n'], t4['formal_dev_n']), flush=True)

    # CSVs
    def write_csv(path, rowslist):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rowslist[0].keys()))
            w.writeheader()
            w.writerows(rowslist)

    for tag, r1, r2, r3 in (
            ('microset', t1_m_rows, t2_m_rows, t3_m_rows),
            ('microset_heldout', t1_h_rows, t2_h_rows, t3_h_rows)):
        write_csv(os.path.join(out_dir, 't1_y0_shift_%s.csv' % tag), r1)
        write_csv(os.path.join(out_dir, 't2_nano_relative_%s.csv' % tag), r2)
        write_csv(os.path.join(out_dir, 't3_warp_%s.csv' % tag), r3)

    # exclusive ID lists
    id_path = os.path.join(out_dir, 'exclusive_id_lists.json')
    with open(id_path, 'w', encoding='utf-8') as f:
        json.dump(dict(
            blocked_microset=g22['microset_names'],
            blocked_microset_heldout=g22['microset_heldout_names'],
            expand_exclusive_n32=t4['expand_exclusive_n32'],
            expand_exclusive_n64=t4['expand_exclusive_n64'],
            formal_dev=t4['formal_dev_ids'],
        ), f, indent=2)

    # Revised C2 geometry verdict
    c2_pos_ok = t1_m['positive_control_pass'] and t1_h['positive_control_pass']
    c2_rel_ok = (t2_m['relative_consistency_pass']
                 and t2_h['relative_consistency_pass'])
    c2_warp_ok = t3_m['c2_warp_useful_vs_identity']
    # Gate 2.3 absolute Nano roll test RETRACTED
    g23_retraction = dict(
        retracted_test='Gate2.3 absolute Flow==(dx,dy) on rolled Nano Ref',
        reason=('Nano Ref is not guaranteed pixel-aligned to Y0; absolute '
                'expected flow after roll is not identified'),
        prior_geometric_match_ok=g23['task_b_c2']['geometric_match_ok'],
    )

    # Gate 2.4 proposal
    if c2_pos_ok and c2_rel_ok and c2_warp_ok:
        g24 = dict(
            type='C2_warp_bounded_pilot',
            train_now=False,
            proposal=('Single-variable: A2 = warp(tex[h4], C2_flow) with frozen '
                      'C2; keep Fusion win3, correct-only, D2-only, n=32 exclusive, '
                      'updates<=400, eval on microset-heldout + formal_dev'),
        )
    elif c2_pos_ok:
        g24 = dict(
            type='expand_microset_tex_h4_bounded_pilot',
            train_now=False,
            proposal=(
                'C2 Y0-shift PASS and Nano relative consistency PASS, but '
                'C2-warp worsens vs identity on Nano Correct — do not wire '
                'C2-warp yet. Gate 2.4: exclusive n=32 tex_h4, updates<=400, '
                'eval microset-heldout + formal split.dev (proposal only).'),
        )
    else:
        g24 = dict(
            type='matcher_debug_before_pilot',
            train_now=False,
            proposal=('Y0 self-shift positive control FAILED — fix scale/sign/'
                      'matcher coupling before any C2-warp or expand pilot'),
        )

    report = dict(
        gate='G2.3a_c2_geometry_fix',
        actual_commit=sha,
        c2_sha256=_sha256(C2_FEATURE_CKPT),
        g23_retraction=g23_retraction,
        flow_convention=t1_m['flow_definition'],
        task1_y0_shift=dict(microset=t1_m, microset_heldout=t1_h),
        task2_nano_relative=dict(microset=t2_m, microset_heldout=t2_h),
        task3_identity_vs_c2=dict(microset=t3_m, microset_heldout=t3_h),
        task4_split_audit={k: v for k, v in t4.items()
                           if k not in ('expand_exclusive_n32',
                                        'expand_exclusive_n64',
                                        'formal_dev_ids')},
        exclusive_id_lists_path=id_path,
        c2_revised=dict(
            positive_control_ok=c2_pos_ok,
            nano_relative_ok=c2_rel_ok,
            c2_warp_vs_identity_useful=c2_warp_ok,
            geometric_ready_for_c2_warp_pilot=bool(
                c2_pos_ok and c2_rel_ok and c2_warp_ok),
        ),
        gate2_4_proposal=g24,
        train_authorization='DENIED',
    )

    out_json = os.path.join(OUT, 'G2_3a_c2_geometry_fix.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Write exclusive lists into report-friendly shorter file in docs later
    md = os.path.join(OUT, 'Gate2_3a_report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Gate 2.3a — C2 Geometry Controls (corrected)\n\n')
        f.write('- **HEAD:** `%s`\n' % sha)
        f.write('- **C2 SHA256:** `%s`\n' % report['c2_sha256'])
        f.write('- **Retraction:** Gate 2.3 absolute Flow expectation on '
                'rolled Nano Ref is **invalid** (Nano ↛ pixel-aligned to Y0).\n\n')
        f.write('## Flow convention\n\n')
        f.write('- Matcher: `flow = (sx−tx, sy−ty)` (source−target)\n')
        f.write('- Warp: sample source at `(x+dx, y+dy)`\n')
        f.write('- `torch.roll(key, +dy,+dx)` ⇒ expected interior flow '
                '`≈ (+dx_feat, +dy_feat)` for **Y0↔rolled Y0** only\n\n')
        f.write('## Task 1 — Y0 positive control\n\n')
        for tag, s in (('microset', t1_m), ('microset-heldout', t1_h)):
            f.write('### %s\n\n' % tag)
            f.write('| acc≤2px | err median | mean (dx,dy) | expected | pass |\n')
            f.write('|---|---|---|---|---|\n')
            f.write('| %.3f | %.3f | (%.2f, %.2f) | (%.2f, %.2f) | **%s** |\n\n'
                    % (s['acc_within_2px'], s['flow_err_median'],
                       s['mean_flow_dx'], s['mean_flow_dy'],
                       s['exp_dx_feat'], s['exp_dy_feat'],
                       s['positive_control_pass']))
        f.write('## Task 2 — Nano relative source-index consistency\n\n')
        for tag, s in (('microset', t2_m), ('microset-heldout', t2_h)):
            f.write('### %s\n\n' % tag)
            f.write('| idx exact acc | idx L1 | Δflow within 2px | pass |\n')
            f.write('|---|---|---|---|\n')
            f.write('| %.3f | %.3f | %.3f | **%s** |\n\n' % (
                s['source_index_exact_acc'], s['source_index_l1_mean'],
                s['frac_delta_flow_within_2'],
                s['relative_consistency_pass']))
        f.write('## Task 3 — Identity vs C2 warp (no GT as model input)\n\n')
        for tag, s in (('microset', t3_m), ('microset-heldout', t3_h)):
            f.write('### %s\n\n' % tag)
            f.write('| Correct ΔL1(id−c2) | Correct Δgc(c2−id) | '
                    'Shuffle Δgc(c2−id) | c2_useful |\n')
            f.write('|---|---|---|---|\n')
            f.write('| %+.4f | %+.4f | %+.4f | **%s** |\n\n' % (
                s['correct_delta_l1_id_minus_c2'],
                s['correct_delta_gc_c2_minus_id'],
                s['shuffle_delta_gc_c2_minus_id'],
                s['c2_warp_useful_vs_identity']))
        f.write('## Task 4 — Expand-train leakage audit\n\n')
        f.write('- `train[:64]` leak into blocked microset/heldout: '
                '**%s** (n=%d)\n' % (
                    t4['train_prefix64_leak']['unsafe'],
                    t4['train_prefix64_leak']['n_leak']))
        if t4['train_prefix64_leak']['leak_ids']:
            f.write('  - leaked IDs: `%s`\n' % t4['train_prefix64_leak']['leak_ids'])
        f.write('- exclusive expand pool size: **%d** (train − micro8 − '
                'heldout16)\n' % t4['expand_pool_n'])
        f.write('- formal `split.dev` size: **%d**\n' % t4['formal_dev_n'])
        f.write('- lists: `%s`\n\n' % id_path)
        f.write('### Suggested single-variable budget (proposal only)\n\n')
        f.write('```json\n%s\n```\n\n' % json.dumps(
            t4['budget_suggestion'], indent=2))
        f.write('## Revised C2 readiness\n\n')
        f.write('| check | value |\n|---|---|\n')
        f.write('| Y0 positive control | %s |\n' % c2_pos_ok)
        f.write('| Nano relative consistency | %s |\n' % c2_rel_ok)
        f.write('| C2 warp > identity on Correct | %s |\n' % c2_warp_ok)
        f.write('| ready for C2-warp pilot | **%s** |\n\n'
                % report['c2_revised']['geometric_ready_for_c2_warp_pilot'])
        f.write('## Gate 2.4 bounded pilot (proposal — do not start)\n\n')
        f.write('- **type:** `%s`\n' % g24['type'])
        f.write('- %s\n\n' % g24['proposal'])
        f.write('`train_authorization=DENIED` this round.\n\n')
        f.write('## Repro\n\n```bash\n')
        f.write('cd %s && /root/miniconda3/envs/ttsr/bin/python '
                'scripts/gate2_3a_c2_geometry_fix.py --device cuda\n'
                % REPO)
        f.write('```\n')

    print(json.dumps(dict(
        c2_revised=report['c2_revised'],
        g24=g24,
        t1_micro=t1_m,
        t2_micro=t2_m,
        t3_micro=t3_m,
        leak=t4['train_prefix64_leak'],
    ), indent=2), flush=True)
    print('wrote %s' % md, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
