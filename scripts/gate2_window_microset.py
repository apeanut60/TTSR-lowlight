#!/usr/bin/env python
"""Gate 2 — single-variable Fusion fix: spatial_window=3, correct-only, D2-only.

Loads Setting-A init (align frozen random as before). Does NOT start Setting B.
Measures Q/K grads, microset overfit, and Y_normal vs Y_swapA2 / shuffle.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

REPO = '/root/projects/TTSR-lowlight'
OUT = '/root/data/experiments/V7_CODE_REVIEW_FIX'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics  # noqa: E402
from model.V6LLFormerRefBridge import (  # noqa: E402
    load_frozen_llformer_base, tiled_v6_forward)
from model.V7Model import V7Model, ARCHITECTURE_WIN3_D2  # noqa: E402
from model.V7RiverFusion import V7RiverFusion  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3b_runtime import b0_loss  # noqa: E402
from v7_river_runtime import ROOT as EXP_ROOT, SRC  # noqa: E402


def _grad_norm(module, prefix):
    total = 0.0
    for n, p in module.named_parameters():
        if n.startswith(prefix) and p.grad is not None:
            total += float(p.grad.detach().norm())
    return total


def qk_grad_probe(spatial_window: int, device: str):
    torch.manual_seed(0)
    f = V7RiverFusion(64, 64, n_heads=4, head_dim=16,
                      spatial_window=spatial_window).to(device)
    with torch.no_grad():
        f.out_conv.weight.add_(1e-2)
    d = torch.randn(1, 64, 8, 8, device=device)
    a = torch.randn(1, 64, 8, 8, device=device)
    out, _ = f(d, a)
    out.sum().backward()
    return dict(
        spatial_window=spatial_window,
        n_keys=f.n_keys,
        q_grad=_grad_norm(f, 'q_proj'),
        k_grad=_grad_norm(f, 'k_proj'),
        v_grad=_grad_norm(f, 'v_proj'),
    )


def _install_identity_flow(model):
    """Force match flow=0, conf=1 so Align sees geometrically registered Ref."""
    orig = model.match_and_texture

    def wrapped(y0_m11, ref_m11):
        match, tex = orig(y0_m11, ref_m11)
        match.flow = torch.zeros_like(match.flow)
        match.confidence = torch.ones_like(match.confidence)
        return match, tex

    model.match_and_texture = wrapped
    return orig


def _forward_tiled(base, model, x, y0, r, roll_a2=False):
    """Tiled forward; optional spatial roll of aligned Ref at each tile's refine_h4."""
    if not roll_a2:
        return tiled_v6_forward(base, model, x, y0, r)
    orig = model.refine_h4.forward

    def wrapped(d, a):
        a = torch.roll(a, shifts=(3, 5), dims=(-2, -1))
        return orig(d, a)

    model.refine_h4.forward = wrapped
    try:
        return tiled_v6_forward(base, model, x, y0, r)
    finally:
        model.refine_h4.forward = orig


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--identity_flow', action='store_true',
                    help='Gate2 next variable: force flow=0/conf=1')
    ap.add_argument('--tag', default='',
                    help='output filename tag (default auto)')
    a = ap.parse_args(_CLI)

    tag = a.tag or ('idflow' if a.identity_flow else 'win3')
    os.makedirs(os.path.join(OUT, 'gate2'), exist_ok=True)
    report = dict(
        architecture=ARCHITECTURE_WIN3_D2,
        spatial_window=3, n_keys=9, d1_enabled=False,
        identity_flow=bool(a.identity_flow),
        states=['correct'], setting='A_fusion_only',
        n_images=a.n, updates=a.updates, lr=a.lr, seed=a.seed, tag=tag,
    )

    # --- Q/K grad unit probe ---
    g1 = qk_grad_probe(1, a.device)
    g3 = qk_grad_probe(3, a.device)
    report['qk_grad_n1'] = g1
    report['qk_grad_win3'] = g3
    report['qk_fixed'] = (
        g1['q_grad'] == 0 and g1['k_grad'] == 0 and g1['v_grad'] > 0
        and g3['q_grad'] > 0 and g3['k_grad'] > 0 and g3['v_grad'] > 0)
    print('Q/K probe n1=%s win3=%s fixed=%s' % (g1, g3, report['qk_fixed']),
          flush=True)
    if not report['qk_fixed']:
        raise SystemExit('window attention Q/K grad probe failed')

    arm = os.path.join(EXP_ROOT, 'A_fusion_only')
    lock = json.load(open(os.path.join(arm, 'artifact_lock.json')))
    init_path = lock['v7_init_path']
    report['v7_init_path'] = init_path
    report['base_ckpt'] = lock['base_ckpt']

    torch.manual_seed(a.seed)
    model = V7Model(spatial_window=3, d1_enabled=False).to(a.device)
    init = torch.load(init_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(init['model'], strict=False)
    # refine_* weights are shape-compatible; window is runtime-only
    report['load_missing'] = list(missing)
    report['load_unexpected'] = list(unexpected)
    if unexpected:
        raise SystemExit('unexpected keys: %s' % unexpected)
    model.apply_setting_freeze('A')
    for name in ('match_encoder', 'texture_encoder', 'matcher',
                 'align_h4', 'align_h2', 'refine_h2'):
        getattr(model, name).eval()
        for p in getattr(model, name).parameters():
            p.requires_grad_(False)
    if a.identity_flow:
        _install_identity_flow(model)

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)
    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       '/root/data/experiments/v3a4_lolv2real/splits/split.json')
    train_rows = splits['train'][:a.n]
    mmap = json.load(open(lock['mismatch_train']))
    ds = make_dataset(
        ns, train_rows,
        os.path.join(EXP_ROOT, lock['cache_name'], 'refiner_train'), mmap)
    pairs = [dict(i=i, name=train_rows[i][0], state='correct')
             for i in range(len(train_rows))]
    report['image_names'] = [p['name'] for p in pairs]

    # step0 identity (zero-init out); must use tiled path (Y0 is tiled Base)
    t0 = sample_tensors(ds, pairs[0]['i'], 'correct', a.device)
    with torch.no_grad():
        y0p = _forward_tiled(base, model, t0['X'], t0['Y0'], t0['R'])
    step0_mae = float((y0p - t0['Y0']).abs().max())
    report['step0_max_abs'] = step0_mae
    if step0_mae > 1e-3:
        raise SystemExit('step0 not identity: %s' % step0_mae)

    opt = torch.optim.Adam(
        [p for p in model.refine_h4.parameters() if p.requires_grad],
        lr=a.lr)
    log_path = os.path.join(OUT, 'gate2', 'train_microset_%s.log' % tag)
    losses = []
    t_wall = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        def log(msg):
            print(msg, flush=True)
            logf.write(msg + '\n')
            logf.flush()

        log('Gate2 microset n=%d updates=%d win=3 D2-only correct-only '
            'identity_flow=%s tag=%s'
            % (a.n, a.updates, a.identity_flow, tag))
        for step in range(1, a.updates + 1):
            model.refine_h4.train()
            opt.zero_grad(set_to_none=True)
            p = pairs[(step - 1) % len(pairs)]
            t = sample_tensors(ds, p['i'], 'correct', a.device)
            # train via tiled path with d1_enabled=False model
            y = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
            loss = b0_loss(y, t['H'])
            loss.backward()
            for p_b in base.parameters():
                if p_b.grad is not None:
                    raise SystemExit('Base got grad')
            for n, p_a in model.named_parameters():
                if (not n.startswith('refine_h4')) and p_a.grad is not None:
                    raise SystemExit('non-fusion grad: %s' % n)
            opt.step()
            lv = float(loss.detach())
            losses.append(lv)
            if step == 1 or step % 20 == 0 or step == a.updates:
                log('  step %d  loss=%.5f  (%.0fs)'
                    % (step, lv, time.time() - t_wall))

    report['loss_first'] = losses[0]
    report['loss_last'] = losses[-1]
    report['loss_min'] = min(losses)
    report['overfit_ok'] = losses[-1] < losses[0] * 0.95

    # --- post-train content response on microset ---
    model.eval()
    mae_swap, mae_shuf, dpsnr_ns = [], [], []
    qk_live = None
    for p in pairs:
        t = sample_tensors(ds, p['i'], 'correct', a.device)
        with torch.no_grad():
            y_n = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
            donor = pairs[(p['i'] + 1) % len(pairs)]
            td = sample_tensors(ds, donor['i'], 'correct', a.device)
            y_swap = _forward_tiled(
                base, model, t['X'], t['Y0'], t['R'], roll_a2=True)
            y_shuf = _forward_tiled(base, model, t['X'], t['Y0'], td['R'])
            mae_swap.append(float((y_n - y_swap).abs().mean()))
            mae_shuf.append(float((y_n - y_shuf).abs().mean()))
            pn, _, _ = _metrics(y_n, t['H'])
            ps, _, _ = _metrics(y_shuf, t['H'])
            dpsnr_ns.append(float(pn - ps))

        if qk_live is None:
            model.refine_h4.zero_grad(set_to_none=True)
            y = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
            b0_loss(y, t['H']).backward()
            qk_live = dict(
                q=_grad_norm(model.refine_h4, 'q_proj'),
                k=_grad_norm(model.refine_h4, 'k_proj'),
                v=_grad_norm(model.refine_h4, 'v_proj'),
            )
            opt.zero_grad(set_to_none=True)

    report['mae_y_swapA2_mean'] = float(sum(mae_swap) / len(mae_swap))
    report['mae_y_shuffleRef_mean'] = float(sum(mae_shuf) / len(mae_shuf))
    report['dpsnr_normal_minus_shuffle_mean'] = float(
        sum(dpsnr_ns) / len(dpsnr_ns))
    report['qk_grad_after_train'] = qk_live
    # Success: content response (swap) clearly above prior ~2e-4 dead zone
    report['fusion_responds'] = report['mae_y_swapA2_mean'] > 1e-3
    report['qk_live_ok'] = (
        qk_live is not None and qk_live['q'] > 0 and qk_live['k'] > 0)
    report['gate2_pass'] = bool(
        report['qk_fixed'] and report['overfit_ok']
        and report['fusion_responds'] and report['qk_live_ok'])

    ckpt = os.path.join(OUT, 'gate2', 'ckpt_microset_%s.pt' % tag)
    torch.save(dict(
        model=model.state_dict(), step=a.updates,
        architecture=ARCHITECTURE_WIN3_D2,
        spatial_window=3, d1_enabled=False, states=['correct'],
        identity_flow=bool(a.identity_flow),
    ), ckpt)
    report['ckpt'] = ckpt

    out_json = os.path.join(OUT, 'G2_window_microset_%s.json' % tag)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    # keep latest + cumulative markdown
    with open(os.path.join(OUT, 'G2_window_microset.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    md = os.path.join(OUT, 'G2_report.md')
    prev = ''
    if os.path.isfile(md):
        prev = open(md, encoding='utf-8').read() + '\n---\n\n'
    with open(md, 'w', encoding='utf-8') as f:
        f.write(prev)
        f.write('# Gate 2 — %s\n\n' % tag)
        f.write('```json\n%s\n```\n' % json.dumps(report, indent=2))
        f.write('\n**Gate 2 (%s) %s**\n' % (
            tag, 'PASS' if report['gate2_pass'] else 'FAIL'))
    print(json.dumps(report, indent=2), flush=True)
    print('wrote %s' % out_json, flush=True)
    return 0 if report['gate2_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
