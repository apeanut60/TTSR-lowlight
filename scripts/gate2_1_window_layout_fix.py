#!/usr/bin/env python
"""Gate 2.1 — re-run G2 win3 microset after F.unfold token-layout fix.

Fixed config (must match G2 win3):
  n=8, updates=200, lr=1e-4, seed=42, spatial_window=3,
  correct-only, D2-only, Setting A (frozen align), frozen LLFormer.
Also evaluates the same metrics on an independent heldout/dev set.
Does NOT start Setting B / joint train / Official Test.
"""

from __future__ import annotations

import json
import os
import subprocess
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

# Locked to original G2 win3
N_IMAGES = 8
UPDATES = 200
LR = 1e-4
SEED = 42
OLD_WIN3_JSON = os.path.join(OUT, 'G2_window_microset_win3.json')
HELDOUT = '/root/data/experiments/V7_REF_DIAG/heldout24.json'
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'


def _git_sha():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


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
        spatial_window=spatial_window, n_keys=f.n_keys,
        q_grad=_grad_norm(f, 'q_proj'),
        k_grad=_grad_norm(f, 'k_proj'),
        v_grad=_grad_norm(f, 'v_proj'),
    )


def _forward_tiled(base, model, x, y0, r, roll_a2=False):
    if not roll_a2:
        return tiled_v6_forward(base, model, x, y0, r)
    orig = model.refine_h4.forward

    def wrapped(d, a):
        return orig(d, torch.roll(a, shifts=(3, 5), dims=(-2, -1)))

    model.refine_h4.forward = wrapped
    try:
        return tiled_v6_forward(base, model, x, y0, r)
    finally:
        model.refine_h4.forward = orig


def _eval_pairs(base, model, ds, pairs, device):
    model.eval()
    mae_swap, mae_shuf, dpsnr, losses = [], [], [], []
    for p in pairs:
        t = sample_tensors(ds, p['i'], 'correct', device)
        with torch.no_grad():
            y_n = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
            donor = pairs[(pairs.index(p) + 1) % len(pairs)]
            td = sample_tensors(ds, donor['i'], 'correct', device)
            y_swap = _forward_tiled(
                base, model, t['X'], t['Y0'], t['R'], roll_a2=True)
            y_shuf = _forward_tiled(base, model, t['X'], t['Y0'], td['R'])
            mae_swap.append(float((y_n - y_swap).abs().mean()))
            mae_shuf.append(float((y_n - y_shuf).abs().mean()))
            pn, _, _ = _metrics(y_n, t['H'])
            ps, _, _ = _metrics(y_shuf, t['H'])
            dpsnr.append(float(pn - ps))
            losses.append(float(b0_loss(y_n, t['H'])))
    return dict(
        n=len(pairs),
        loss_mean=float(sum(losses) / len(losses)),
        mae_y_swapA2_mean=float(sum(mae_swap) / len(mae_swap)),
        mae_y_shuffleRef_mean=float(sum(mae_shuf) / len(mae_shuf)),
        dpsnr_normal_minus_shuffle_mean=float(sum(dpsnr) / len(dpsnr)),
    )


def _make_ds(lock, rows, device_unused=None):
    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = lock['reference_variant']
    mmap = json.load(open(lock['mismatch_train']))
    return make_dataset(
        ns, rows,
        os.path.join(EXP_ROOT, lock['cache_name'], 'refiner_train'), mmap)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip_train', action='store_true')
    a = ap.parse_args(_CLI)

    os.makedirs(os.path.join(OUT, 'gate2_1'), exist_ok=True)
    old = json.load(open(OLD_WIN3_JSON))
    sha = _git_sha()
    report = dict(
        gate='G2.1_window_layout_fix',
        baseline_commit='3aa86222af2af4d17d95b13beb6975427706b4d9',
        actual_commit=sha,
        architecture=ARCHITECTURE_WIN3_D2,
        spatial_window=3, n_keys=9, d1_enabled=False,
        identity_flow=False,
        states=['correct'], setting='A_fusion_only',
        frozen_llformer=True, frozen_alignment=True,
        n_images=N_IMAGES, updates=UPDATES, lr=LR, seed=SEED,
        optimizer='Adam', loss='MSE',
        old_win3_json=OLD_WIN3_JSON,
        fix='V7RiverFusion._window_kv: reshape(B,C,K*K,HW).permute(0,3,2,1)',
    )

    # --- static probes ---
    g1 = qk_grad_probe(1, a.device)
    g3 = qk_grad_probe(3, a.device)
    report['qk_grad_n1'] = g1
    report['qk_grad_win3'] = g3
    report['qk_fixed'] = (
        g1['q_grad'] == 0 and g1['k_grad'] == 0 and g1['v_grad'] > 0
        and g3['q_grad'] > 0 and g3['k_grad'] > 0 and g3['v_grad'] > 0)
    print('Q/K probe fixed=%s' % report['qk_fixed'], flush=True)
    if not report['qk_fixed']:
        raise SystemExit('Q/K probe failed')

    arm = os.path.join(EXP_ROOT, 'A_fusion_only')
    lock = json.load(open(os.path.join(arm, 'artifact_lock.json')))
    report['v7_init_path'] = lock['v7_init_path']
    report['base_ckpt'] = lock['base_ckpt']

    # Confirm image list matches old win3
    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    train_rows = splits['train'][:N_IMAGES]
    names = [r[0] for r in train_rows]
    report['image_names'] = names
    if names != old['image_names']:
        raise SystemExit('image list drift: %s vs %s' % (names, old['image_names']))
    if (old['updates'] != UPDATES or old['lr'] != LR or old['seed'] != SEED
            or old['n_images'] != N_IMAGES):
        raise SystemExit('config drift vs old win3 JSON')

    torch.manual_seed(SEED)
    model = V7Model(spatial_window=3, d1_enabled=False).to(a.device)
    init = torch.load(lock['v7_init_path'], map_location='cpu')
    miss, unexp = model.load_state_dict(init['model'], strict=False)
    report['load_missing'] = list(miss)
    report['load_unexpected'] = list(unexp)
    if unexp:
        raise SystemExit('unexpected keys %s' % unexp)
    model.apply_setting_freeze('A')
    for name in ('match_encoder', 'texture_encoder', 'matcher',
                 'align_h4', 'align_h2', 'refine_h2'):
        getattr(model, name).eval()
        for p in getattr(model, name).parameters():
            p.requires_grad_(False)

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)
    ds = _make_ds(lock, train_rows)
    pairs = [dict(i=i, name=train_rows[i][0], state='correct')
             for i in range(len(train_rows))]

    t0 = sample_tensors(ds, pairs[0]['i'], 'correct', a.device)
    with torch.no_grad():
        y0p = _forward_tiled(base, model, t0['X'], t0['Y0'], t0['R'])
    step0 = float((y0p - t0['Y0']).abs().max())
    report['step0_max_abs'] = step0
    if step0 > 1e-3:
        raise SystemExit('step0 not identity: %s' % step0)

    log_path = os.path.join(OUT, 'gate2_1', 'train_microset_win3_fixed.log')
    losses = []
    if not a.skip_train:
        opt = torch.optim.Adam(
            [p for p in model.refine_h4.parameters() if p.requires_grad], lr=LR)
        t_wall = time.time()
        with open(log_path, 'w', encoding='utf-8') as logf:
            def log(msg):
                print(msg, flush=True)
                logf.write(msg + '\n')
                logf.flush()

            log('G2.1 win3-fixed n=%d updates=%d seed=%d commit=%s'
                % (N_IMAGES, UPDATES, SEED, sha))
            for step in range(1, UPDATES + 1):
                model.refine_h4.train()
                opt.zero_grad(set_to_none=True)
                p = pairs[(step - 1) % len(pairs)]
                t = sample_tensors(ds, p['i'], 'correct', a.device)
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
                if step == 1 or step % 20 == 0 or step == UPDATES:
                    log('  step %d  loss=%.5f  (%.0fs)'
                        % (step, lv, time.time() - t_wall))
        report['loss_first'] = losses[0]
        report['loss_last'] = losses[-1]
        report['loss_min'] = min(losses)
        report['overfit_ok'] = losses[-1] < losses[0] * 0.95
        ckpt = os.path.join(OUT, 'gate2_1', 'ckpt_microset_win3_fixed.pt')
        torch.save(dict(
            model=model.state_dict(), step=UPDATES,
            architecture=ARCHITECTURE_WIN3_D2,
            spatial_window=3, d1_enabled=False, states=['correct'],
            seed=SEED, lr=LR, actual_commit=sha,
            layout_fix=True,
        ), ckpt)
        report['ckpt'] = ckpt
    else:
        ckpt = os.path.join(OUT, 'gate2_1', 'ckpt_microset_win3_fixed.pt')
        blob = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(blob['model'], strict=True)
        report['ckpt'] = ckpt
        report['loss_first'] = None
        report['loss_last'] = None
        report['overfit_ok'] = None

    # live Q/K after train
    model.refine_h4.zero_grad(set_to_none=True)
    t = sample_tensors(ds, pairs[0]['i'], 'correct', a.device)
    y = _forward_tiled(base, model, t['X'], t['Y0'], t['R'])
    b0_loss(y, t['H']).backward()
    report['qk_grad_after_train'] = dict(
        q=_grad_norm(model.refine_h4, 'q_proj'),
        k=_grad_norm(model.refine_h4, 'k_proj'),
        v=_grad_norm(model.refine_h4, 'v_proj'),
    )

    micro = _eval_pairs(base, model, ds, pairs, a.device)
    report['microset'] = micro

    # Independent heldout/dev (names from heldout24 that are in train split
    # but outside the microset — same as Gate1 heldout protocol)
    held = json.load(open(HELDOUT))
    name_to_row = {r[0]: r for r in splits['train']}
    # Prefer names not in microset
    micro_set = set(names)
    dev_names = [n for n in held['names'] if n in name_to_row and n not in micro_set][:16]
    if len(dev_names) < 8:
        # fall back to split['dev']
        dev_names = [r[0] for r in splits['dev'][:16]]
        name_to_row = {r[0]: r for r in splits['dev']}
        dev_rows = [name_to_row[n] for n in dev_names]
        # need Y0 cache — train cache may not cover; use train heldout subset
        raise SystemExit('insufficient heldout outside microset')
    dev_rows = [name_to_row[n] for n in dev_names]
    ds_dev = _make_ds(lock, dev_rows)
    # indices are 0..len(dev_rows)-1 in this dataset
    dev_pairs = [dict(i=i, name=dev_rows[i][0], state='correct')
                 for i in range(len(dev_rows))]
    report['dev_image_names'] = [p['name'] for p in dev_pairs]
    dev_m = _eval_pairs(base, model, ds_dev, dev_pairs, a.device)
    report['dev'] = dev_m

    report['old_win3'] = dict(
        loss_first=old['loss_first'], loss_last=old['loss_last'],
        mae_y_swapA2_mean=old['mae_y_swapA2_mean'],
        mae_y_shuffleRef_mean=old['mae_y_shuffleRef_mean'],
        dpsnr_normal_minus_shuffle_mean=old['dpsnr_normal_minus_shuffle_mean'],
        fusion_responds=old.get('fusion_responds'),
        gate2_pass=old.get('gate2_pass'),
    )
    report['delta_vs_old_micro'] = dict(
        loss_last=(None if report['loss_last'] is None else
                   report['loss_last'] - old['loss_last']),
        mae_swapA2=(micro['mae_y_swapA2_mean'] - old['mae_y_swapA2_mean']),
        mae_shuffleRef=(micro['mae_y_shuffleRef_mean']
                        - old['mae_y_shuffleRef_mean']),
        dpsnr=(micro['dpsnr_normal_minus_shuffle_mean']
               - old['dpsnr_normal_minus_shuffle_mean']),
    )

    # Pass: layout/qk/step0 already enforced; content response on microset
    report['fusion_responds'] = micro['mae_y_swapA2_mean'] > 1e-3
    report['improved_vs_old'] = (
        micro['mae_y_swapA2_mean'] > old['mae_y_swapA2_mean'] * 1.5
        or micro['mae_y_shuffleRef_mean'] > old['mae_y_shuffleRef_mean'] * 1.5)
    report['qk_live_ok'] = (
        report['qk_grad_after_train']['q'] > 0
        and report['qk_grad_after_train']['k'] > 0)
    report['gate2_1_pass'] = bool(
        report['qk_fixed'] and report.get('overfit_ok', True)
        and report['fusion_responds'] and report['qk_live_ok'])

    out_json = os.path.join(OUT, 'G2_1_window_layout_fix.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    repro = (
        '# unit tests\n'
        'cd %s && /root/miniconda3/envs/ttsr/bin/python -m unittest '
        'tests.test_v7_river -v\n'
        '# microset + dev eval\n'
        '/root/miniconda3/envs/ttsr/bin/python '
        '%s/scripts/gate2_1_window_layout_fix.py --device cuda\n'
        % (REPO, OUT))
    report['repro_commands'] = repro

    md = os.path.join(OUT, 'G2_1_window_layout_fix.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Gate 2.1 — Window Attention Layout Fix\n\n')
        f.write('- **baseline commit:** `%s`\n' % report['baseline_commit'])
        f.write('- **actual commit:** `%s`\n' % sha)
        f.write('- **seed / lr / updates / n:** %d / %s / %d / %d\n'
                % (SEED, LR, UPDATES, N_IMAGES))
        f.write('- **fix:** `%s`\n\n' % report['fix'])
        f.write('## Microset (train, same 8 images as G2 win3)\n\n')
        f.write('| metric | old win3 (buggy layout) | corrected win3 |\n')
        f.write('|---|---|---|\n')
        f.write('| loss_first | %.6f | %s |\n' % (
            old['loss_first'],
            '%.6f' % report['loss_first'] if report['loss_first'] is not None else 'n/a'))
        f.write('| loss_last | %.6f | %s |\n' % (
            old['loss_last'],
            '%.6f' % report['loss_last'] if report['loss_last'] is not None else 'n/a'))
        f.write('| mae_swapA2 | %.6e | %.6e |\n' % (
            old['mae_y_swapA2_mean'], micro['mae_y_swapA2_mean']))
        f.write('| mae_shuffleRef | %.6e | %.6e |\n' % (
            old['mae_y_shuffleRef_mean'], micro['mae_y_shuffleRef_mean']))
        f.write('| ΔPSNR n−shuf | %.6f | %.6f |\n\n' % (
            old['dpsnr_normal_minus_shuffle_mean'],
            micro['dpsnr_normal_minus_shuffle_mean']))
        f.write('## Independent heldout/dev (n=%d)\n\n' % dev_m['n'])
        f.write('| metric | value |\n|---|---|\n')
        f.write('| loss_mean | %.6f |\n' % dev_m['loss_mean'])
        f.write('| mae_swapA2 | %.6e |\n' % dev_m['mae_y_swapA2_mean'])
        f.write('| mae_shuffleRef | %.6e |\n' % dev_m['mae_y_shuffleRef_mean'])
        f.write('| ΔPSNR n−shuf | %.6f |\n\n' % (
            dev_m['dpsnr_normal_minus_shuffle_mean']))
        f.write('## Checks\n\n')
        f.write('- step0 identity: %s\n' % (step0 <= 1e-3))
        f.write('- Q/K win3 nonzero: %s\n' % report['qk_fixed'])
        f.write('- fusion_responds (swapA2>1e-3): %s\n' % report['fusion_responds'])
        f.write('- improved_vs_old: %s\n' % report['improved_vs_old'])
        f.write('\n**Gate 2.1 %s**\n\n' % (
            'PASS' if report['gate2_1_pass'] else 'FAIL'))
        f.write('## Repro\n\n```bash\n%s```\n' % repro)
        f.write('\n```json\n%s\n```\n' % json.dumps(report, indent=2))

    # rewrite json with repro
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in (
        'actual_commit', 'gate2_1_pass', 'fusion_responds', 'improved_vs_old',
        'microset', 'dev', 'delta_vs_old_micro', 'loss_first', 'loss_last')},
        indent=2), flush=True)
    print('wrote %s' % md, flush=True)
    return 0 if report['gate2_1_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
