#!/usr/bin/env python
"""V8-B0 Gate 0 — setup + HARD checks (no training).

Strict-load frozen R1 SharedEncoder+LocalSoftMatch; re-init V3B0ResidualFusion;
LLFormer E110 Y0 cache; exclusivity vs microset/heldout; step0 identity;
chunked match; grad smoke.
"""

from __future__ import annotations

import json
import os
import sys

import torch

REPO = '/root/projects/TTSR-lowlight'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion, count_params  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha  # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402
from v3b_runtime import b0_forward, b0_loss, match_features, proposal_core  # noqa: E402
from v8_b0_runtime import (  # noqa: E402
    ARCHITECTURE, BASE_CKPT, EXCLUSIVE_IDS, GRAD_ACCUM, LR, LOSS, N_TRAIN,
    OPTIMIZER, PLAN_NOTE, R1_CKPT, REFERENCE_VARIANT, ROOT, SEED,
    SPLIT_JSON, SRC, UPDATES_GATE1, Y0_CACHE, MISMATCH_TRAIN, MISMATCH_DEV,
    setting_paths)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    paths = setting_paths(a.root)
    for d in paths.values():
        os.makedirs(d, exist_ok=True)
    for arm in ('B0_Self', 'B0_Nano'):
        os.makedirs(os.path.join(a.root, 'arms', arm, 'checkpoints'),
                    exist_ok=True)
        os.makedirs(os.path.join(a.root, 'arms', arm, 'logs'), exist_ok=True)

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force')

    ids = json.load(open(EXCLUSIVE_IDS))
    train_ids = list(ids['expand_exclusive_n32'])
    held_ids = list(ids['blocked_microset_heldout'])
    micro_ids = list(ids['blocked_microset'])
    formal_dev = list(ids['formal_dev'])
    if len(train_ids) != N_TRAIN:
        raise SystemExit('expected %d exclusive ids, got %d'
                         % (N_TRAIN, len(train_ids)))
    blocked = set(micro_ids) | set(held_ids)
    leak = sorted(set(train_ids) & blocked)
    if leak:
        raise SystemExit('train exclusivity FAIL: %s' % leak)
    if set(train_ids) & set(formal_dev):
        raise SystemExit('train overlaps formal_dev')

    # --- SHA / paths ---
    report = dict(
        gate='V8_B0_Gate0',
        plan_note=PLAN_NOTE,
        repo_commit=git_head(),
        r1_ckpt=R1_CKPT,
        r1_sha256=file_sha256(R1_CKPT),
        base_ckpt=BASE_CKPT,
        base_sha256=file_sha256(BASE_CKPT),
        y0_cache=Y0_CACHE,
        split_json=SPLIT_JSON,
        split_sha256=file_sha256(SPLIT_JSON),
        exclusive_ids_path=EXCLUSIVE_IDS,
        exclusive_ids_sha256=file_sha256(EXCLUSIVE_IDS),
        train_ids=train_ids,
        microset_heldout_ids=held_ids,
        formal_dev_ids=formal_dev,
        reference_variant=REFERENCE_VARIANT,
        architecture=ARCHITECTURE,
        seed=SEED, lr=LR, grad_accum=GRAD_ACCUM,
        updates_gate1=UPDATES_GATE1, loss=LOSS, optimizer=OPTIMIZER,
        official_test_allowed=False,
    )

    # --- strict load R1 ---
    wrapper = load_proposal(R1_CKPT, a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()
    report['r1_strict_load'] = True
    report['match_chunk'] = int(core.match.chunk)
    report['encoder_cls'] = type(core.encoder).__name__
    report['match_cls'] = type(core.match).__name__

    # --- re-init fusion head ---
    torch.manual_seed(SEED)
    head = V3B0ResidualFusion(in_ch=96)
    init_path = os.path.join(paths['init'], 'v8_b0_zero_out_s42.pt')
    sd = snapshot_(head)
    torch.save(dict(model=sd, architecture=ARCHITECTURE, seed=SEED), init_path)
    init_sha = state_dict_sha(sd)
    report['init_path'] = init_path
    report['init_sha'] = init_sha
    report['n_params'] = count_params(head)

    # synthetic step0 zero
    with torch.no_grad():
        d = head(torch.randn(1, 32, 8, 8), torch.randn(1, 32, 8, 8), (16, 16))
    report['synthetic_delta_max'] = float(d.abs().max())
    if report['synthetic_delta_max'] > 1e-7:
        raise SystemExit('init ΔY not zero')

    # --- data + real step0 / chunk match / grad smoke ---
    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    name_to = {r[0]: r for r in splits['train']}
    train_rows = [name_to[n] for n in train_ids]
    # verify Y0 cache files
    y0_dir = os.path.join(Y0_CACHE, 'refiner_train')
    missing = [n for n in train_ids
               if not os.path.isfile(os.path.join(y0_dir, n + '.npy'))]
    if missing:
        raise SystemExit('missing Y0 cache: %s' % missing[:5])
    report['y0_cache_train_ok'] = True

    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = REFERENCE_VARIANT
    mmap = json.load(open(MISMATCH_TRAIN))
    ds = make_dataset(ns, train_rows, y0_dir, mmap)

    head = head.to(a.device)
    t0 = sample_tensors(ds, 0, 'correct', a.device)
    with torch.no_grad():
        f0, tt = match_features(wrapper, t0['Y0'], t0['R'])
    report['match_feat_shape'] = dict(
        F0=list(f0.shape), T=list(tt.shape))
    if int(core.match.chunk) != int(EVAL_QUERY_CHUNK):
        raise SystemExit('chunk not set')
    report['chunked_match_ok'] = True

    y, dlt = b0_forward(head, f0.detach(), tt.detach(), t0['Y0'].detach())
    step0 = float(dlt.abs().max())
    report['step0_delta_max'] = step0
    if step0 > 1e-6:
        raise SystemExit('step0 identity FAIL: %s' % step0)
    report['step0_identity_ok'] = True

    # grad smoke: only head
    loss = b0_loss(y, t0['H'])
    loss.backward()
    g_out = float(head.out.weight.grad.abs().sum())
    report['grad_out_sum'] = g_out
    if g_out <= 0:
        raise SystemExit('head grad smoke FAIL')
    for n, p in core.encoder.named_parameters():
        if p.grad is not None:
            raise SystemExit('encoder got grad: %s' % n)
    for n, p in core.match.named_parameters():
        if p.grad is not None:
            raise SystemExit('match got grad: %s' % n)
    report['grad_smoke_ok'] = True
    report['frozen_encoder_match_ok'] = True

    # exclusivity summary
    report['data_exclusivity_ok'] = True
    report['gate0_pass'] = bool(
        report['r1_strict_load'] and report['step0_identity_ok']
        and report['chunked_match_ok'] and report['grad_smoke_ok']
        and report['data_exclusivity_ok'] and report['y0_cache_train_ok'])

    lock = dict(
        stage='V8-B0',
        root=a.root,
        repo_commit=report['repo_commit'],
        proposal_ckpt=R1_CKPT,
        proposal_sha256=report['r1_sha256'],
        base_ckpt=BASE_CKPT,
        base_ckpt_sha256=report['base_sha256'],
        y0_cache=Y0_CACHE,
        cache_name='cache_y0_llformer_e110',
        split_json=SPLIT_JSON,
        split_sha256=report['split_sha256'],
        mismatch_train=MISMATCH_TRAIN,
        mismatch_train_sha256=file_sha256(MISMATCH_TRAIN),
        mismatch_dev=MISMATCH_DEV,
        mismatch_dev_sha256=file_sha256(MISMATCH_DEV),
        reference_variant=REFERENCE_VARIANT,
        architecture=ARCHITECTURE,
        init_path=init_path,
        init_sha=init_sha,
        n_params=report['n_params'],
        optimizer=OPTIMIZER, lr=LR, weight_decay=0.0,
        updates=UPDATES_GATE1, grad_accum=GRAD_ACCUM, seed=SEED,
        pair_schedule_seed=SEED,
        official_test_allowed=False,
        train_ids=train_ids,
        microset_heldout_ids=held_ids,
        formal_dev_ids=formal_dev,
        exclusive_ids_sha256=report['exclusive_ids_sha256'],
        fusion_inputs=('F0', 'T', 'F0-T'),
        forbidden=('SKFF', 'DINO', 'C2', 'DCN', 'extra_gate',
                    'llformer_finetune', 'official_test_tuning'),
        arms=('B0_Self', 'B0_Nano'),
        plan_note=PLAN_NOTE,
    )
    dump_json(lock_path, lock)

    out_json = os.path.join(paths['gate0'], 'G0_report.json')
    dump_json(out_json, report)
    md = os.path.join(paths['gate0'], 'Gate0_report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# V8-B0 Gate 0\n\n')
        f.write('- **commit:** `%s`\n' % report['repo_commit'])
        f.write('- **R1 SHA:** `%s`\n' % report['r1_sha256'])
        f.write('- **Base E110 SHA:** `%s`\n' % report['base_sha256'])
        f.write('- **init_sha:** `%s`\n' % init_sha[:16])
        f.write('- **n_params:** %d\n' % report['n_params'])
        f.write('- **train n=32 exclusive:** OK (no ∩ micro/heldout/dev)\n')
        f.write('- **step0 identity:** %s\n' % report['step0_identity_ok'])
        f.write('- **chunked match:** %s (chunk=%d)\n'
                % (report['chunked_match_ok'], report['match_chunk']))
        f.write('- **grad smoke:** %s (out_grad=%.3e)\n'
                % (report['grad_smoke_ok'], g_out))
        f.write('\n**Gate 0 %s**\n' % (
            'PASS' if report['gate0_pass'] else 'FAIL'))
        f.write('\nNote: %s\n' % PLAN_NOTE)

    print(json.dumps(dict(
        gate0_pass=report['gate0_pass'],
        init_sha=init_sha[:16],
        step0=step0,
        grad_out=g_out,
        n_train=len(train_ids),
    ), indent=2), flush=True)
    print('wrote %s' % md, flush=True)
    return 0 if report['gate0_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
