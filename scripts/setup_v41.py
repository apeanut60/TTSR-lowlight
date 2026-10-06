#!/usr/bin/env python
"""V4.1a setup: freeze B0@20k + zero-init RefGrounder; HARD step0 Y==B0."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK                       # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion, count_params    # noqa: E402
from model.V4RefGrounder import (FORWARD_DIR, GROUND_DIR, RefGrounder,  # noqa: E402
                                 frozen_pair_and_tlow, grounded_forward)
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha                      # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3b_runtime import (CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, LR,   # noqa: E402
                         SEED, json_ready, proposal_core)
from v41_runtime import ARM_A1                                          # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
B0 = '/root/data/experiments/v3b0_implicit_residual'
ROOT = '/root/data/experiments/v41_ground_then_transfer'
CACHE_META = os.path.join(SRC, 'cache_y0_lolbase', 'refiner_train', 'metadata.json')
CHECK_N = 8
ABS_TOL = 1e-6


def _load_b0_head(path, device):
    blob = torch.load(path, map_location='cpu')
    h = V3B0ResidualFusion(in_ch=96)
    h.load_state_dict(blob['model'], strict=True)
    h.to(device).eval()
    for p in h.parameters():
        p.requires_grad_(False)
    return h, blob, state_dict_sha(blob['model'])


def _check_step0(wrapper, b0_head, grounder, ds, indices, device, tag):
    core = proposal_core(wrapper)
    worst_y = 0.0
    worst_t = 0.0
    with torch.no_grad():
        for i in indices:
            t = sample_tensors(ds, int(i), 'correct', device)
            f0, fr, t_low, t_raw = frozen_pair_and_tlow(wrapper, t['Y0'], t['R'])
            y_b0 = t['Y0'] + b0_head(f0, t_raw, t['Y0'].shape[-2:], E=None)
            y_a1, dfr, fr_star, t_star = grounded_forward(
                core.match, b0_head, f0, fr, t_low, t['Y0'], grounder)
            dy = float((y_a1 - y_b0).abs().max())
            dt = float((t_star - t_raw).abs().max())
            df = float(dfr.abs().max())
            worst_y = max(worst_y, dy)
            worst_t = max(worst_t, dt)
            if dy > ABS_TOL or dt > ABS_TOL or df > 1e-7:
                raise SystemExit(
                    'HARD STOP step0 %s[%d] %s: dY=%.3e dT=%.3e dFR=%.3e'
                    % (tag, i, t['name'], dy, dt, df))
    return worst_y, worst_t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', ARM_A1 + '/checkpoints'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    b0 = json.load(open(os.path.join(a.b0_root, 'artifact_lock.json')))
    b0_psnr = json.load(open(os.path.join(
        a.b0_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    b0_ckpt = os.path.join(a.b0_root, 'checkpoints', 'ckpt_020000.pt')
    meta = json.load(open(CACHE_META))

    torch.manual_seed(SEED)
    gnd = RefGrounder()
    fr = torch.randn(1, 32, 8, 8)
    tl = torch.randn(1, 32, 8, 8)
    if float(gnd(fr, tl).abs().max()) > 1e-7:
        raise SystemExit('grounder step0 ΔFR not zero')
    n_g = count_params(gnd)
    sd = snapshot_(gnd)
    sha = state_dict_sha(sd)
    init_path = os.path.join(a.root, 'init', 'A1_grounder_s42.pt')
    torch.save(dict(model=sd, arm=ARM_A1, ground_direction=GROUND_DIR,
                    forward_direction=FORWARD_DIR), init_path)

    print('=== loading frozen B0@20k + proposal ===', flush=True)
    b0_head, b0_blob, b0_state_sha = _load_b0_head(b0_ckpt, a.device)
    if int(b0_blob.get('step', -1)) != 20000:
        raise SystemExit('B0 ckpt step != 20000')
    wrapper = load_proposal(b0['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = b0['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       b0['split_json'])
    mmap_tr = json.load(open(b0['mismatch_train']))
    mmap_dv = json.load(open(b0['mismatch_dev']))
    cache = os.path.join(a.src_root, b0.get('cache_name', 'cache_y0_lolbase'),
                         'refiner_train')
    ds_tr = make_dataset(ns, splits['train'], cache, mmap_tr)
    ds_dv = make_dataset(ns, splits['dev'], cache, mmap_dv)
    g = torch.Generator().manual_seed(SEED)
    idx_tr = torch.randperm(len(splits['train']), generator=g)[:CHECK_N].tolist()
    idx_dv = torch.randperm(len(splits['dev']), generator=g)[:CHECK_N].tolist()
    gnd = gnd.to(a.device).eval()
    print('=== step0 A1 vs frozen B0 train8 ===', flush=True)
    wtr = _check_step0(wrapper, b0_head, gnd, ds_tr, idx_tr, a.device, 'train')
    print('  worst dY=%.3e dT=%.3e' % wtr, flush=True)
    print('=== step0 A1 vs frozen B0 dev8 ===', flush=True)
    wdv = _check_step0(wrapper, b0_head, gnd, ds_dv, idx_dv, a.device, 'dev')
    print('  worst dY=%.3e dT=%.3e' % wdv, flush=True)

    lock = dict(
        stage='V4.1a',
        root=a.root,
        prior='V4_CASE_D_BROAD_HARM',
        b0_root=a.b0_root,
        repo_commit=git_head(),
        proposal_ckpt=b0['proposal_ckpt'],
        proposal_sha256=b0['proposal_sha256'],
        cache_name=b0.get('cache_name', 'cache_y0_lolbase'),
        base_cache_metadata_sha256=b0['base_cache_metadata_sha256'],
        cache_metadata_path=CACHE_META,
        cache_metadata_sha256=file_sha256(CACHE_META),
        base_ckpt=meta.get('base_checkpoint'),
        base_ckpt_sha256=file_sha256(meta['base_checkpoint']) if meta.get('base_checkpoint') else None,
        split_json=b0['split_json'],
        split_sha256=b0['split_sha256'],
        mismatch_train=b0['mismatch_train'],
        mismatch_train_sha256=b0['mismatch_train_sha256'],
        mismatch_dev=b0['mismatch_dev'],
        mismatch_dev_sha256=b0['mismatch_dev_sha256'],
        reference_variant=b0['reference_variant'],
        b0_head_ckpt=b0_ckpt,
        b0_head_ckpt_sha256=file_sha256(b0_ckpt),
        b0_head_state_sha=b0_state_sha,
        b0_head_init_sha=b0.get('init_sha'),
        ground_direction=GROUND_DIR,
        forward_direction=FORWARD_DIR,
        architecture='RefGrounder',
        grounder_zero_output=True,
        frozen_b0=True,
        trainable_modules=['RefGrounder'],
        n_params=n_g,
        grounder_init_path=init_path,
        grounder_init_sha=sha,
        optimizer='Adam',
        lr=LR,
        weight_decay=0.0,
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        seed=SEED,
        pair_schedule_seed=SEED,
        checkpoint_steps=list(CKPT_STEPS),
        official_test_allowed=False,
        frozen_b0_psnr=b0_psnr['B0_normal'],
        step0_check=dict(n=CHECK_N, train_worst_y=wtr[0], train_worst_t=wtr[1],
                         dev_worst_y=wdv[0], dev_worst_t=wdv[1]),
        note='A0=frozen B0@20k; A1=RefGrounder then frozen B0; sole trainable=Grounder',
    )
    dump_json(lock_path, json_ready(lock))
    print('V4.1a setup OK  grounder n=%d sha=%s' % (n_g, sha[:12]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
