#!/usr/bin/env python
"""V3-B.4.1 setup: A0 RGB + A1 base-conditioned H/2; HARD Base==bridge."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import infer_n0, load_frozen_n0, metrics as _metrics  # noqa: E402
from model.V3BBaseConditionedFeatureResidual import (  # noqa: E402
    IN_CH, BaseConditionedFeatureResidual)
from model.V3BFeatureBridge import INJECTION_POINT, tiled_bridge_decode  # noqa: E402
from model.V3BFeatureResidual import FeatureResidualAdapter             # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion, count_params    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors       # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha                      # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3b_runtime import CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED, json_ready  # noqa: E402
from v3b41_runtime import (ARM_A0, ARM_A1, ARMS, BASE_CONDITIONED,      # noqa: E402
                           BASE_FEATURE_CH, REF_INPUT_CH)

SRC = '/root/data/experiments/v3a1_lolv2real'
B0 = '/root/data/experiments/v3b0_implicit_residual'
B4 = '/root/data/experiments/v3b4_feature_residual'
ROOT = '/root/data/experiments/v3b41_base_conditioned_h2'
CACHE_META = os.path.join(SRC, 'cache_y0_lolbase', 'refiner_train', 'metadata.json')
BRIDGE_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'model', 'V3BFeatureBridge.py')
BRIDGE_N = 16
ABS_TOL = 1e-6
PSNR_TOL = 1e-6
MAX_A1_PARAMS = 200000


def _check_bridge(n0, trainer, ds, indices, device, tag):
    main = n0.MainNet
    worst_abs = 0.0
    worst_psnr = 0.0
    with torch.no_grad():
        for i in indices:
            t = sample_tensors(ds, int(i), 'correct', device)
            y_base = infer_n0(n0, trainer, t['X'])
            y_br = tiled_bridge_decode(main, t['X'], delta_fn=None)
            d = float((y_base - y_br).abs().max())
            p0 = float(_metrics(y_base, t['H'])[0])
            p1 = float(_metrics(y_br, t['H'])[0])
            pd = abs(float(p0) - float(p1))
            worst_abs = max(worst_abs, d)
            worst_psnr = max(worst_psnr, pd)
            if d > ABS_TOL or pd > PSNR_TOL:
                raise SystemExit(
                    'HARD STOP bridge vs Base on %s[%d] %s: max_abs=%.3e psnr_diff=%.3e'
                    % (tag, i, t['name'], d, pd))
    return worst_abs, worst_psnr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--b4_root', default=B4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', ARM_A0 + '/checkpoints',
              ARM_A1 + '/checkpoints'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    b0 = json.load(open(os.path.join(a.b0_root, 'artifact_lock.json')))
    b0_psnr = json.load(open(os.path.join(
        a.b0_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    b4_psnr = json.load(open(os.path.join(
        a.b4_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    meta = json.load(open(CACHE_META))
    base_ckpt = meta['base_checkpoint']
    base_run = os.path.dirname(os.path.dirname(base_ckpt))
    bridge_sha = file_sha256(BRIDGE_PY)

    torch.manual_seed(SEED)
    a0 = V3B0ResidualFusion(in_ch=96)
    a1 = BaseConditionedFeatureResidual()
    blind = FeatureResidualAdapter()
    f_dec = torch.randn(1, 80, 8, 8)
    f0 = torch.randn(1, 32, 8, 8)
    tt = torch.randn(1, 32, 8, 8)
    if float(a0(f0, tt, (16, 16)).abs().max()) > 1e-7:
        raise SystemExit('A0 step0 ΔY not zero')
    if float(a1(f_dec, f0, tt).abs().max()) > 1e-7:
        raise SystemExit('A1 step0 ΔF not zero')
    if int(a1.in_ch) != IN_CH:
        raise SystemExit('A1 in_ch %d != %d' % (a1.in_ch, IN_CH))
    n_a0 = count_params(a0)
    n_a1 = count_params(a1)
    n_blind = count_params(blind)
    if n_a1 > MAX_A1_PARAMS:
        raise SystemExit('adapter params %d > %d' % (n_a1, MAX_A1_PARAMS))

    a0_path = os.path.join(a.root, 'init', 'A0_b0_replay_s42.pt')
    a1_path = os.path.join(a.root, 'init', 'A1_base_cond_h2_s42.pt')
    sd0, sd1 = snapshot_(a0), snapshot_(a1)
    torch.save(dict(model=sd0, arm=ARM_A0, in_ch=96), a0_path)
    torch.save(dict(
        model=sd1, arm=ARM_A1, injection_point=INJECTION_POINT,
        base_conditioned=True, base_feature_ch=BASE_FEATURE_CH,
        ref_input_ch=REF_INPUT_CH, in_ch=IN_CH), a1_path)

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

    print('=== loading frozen N0 for bridge HARD check ===', flush=True)
    n0, trainer, _cfg = load_frozen_n0(base_ckpt, base_run, a.device)
    g = torch.Generator().manual_seed(SEED)
    idx_tr = torch.randperm(len(splits['train']), generator=g)[:BRIDGE_N].tolist()
    idx_dv = torch.randperm(len(splits['dev']), generator=g)[:BRIDGE_N].tolist()
    print('=== bridge vs Base train16 ===', flush=True)
    wtr = _check_bridge(n0, trainer, ds_tr, idx_tr, a.device, 'train')
    print('  worst max_abs=%.3e psnr_diff=%.3e' % wtr, flush=True)
    print('=== bridge vs Base dev16 ===', flush=True)
    wdv = _check_bridge(n0, trainer, ds_dv, idx_dv, a.device, 'dev')
    print('  worst max_abs=%.3e psnr_diff=%.3e' % wdv, flush=True)

    lock = dict(
        stage='V3-B.4.1',
        root=a.root,
        prior='V3B4_CASE_D_HARM',
        b0_root=a.b0_root,
        b4_root=a.b4_root,
        repo_commit=git_head(),
        proposal_ckpt=b0['proposal_ckpt'],
        proposal_sha256=b0['proposal_sha256'],
        cache_name=b0.get('cache_name', 'cache_y0_lolbase'),
        base_cache_metadata_sha256=b0['base_cache_metadata_sha256'],
        cache_metadata_path=CACHE_META,
        cache_metadata_sha256=file_sha256(CACHE_META),
        base_ckpt=base_ckpt,
        base_ckpt_sha256=file_sha256(base_ckpt),
        base_run_dir=base_run,
        split_json=b0['split_json'],
        split_sha256=b0['split_sha256'],
        mismatch_train=b0['mismatch_train'],
        mismatch_train_sha256=b0['mismatch_train_sha256'],
        mismatch_dev=b0['mismatch_dev'],
        mismatch_dev_sha256=b0['mismatch_dev_sha256'],
        reference_variant=b0['reference_variant'],
        injection_point=INJECTION_POINT,
        bridge_file=BRIDGE_PY,
        bridge_file_sha=bridge_sha,
        base_conditioned=BASE_CONDITIONED,
        base_feature_ch=BASE_FEATURE_CH,
        ref_input_ch=REF_INPUT_CH,
        reference_feature_ch=32,
        architecture_a0='V3B0ResidualFusion',
        architecture_a1='BaseConditionedFeatureResidual',
        arms=list(ARMS),
        n_params={ARM_A0: n_a0, ARM_A1: n_a1, 'B4_blind_h2': n_blind},
        adapter_n_params=n_a1,
        param_ratio_a1_a0=float(n_a1) / float(n_a0),
        init_paths={ARM_A0: a0_path, ARM_A1: a1_path},
        a0_init_sha=state_dict_sha(sd0),
        a1_init_sha=state_dict_sha(sd1),
        init_shas={ARM_A0: state_dict_sha(sd0), ARM_A1: state_dict_sha(sd1)},
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
        frozen_blind_h2_psnr=b4_psnr['A1_h2_feature'],
        bridge_check=dict(n=BRIDGE_N, train_worst_abs=wtr[0], train_worst_psnr=wtr[1],
                          dev_worst_abs=wdv[0], dev_worst_psnr=wdv[1]),
        note='A0=B0 RGB; A1=base-conditioned H/2 ΔF; sole variable=F_dec conditioning',
    )
    dump_json(lock_path, json_ready(lock))
    print('V3-B.4.1 setup OK  A0 n=%d A1 n=%d blind=%d ratio=%.3f' % (
        n_a0, n_a1, n_blind, n_a1 / float(n_a0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
