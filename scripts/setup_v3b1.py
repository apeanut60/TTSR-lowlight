#!/usr/bin/env python
"""V3-B.1 setup: B1a/B1b inits sharing B0 residual zero-init. No training."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3BReferenceAdapt import ARMS, V3B1Model                      # noqa: E402
from model.V3BResidualFusion import count_params                        # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha                      # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3b_runtime import DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED, json_ready  # noqa: E402

B0 = '/root/data/experiments/v3b0_implicit_residual'
ROOT = '/root/data/experiments/v3b1_reference_adapt'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    for arm in ARMS:
        os.makedirs(os.path.join(a.root, arm, 'checkpoints'), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    b0 = json.load(open(os.path.join(a.b0_root, 'artifact_lock.json')))
    b0_init = torch.load(b0['init_path'], map_location='cpu')
    b0_head = b0_init['model']
    if state_dict_sha(b0_head) != b0['init_sha']:
        raise SystemExit('B0 init sha mismatch')

    torch.manual_seed(SEED)
    inits, n_params, extra = {}, {}, {}
    f0 = torch.randn(1, 32, 8, 8)
    t = torch.randn(1, 32, 8, 8)
    y0 = torch.zeros(1, 3, 16, 16)
    for arm in ARMS:
        m = V3B1Model(arm)
        m.head.load_state_dict(b0_head, strict=True)
        delta, aux = m(f0, t, y0.shape[-2:], return_aux=True)
        if float(delta.abs().max()) > 1e-7:
            raise SystemExit('%s step0 ΔY not 0' % arm)
        if arm == 'B1a_identity_adapt':
            if float((aux['t_adapt'] - t).abs().max()) > 1e-6:
                raise SystemExit('B1a T_adapt != T at init')
        else:
            if float((aux['t_adapt'] - t).abs().max()) < 1e-3:
                raise SystemExit('B1b T_adapt unexpectedly == T')
        path = os.path.join(a.root, 'init', '%s_s42.pt' % arm)
        sd = snapshot_(m)
        torch.save(dict(model=sd, arm=arm), path)
        inits[arm] = dict(path=path, sha=state_dict_sha(sd))
        n_params[arm] = count_params(m)
        extra[arm] = n_params[arm] - int(b0['n_params'])

    b0_20k = json.load(open(os.path.join(
        a.b0_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    b0_safe = json.load(open(os.path.join(
        a.b0_root, 'diagnostics', 'eval_020000', 'safety.json')))

    lock = dict(
        stage='V3-B.1',
        root=a.root,
        prior='V3B0_CASE_A_STRONG_GO',
        b0_root=a.b0_root,
        repo_commit=git_head(),
        proposal_ckpt=b0['proposal_ckpt'],
        proposal_sha256=b0['proposal_sha256'],
        cache_name=b0.get('cache_name', 'cache_y0_lolbase'),
        base_cache_metadata_sha256=b0['base_cache_metadata_sha256'],
        split_json=b0['split_json'],
        split_sha256=b0['split_sha256'],
        mismatch_train=b0['mismatch_train'],
        mismatch_train_sha256=b0['mismatch_train_sha256'],
        mismatch_dev=b0['mismatch_dev'],
        mismatch_dev_sha256=b0['mismatch_dev_sha256'],
        reference_variant=b0['reference_variant'],
        architecture='V3B1Model',
        arm='B1a_identity_adapt',
        b1_arms=list(ARMS),
        n_params=n_params,
        extra_params=extra,
        init_sha=inits['B1a_identity_adapt']['sha'],
        b0_init_sha=b0['init_sha'],
        init_paths={k: v['path'] for k, v in inits.items()},
        init_shas={k: v['sha'] for k, v in inits.items()},
        optimizer='Adam',
        lr=LR,
        weight_decay=0.0,
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        seed=SEED,
        pair_schedule_seed=SEED,
        official_test_allowed=False,
        frozen_b0_psnr=b0_20k['B0_normal'],
        frozen_b0_safety=b0_safe,
        note='sole variable=T_adapt; residual arch/init/loss/schedule match B0',
    )
    dump_json(lock_path, json_ready(lock))
    print('V3-B.1 setup OK')
    for arm in ARMS:
        print('  %s  n=%d extra=%d sha=%s' % (
            arm, n_params[arm], extra[arm], inits[arm]['sha'][:16]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
