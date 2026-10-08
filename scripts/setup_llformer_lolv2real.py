#!/usr/bin/env python
"""Lock splits, verify bridge hard gates, write artifact_lock.json. No training."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from llformer_runtime import (ROOT, DATA_DIR, SEED, build_train625_dev64,  # noqa: E402
                              dump_json, file_sha256, lock_architecture_fields,
                              set_seed, write_split_txt)
from model.LLFormerBridge import (OFFICIAL_LOL_CKPT, LLFormerBridge,  # noqa: E402
                                  build_official_llformer, load_into_llformer,
                                  reflect_pad_to_multiple)
from v3a6_runtime import git_head  # noqa: E402

ABS_TOL = 1e-6


def _equiv_check(device):
    off = build_official_llformer().to(device).eval()
    load_into_llformer(off, OFFICIAL_LOL_CKPT)
    br = LLFormerBridge().to(device).eval()
    info = load_into_llformer(br, OFFICIAL_LOL_CKPT)
    worst = 0.0
    with torch.no_grad():
        for h, w in ((128, 128), (400, 600), (401, 603)):
            x = torch.rand(1, 3, h, w, device=device)
            xp, h0, w0 = reflect_pad_to_multiple(x)
            ya = off(xp)[..., :h0, :w0]
            yb = br(xp)[..., :h0, :w0]
            ys = br.forward_staged(xp)[..., :h0, :w0]
            d1 = float((ya - yb).abs().max())
            d2 = float((yb - ys).abs().max())
            worst = max(worst, d1, d2)
            if d1 > ABS_TOL or d2 > ABS_TOL:
                raise SystemExit('HARD STOP equivalence d_off_br=%.3e d_staged=%.3e'
                                 % (d1, d2))
    return info, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('splits', 'checkpoints', 'diagnostics', 'logs', 'canonical'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    set_seed(SEED)
    train, dev = build_train625_dev64(data_dir=a.data_dir)
    train_path = os.path.join(a.root, 'splits', 'llformer_train625.txt')
    dev_path = os.path.join(a.root, 'splits', 'llformer_dev64.txt')
    write_split_txt(train_path, train)
    write_split_txt(dev_path, dev)

    print('=== bridge equivalence hard gate ===', flush=True)
    info, worst = _equiv_check(a.device)

    lock = lock_architecture_fields()
    lock.update(
        official_lol_checkpoint_sha256=file_sha256(OFFICIAL_LOL_CKPT),
        bridge_equivalence_max_abs=worst,
        train625_path=train_path,
        dev64_path=dev_path,
        train625_sha=file_sha256(train_path),
        dev64_sha=file_sha256(dev_path),
        n_train=len(train),
        n_dev=len(dev),
        ckpt_tensors_loaded=info['loaded'],
        setup_repo_commit=git_head(),
    )
    dump_json(lock_path, lock)
    print('wrote', lock_path, flush=True)
    print('train625=%d dev64=%d equiv_max_abs=%.3e' % (len(train), len(dev), worst))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
