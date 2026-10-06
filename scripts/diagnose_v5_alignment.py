#!/usr/bin/env python
"""V5.0 alignment diagnostics on a checkpoint (dev, normal Ref)."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import load_frozen_n0                         # noqa: E402
from model.V5Model import V5Model                                       # noqa: E402
from model.V5RetinexBridge import INJECTION_POINT, tiled_v5_forward     # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors       # noqa: E402
from v3a5_runtime import STATES                                         # noqa: E402
from v3a6_runtime import dump_json, git_head, nanmean                   # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v5_runtime import ARM_A1, flow_stats, require_ckpt_blob_v5, residual_offset_stats  # noqa: E402

ROOT = '/root/data/experiments/v5_aligned_ref'
SRC = '/root/data/experiments/v3a1_lolv2real'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=8)
    a = ap.parse_args(_CLI)
    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official test forbidden')
    path = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    blob = torch.load(path, map_location='cpu')
    require_ckpt_blob_v5(
        blob, arm=ARM_A1, step=a.step, init_sha=lock['v5_init_sha'],
        repo_commit=lock['repo_commit'], injection_point=INJECTION_POINT)
    model = V5Model().to(a.device).eval()
    model.load_state_dict(blob['model'], strict=True)
    n0, _tr, _cfg = load_frozen_n0(lock['base_ckpt'], lock['base_run_dir'], a.device)
    mainnet = n0.MainNet.eval()
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    mmap = json.load(open(lock['mismatch_dev']))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.src_root, lock.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap)
    n = min(a.limit, len(splits['dev']))
    rows = []
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                _y, aux = tiled_v5_forward(
                    mainnet, model, t['X'], t['Y0'], t['R'], collect_aux=True)
                rec = dict(name=t['name'], state=state)
                rec.update(flow_stats(aux['match_flow_h4']))
                rec.update(residual_offset_stats(aux['residual_offset']))
                rec['sim_max'] = float(aux['sim_max'].mean())
                rec['margin'] = float(aux['margin'].mean())
                rec['entropy'] = float(aux['entropy'].mean())
                rows.append(rec)
    def by_state(key):
        out = {s: nanmean([r[key] for r in rows if r['state'] == s]) for s in STATES}
        out['mean'] = nanmean([r[key] for r in rows])
        return out
    summary = {k: by_state(k) for k in rows[0] if k not in ('name', 'state')}
    out = os.path.join(a.root, 'diagnostics', 'align_%06d.json' % a.step)
    dump_json(out, json_ready(dict(n=n, commit=git_head(), stats=summary)))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
