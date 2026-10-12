#!/usr/bin/env python
"""Quick alignment dump for a V6 ckpt (normal Ref, few images)."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V6LLFormerRefBridge import load_frozen_llformer_base, tiled_v6_forward  # noqa: E402
from model.V6Model import V6Model  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a6_runtime import dump_json  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v5_runtime import flow_stats, r_delta  # noqa: E402
from v6_runtime import ARM_A1, ROOT, SRC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, default=30000)
    ap.add_argument('--limit', type=int, default=8)
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    path = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    blob = torch.load(path, map_location='cpu')
    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)
    model = V6Model().to(a.device).eval()
    model.load_state_dict(blob['model'], strict=True)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    mmap = json.load(open(lock['mismatch_dev']))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.root, lock['cache_name'], 'refiner_train'), mmap)

    rows = []
    with torch.no_grad():
        for i in range(min(a.limit, len(splits['dev']))):
            t = sample_tensors(ds, i, 'correct', a.device)
            y, aux = tiled_v6_forward(
                base, model, t['X'], t['Y0'], t['R'], collect_aux=True)
            rec = dict(name=t['name'], **flow_stats(aux['match_flow_h4']))
            r2, _ = r_delta(aux['delta_d2'], aux['d2'])
            r1, _ = r_delta(aux['h2_delta_d1'], aux['d1'])
            rec['r_D2'] = r2
            rec['r_D1'] = r1
            rows.append(rec)
            print(t['name'], 'disp=%.1f rD2=%.3f' % (rec['disp_mean'], r2),
                  flush=True)
    out = os.path.join(a.root, 'diagnostics', 'align_%06d' % a.step)
    os.makedirs(out, exist_ok=True)
    dump_json(os.path.join(out, 'rows.json'), json_ready(rows))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
