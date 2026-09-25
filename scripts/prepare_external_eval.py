#!/usr/bin/env python
"""Build the strict external manifest, lock artifacts, and cache Y0 (plan §B/§D).

Read-only w.r.t. the model: the frozen N0 is only used to produce the base
output cache, exactly as in the source-domain V2.1 run.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.data1_localrefine import read_rgb, read_rgb_model_tensor   # noqa: E402
from local_refine_runtime import (infer_n0, load_frozen_n0, sha256,     # noqa: E402
                                  write_metadata)

MANIFEST_FIELDS = ['sample_id', 'camera', 'dataset', 'split', 'scene_id',
                   'group', 'low_path', 'high_path', 'nano_path',
                   'low_sha256', 'high_sha256', 'nano_sha256',
                   'low_h', 'low_w', 'nano_h', 'nano_w',
                   'reference_protocol_id', 'prior_use_status']


def _files(d):
    """Files only: the reference dirs contain a stray ``.ipynb_checkpoints``."""
    return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))


def _repo_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip()


def _repo_clean():
    out = subprocess.check_output(
        ['git', 'status', '--porcelain'],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode()
    return not out.strip()


def build_manifest(cfg):
    """low -> normal by prefix replacement; reference by identical basename."""
    tgt = cfg['target']
    low_dir, high_dir, ref_dir = tgt['low_dir'], tgt['high_dir'], tgt['reference_dir']
    refs = {f: os.path.join(ref_dir, f) for f in _files(ref_dir)}
    rows, missing = [], []
    for name in _files(low_dir):
        base, ext = os.path.splitext(name)
        if not base.startswith('low'):
            raise SystemExit('%s: unexpected low filename %s' % (low_dir, name))
        gt_name = base.replace('low', 'normal', 1) + ext
        gt = os.path.join(high_dir, gt_name)
        if not os.path.isfile(gt):
            raise SystemExit('no GT for %s (expected %s)' % (name, gt_name))
        if name not in refs:
            missing.append(name)
            continue
        sid = base
        rows.append(dict(
            sample_id=sid, camera=tgt['camera_group_value'],
            dataset=tgt['dataset'], split=tgt['split'], scene_id=sid,
            group=tgt['camera_group_value'],
            low_path=os.path.join(low_dir, name), high_path=gt,
            nano_path=refs[name],
            low_sha256=sha256(os.path.join(low_dir, name)),
            high_sha256=sha256(gt), nano_sha256=sha256(refs[name]),
            reference_protocol_id='lolv2real_nanobanana_ref_v3',
            prior_use_status=tgt['prior_use_status']))
    if missing:
        raise SystemExit('reference missing for %d sample(s): %s'
                         % (len(missing), missing[:5]))
    return rows


def validate(rows, cfg):
    if len(rows) != cfg['target']['expected_official_pair_count']:
        raise SystemExit('expected %d pairs, built %d'
                         % (cfg['target']['expected_official_pair_count'], len(rows)))
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise SystemExit('duplicate sample_id in manifest')
    if len({r['low_sha256'] for r in rows}) != len(rows):
        raise SystemExit('duplicate low image content in the target set')
    for r in rows:
        lo, hi = read_rgb(r['low_path']), read_rgb(r['high_path'])
        if lo.shape != hi.shape:
            raise SystemExit('%s: low/GT size mismatch %s vs %s'
                             % (r['sample_id'], lo.shape, hi.shape))
        r['low_h'], r['low_w'] = lo.shape[0], lo.shape[1]
        # normalised FIRST, then resized (V2.1 fixed order)
        n = read_rgb_model_tensor(r['nano_path'], size=(lo.shape[0], lo.shape[1]))
        r['nano_h'], r['nano_w'] = n.shape[-2], n.shape[-1]
        if not np.isfinite(n.numpy()).all():
            raise SystemExit('%s: non-finite reference' % r['sample_id'])
    resized = sum(1 for r in rows if (r['nano_h'], r['nano_w']) != (r['low_h'], r['low_w']))
    return resized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    cfg = json.load(open(a.config, encoding='utf-8'))
    out_root = cfg['output_root']
    tdir = os.path.join(out_root, 'target_lolv2real')
    os.makedirs(tdir, exist_ok=True)

    rows = build_manifest(cfg)
    resized = validate(rows, cfg)
    print('manifest: %d samples, reference resize needed: %d' % (len(rows), resized))

    man_path = os.path.join(tdir, 'external_manifest.csv')
    with open(man_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in MANIFEST_FIELDS})

    src = cfg['source']
    lock = dict(
        experiment=cfg['experiment_name'], source_protocol=src['required_protocol'],
        base_checkpoint=src['base_checkpoint'], base_checkpoint_sha256=sha256(src['base_checkpoint']),
        self_checkpoint=src['self_checkpoint'], self_checkpoint_sha256=sha256(src['self_checkpoint']),
        nano_checkpoint=src['nano_checkpoint'], nano_checkpoint_sha256=sha256(src['nano_checkpoint']),
        shared_init=src['shared_init'], shared_init_sha256=sha256(src['shared_init']),
        repo_commit=_repo_commit(), repo_clean_at_lock=_repo_clean(),
        tile=dict(size=cfg['inference']['base_tile_size'],
                  overlap=cfg['inference']['base_tile_overlap'],
                  window=cfg['inference']['base_tile_window']),
        manifest_sha256=sha256(man_path), sample_count=len(rows),
        reference_resize_count=resized,
        reference_protocol=cfg['reference_generation'],
        prior_use_status=cfg['target']['prior_use_status'],
        _expected=dict(base_checkpoint_sha256=src.get('base_checkpoint_sha256'),
                       self_checkpoint_sha256=src.get('self_checkpoint_sha256'),
                       nano_checkpoint_sha256=src.get('nano_checkpoint_sha256')))
    for k in ('base_checkpoint_sha256', 'self_checkpoint_sha256', 'nano_checkpoint_sha256'):
        exp = lock['_expected'].get(k)
        if exp and exp != lock[k]:
            raise SystemExit('artifact drift on %s: config %s, actual %s' % (k, exp, lock[k]))
    lock.pop('_expected')
    json.dump(lock, open(os.path.join(out_root, 'artifact_lock.json'), 'w',
                         encoding='utf-8'), indent=2, sort_keys=True)

    json.dump(dict(
        generator_model='gemini-3.1-flash-image',
        generator_script='/root/data/nanobanana_enhance.py (via /root/v2ray/run_nanobanana.sh)',
        prompt=None,
        prompt_note='not logged for the LOL v3 batch (the "提示词来源" print was added 09-15, after '
                    'this batch) and no log exists at all for the data1 batch; the generator model '
                    'string is the only verifiable part of the protocol',
        resize_to_input=True, input_source='Test/Low only',
        generation_log='/root/data/.nanobanana_full_20260911_004018.log',
        selected_variant='nanobanana_ref_v3',
        variant_note='v1/v2/v3 all complete (100/100); v3 is the project-wide default; using v3 with a '
                     'refiner trained on the data1 batch is a data+protocol composite transfer, not a '
                     'clean single-variable external test',
        refused_ids_in_scope=0),
        open(os.path.join(out_root, 'reference_protocol.json'), 'w',
             encoding='utf-8'), indent=2, sort_keys=True)

    base_run_dir = os.path.dirname(os.path.dirname(src['base_checkpoint']))
    model, tr, _cfg = load_frozen_n0(src['base_checkpoint'], base_run_dir, a.device)
    cdir = os.path.join(tdir, 'cache_n0_eval')
    os.makedirs(cdir, exist_ok=True)
    for i, r in enumerate(rows):
        low = read_rgb_model_tensor(r['low_path'])[None].to(a.device)
        y0 = infer_n0(model, tr, low)
        np.save(os.path.join(cdir, r['sample_id'] + '.npy'),
                y0[0].float().cpu().numpy())
        if (i + 1) % 25 == 0:
            print('  cached %d/%d' % (i + 1, len(rows)))
    write_metadata(cdir, dict(
        split='external', dataset=cfg['target']['dataset'], n=len(rows),
        sample_count=len(rows), sample_ids=[r['sample_id'] for r in rows],
        base_checkpoint_sha256=lock['base_checkpoint_sha256'],
        manifest_sha256=lock['manifest_sha256'],
        preprocess_version='v21:decode->normalise(div127.5-1)->resize(bilinear,ac=False)->geometry',
        repo_commit=lock['repo_commit'], tile_size=lock['tile']['size'],
        tile_overlap=lock['tile']['overlap'], tile_window=lock['tile']['window'],
        dtype='float32', layout='CHW', clamp=False, round=False, complete=True))
    print('done -> %s' % tdir)


if __name__ == '__main__':
    main()
