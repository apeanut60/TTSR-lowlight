#!/usr/bin/env python3
"""Read-only audit of the three short-finetune stages.

Does not train, does not write into experiment directories, and does not
import a training entry point's main().
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import statistics
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EXP = '/root/data/experiments/TTSR-lowlight-block-amend'
OUT = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ('lolv2_real', 'csfi_lfca_v1_lolv2_partial_20261119',
     '/root/data/datasets/lol-v2-real', 'nanobanana_ref_v3'),
    ('data1_huawei', 'csfi_lfca_v1_data1_huawei_partial_20261119',
     os.path.join(EXP, 'views/data1_huawei'), 'nanobanana_ref'),
    ('data1_nikon', 'csfi_lfca_v1_data1_nikon_partial_20261119',
     os.path.join(EXP, 'views/data1_nikon'), 'nanobanana_ref'),
]
STAGES = ('short_ft_5ep', 'short_ft_r2', 'short_ft_w91')
EXPECTED_N = {
    'lolv2_real': {'internal_val': 64, 'official_test': 100},
    'data1_huawei': {'internal_val': 50, 'official_test': 30},
    'data1_nikon': {'internal_val': 50, 'official_test': 20},
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_state(state) -> str:
    import torch
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        if not torch.is_tensor(value):
            h.update(key.encode())
            h.update(repr(value).encode())
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json(name: str, obj) -> None:
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=2)


def quantile(values, q):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def image_hw(path: str):
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return int(h), int(w)


def luminance_means(path: str):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGB'), dtype=np.float64)
    y = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return {
        'y_mean': float(y.mean() / 255.0),
        'y_p10': float(np.quantile(y, 0.10) / 255.0),
        'y_p50': float(np.quantile(y, 0.50) / 255.0),
        'y_p90': float(np.quantile(y, 0.90) / 255.0),
        'rgb_mean': [float(arr[:, :, c].mean() / 255.0) for c in range(3)],
        'h': int(arr.shape[0]),
        'w': int(arr.shape[1]),
    }


def replay_geometry(h, w, seed, epoch, sample_id, crop=160):
    """Mirror dataset._paired_pad_crop / _paired_flip_rot decision sequence."""
    from dataset.block_amend_lolv2 import stable_aug_seed
    rng = random.Random(stable_aug_seed(seed, epoch, sample_id))
    ph = max(0, crop - h)
    pw = max(0, crop - w)
    hh = h + ph
    ww = w + pw
    iy = rng.randrange(0, hh - crop + 1)
    ix = rng.randrange(0, ww - crop + 1)
    rot = rng.randint(0, 3)
    flip_lr = rng.randint(0, 1)
    flip_ud = rng.randint(0, 1)
    return {
        'iy': iy, 'ix': ix, 'rot90': rot,
        'flip_lr': flip_lr, 'flip_ud': flip_ud,
        'padded_h': ph, 'padded_w': pw,
    }


def replay_order(n, seed, epochs, batch_size):
    import torch
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    out = {}
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen).tolist()
        batches = [perm[i:i + batch_size] for i in range(0, len(perm) - batch_size + 1, batch_size)]
        used = [i for batch in batches for i in batch]
        dropped = [i for i in perm if i not in set(used)]
        # Preserve perm order of dropped tail: drop_last drops the tail of the permutation.
        dropped = perm[len(used):]
        out[epoch] = {'perm': perm, 'batches': batches, 'dropped': dropped}
    # A second generator must reproduce epoch 1.
    gen2 = torch.Generator()
    gen2.manual_seed(int(seed))
    again = torch.randperm(n, generator=gen2).tolist()
    out['_epoch1_repeat_ok'] = again == out[1]['perm']
    return out


def adam_steps(opt_state):
    steps = []
    for slot in opt_state.get('state', {}).values():
        step = slot.get('step')
        if step is None:
            continue
        if hasattr(step, 'item'):
            steps.append(int(step.item()))
        else:
            steps.append(int(step))
    return sorted(set(steps))


def check_eval_doc(path, expected_n, role):
    doc = load_json(path)
    rows = doc.get('per_image') or []
    ids = [str(r.get('id')) for r in rows]
    problems = []
    if len(rows) != expected_n:
        problems.append('n_rows=%d expected=%d' % (len(rows), expected_n))
    if len(ids) != len(set(ids)):
        problems.append('duplicate_ids')
    if any(r.get('protocol') not in (None, 'rgb_float_clipped_v1') for r in rows):
        problems.append('row_protocol')
    if doc.get('protocol') != 'rgb_float_clipped_v1':
        problems.append('doc_protocol=%s' % doc.get('protocol'))
    finite = True
    psnrs, ssims = [], []
    for r in rows:
        for key in ('psnr_rgb', 'ssim_rgb', 'frac_lt0', 'frac_gt1', 'raw_min', 'raw_max'):
            value = r.get(key)
            if not isinstance(value, (int, float)) or value != value or abs(value) == float('inf'):
                finite = False
        psnrs.append(float(r['psnr_rgb']))
        ssims.append(float(r['ssim_rgb']))
    if not finite:
        problems.append('non_finite')
    mean_p = sum(psnrs) / len(psnrs) if psnrs else None
    mean_s = sum(ssims) / len(ssims) if ssims else None
    if mean_p is not None and abs(mean_p - float(doc['psnr_rgb_mean'])) > 1e-9:
        problems.append('psnr_mean_mismatch')
    if mean_s is not None and abs(mean_s - float(doc['ssim_rgb_mean'])) > 1e-9:
        problems.append('ssim_mean_mismatch')
    return {
        'role': role,
        'path': path,
        'n': len(rows),
        'ids': ids,
        'problems': problems,
        'checkpoint_path': doc.get('checkpoint_path'),
        'checkpoint_epoch': doc.get('checkpoint_epoch'),
        'model_variant': doc.get('model_variant'),
        'metric_protocol': doc.get('metric_protocol'),
        'has_historical_checkpoint_hash': 'checkpoint_sha256' in doc,
        'psnr_rgb_mean': doc.get('psnr_rgb_mean'),
        'ssim_rgb_mean': doc.get('ssim_rgb_mean'),
        'by_id': {str(r['id']): r for r in rows},
    }


def inventory_and_evals():
    import torch
    from block_amend_ckpt import load_local_checkpoint
    rows = []
    fingerprints = {}
    eval_cache = {}
    for label, run, _dataset, _test_ref in RUNS:
        run_dir = os.path.join(EXP, run)
        targets = [('base_internal_best', os.path.join(run_dir, 'checkpoint_best_val_psnr.pt'), None)]
        for stage in STAGES:
            for epoch in range(1, 6):
                targets.append((
                    stage,
                    os.path.join(run_dir, stage, 'checkpoint_epoch_%04d.pt' % epoch),
                    epoch,
                ))
        for stage_name, path, epoch_hint in targets:
            if not os.path.isfile(path):
                rows.append({'dataset': label, 'stage': stage_name, 'path': path, 'missing': True})
                continue
            file_sha = sha256_file(path)
            pack = load_local_checkpoint(path)
            cfg = pack.get('config') or {}
            opt = pack.get('optimizer_state_dict') or {}
            groups = opt.get('param_groups') or [{}]
            sched = pack.get('scheduler_state_dict') or {}
            fp = fingerprint_state(pack['model_state_dict'])
            fingerprints[path] = fp
            steps = adam_steps(opt)
            rec = {
                'dataset': label,
                'run': run,
                'stage': stage_name,
                'local_epoch_filename': epoch_hint,
                'pack_epoch': pack.get('epoch'),
                'global_step': pack.get('global_step'),
                'path': path,
                'file_sha256': file_sha,
                'model_state_sha256': fp,
                'model_variant': pack.get('model_variant'),
                'schema': pack.get('checkpoint_schema_version'),
                'init_scope': pack.get('init_scope') or cfg.get('init_scope'),
                'init_checkpoint': pack.get('init_checkpoint') or cfg.get('init_checkpoint'),
                'rec_w': cfg.get('rec_w'),
                'ms_ssim_w': cfg.get('ms_ssim_w'),
                'rec_loss_type': cfg.get('rec_loss_type'),
                'optimizer': 'Adam',
                'lr_config': cfg.get('lr_rate'),
                'lr_param_group': groups[0].get('lr') if groups else None,
                'gamma': cfg.get('gamma'),
                'decay': cfg.get('decay'),
                'scheduler_last_epoch': sched.get('last_epoch'),
                'adam_steps': steps,
                'seed': cfg.get('seed'),
                'crop': cfg.get('train_crop_size'),
                'batch_size': cfg.get('batch_size'),
                'num_workers': cfg.get('num_workers'),
                'drop_last_code': True,
                'train_hash': (pack.get('split_identity') or {}).get('train_hash'),
                'val_hash': (pack.get('split_identity') or {}).get('val_hash'),
                'ref_subdir_identity': (pack.get('split_identity') or {}).get('ref_subdir'),
                'train_ref_subdir': cfg.get('train_ref_subdir'),
                'test_ref_subdir': cfg.get('test_ref_subdir'),
                'dataset_dir': cfg.get('dataset_dir'),
                'metric_protocol': pack.get('metric_protocol') or cfg.get('metric_protocol'),
                'best_val_psnr_in_pack': pack.get('best_val_psnr'),
                'source': 'checkpoint pack via load_local_checkpoint; file sha computed now',
            }
            if stage_name != 'base_internal_best':
                val_path = os.path.join(
                    run_dir, stage_name, 'val_metrics_ep%05d.json' % epoch_hint)
                test_path = os.path.join(
                    run_dir, 'official_test_%s' % stage_name, 'epoch_%04d' % epoch_hint,
                    'eval_official_test.json')
                # official dir names: official_test_short_ft, official_test_short_ft_r2, official_test_short_ft_w91
                eval_rel = {
                    'short_ft_5ep': 'official_test_short_ft',
                    'short_ft_r2': 'official_test_short_ft_r2',
                    'short_ft_w91': 'official_test_short_ft_w91',
                }[stage_name]
                test_path = os.path.join(
                    run_dir, eval_rel, 'epoch_%04d' % epoch_hint, 'eval_official_test.json')
                image_dir = os.path.join(
                    run_dir, eval_rel, 'epoch_%04d' % epoch_hint, 'images_official_test')
                rec['val_json'] = val_path if os.path.isfile(val_path) else None
                rec['test_json'] = test_path if os.path.isfile(test_path) else None
                rec['test_image_dir'] = image_dir if os.path.isdir(image_dir) else None
                for role, epath, key in (
                    ('internal_val', val_path, 'val_check'),
                    ('official_test', test_path, 'test_check'),
                ):
                    if not os.path.isfile(epath):
                        rec[key] = {'missing': True, 'path': epath}
                        continue
                    checked = check_eval_doc(epath, EXPECTED_N[label][role], role)
                    eval_cache[(label, stage_name, epoch_hint, role)] = checked
                    path_ok = True
                    if role == 'official_test':
                        path_ok = os.path.abspath(checked.get('checkpoint_path') or '') == os.path.abspath(path)
                        epoch_ok = checked.get('checkpoint_epoch') == epoch_hint
                    else:
                        # internal val json is the train_loop summary; it stores split identity, not checkpoint path
                        epoch_ok = None
                        path_ok = None
                    checked_public = {k: v for k, v in checked.items() if k != 'by_id' and k != 'ids'}
                    checked_public['ids_n'] = len(checked['ids'])
                    checked_public['ids_unique'] = len(set(checked['ids']))
                    checked_public['path_matches_this_file'] = path_ok
                    checked_public['epoch_matches_filename'] = epoch_ok
                    checked_public['source_guarantee'] = (
                        'JSON has no checkpoint hash. Path/epoch checked against the file that exists now.'
                        if role == 'official_test' else
                        'Internal-val JSON is written by the training loop and has no checkpoint hash.'
                    )
                    rec[key] = checked_public
            rows.append(rec)
            del pack
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, fingerprints, eval_cache


def parent_checks(rows):
    by_path = {r['path']: r for r in rows if not r.get('missing')}
    checks = []
    for r in rows:
        if r.get('missing') or r['stage'] == 'base_internal_best':
            continue
        parent = r.get('init_checkpoint')
        parent_row = by_path.get(os.path.abspath(parent)) if parent else None
        # paths in the pack are absolute already
        parent_row = by_path.get(parent) or parent_row
        item = {
            'dataset': r['dataset'],
            'stage': r['stage'],
            'epoch': r['local_epoch_filename'],
            'init_checkpoint': parent,
            'parent_in_inventory': parent_row is not None,
            'same_dataset_dir_token': (r['run'] in (parent or '')),
            'rec_w': r['rec_w'],
            'ms_ssim_w': r['ms_ssim_w'],
            'lr_param_group': r['lr_param_group'],
            'gamma': r['gamma'],
            'pack_epoch_matches_name': r['pack_epoch'] == r['local_epoch_filename'],
            'model_variant': r['model_variant'],
            'schema': r['schema'],
        }
        if parent and os.path.isfile(parent):
            parent_sha = sha256_file(parent) if parent not in by_path else by_path[parent]['file_sha256']
            if parent in by_path:
                parent_sha = by_path[parent]['file_sha256']
            item['parent_file_sha256'] = parent_sha
            item['parent_is_expected_file'] = None
        if parent_row:
            item['parent_model_sha'] = parent_row['model_state_sha256']
            item['parent_stage'] = parent_row['stage']
            item['parent_epoch'] = parent_row.get('local_epoch_filename') or parent_row.get('pack_epoch')
        checks.append(item)
    # uniqueness of model fingerprints within a stage
    groups = defaultdict(list)
    for r in rows:
        if r.get('missing') or r['stage'] == 'base_internal_best':
            continue
        groups[(r['dataset'], r['stage'])].append(r['model_state_sha256'])
    unique = {
        '%s/%s' % k: {'n': len(v), 'unique_model_fingerprints': len(set(v))}
        for k, v in groups.items()
    }
    return checks, unique


def augmentation(rows):
    from dataset.block_amend_lolv2 import load_manifest_items
    summary = {
        'status': (
            'Geometric augmentation is a pure function of (seed, local epoch, sample id, H, W). '
            'dataset/block_amend_lolv2.py is unchanged between amend-4 (2026-11-19, before these runs) '
            'and HEAD. train_loop calls set_epoch with the same integer it logs as Epoch. '
            'Each short-ft log starts at Epoch 1. Shuffle uses a fresh torch.Generator seeded with config seed '
            'inside prepare_data, which is unchanged since amend-4. This replay is the sequence that code produces; '
            'H/W are read from the files now, and no per-step trace was stored on 2026-11-20.'
        ),
        'datasets': {},
    }
    trace_path = os.path.join(OUT, 'augmentation_trace.jsonl')
    brightness = []
    with open(trace_path, 'w') as trace:
        for label, run, _dataset, _test_ref in RUNS:
            stage_rows = [
                r for r in rows
                if r.get('dataset') == label and r.get('stage') == 'short_ft_5ep' and not r.get('missing')
            ]
            cfg_seed = stage_rows[0]['seed'] if stage_rows else None
            batch_size = int(stage_rows[0]['batch_size']) if stage_rows else 4
            crop = int(stage_rows[0]['crop']) if stage_rows else 160
            seeds = sorted({r.get('seed') for r in rows if r.get('dataset') == label and not r.get('missing')})
            split_dir = os.path.join(EXP, run, 'splits')
            items = load_manifest_items(os.path.join(split_dir, 'split_train.json'))
            sizes = {}
            for it in items:
                sizes[str(it['id'])] = image_hw(it['low'])
            order = replay_order(len(items), int(cfg_seed), 5, batch_size)
            epochs = {}
            for epoch in range(1, 6):
                perm = order[epoch]['perm']
                ids = [str(items[i]['id']) for i in perm]
                geos = []
                for sid in ids:
                    h, w = sizes[sid]
                    geo = replay_geometry(h, w, int(cfg_seed), epoch, sid, crop)
                    geos.append((sid, geo['iy'], geo['ix'], geo['rot90'], geo['flip_lr'], geo['flip_ud']))
                    trace.write(json.dumps({
                        'dataset': label, 'epoch': epoch, 'id': sid,
                        'h': h, 'w': w, **geo,
                    }) + '\n')
                aug_blob = json.dumps(geos, separators=(',', ':')).encode()
                order_blob = json.dumps(ids).encode()
                batches = []
                id_batches = []
                for batch_idx in order[epoch]['batches']:
                    id_batches.append([str(items[i]['id']) for i in batch_idx])
                dropped = [str(items[i]['id']) for i in order[epoch]['dropped']]
                epochs[str(epoch)] = {
                    'order_sha256': hashlib.sha256(order_blob).hexdigest(),
                    'geometry_sha256': hashlib.sha256(aug_blob).hexdigest(),
                    'n_batches': len(id_batches),
                    'last_10_batch_ids': id_batches[-10:],
                    'dropped_ids': dropped,
                }
                # Brightness of the last 10 batches, fixed definition, current files.
                last_ids = [sid for batch in id_batches[-10:] for sid in batch]
                by_id = {str(it['id']): it for it in items}
                y_low, y_gt, y_ref = [], [], []
                for sid in last_ids:
                    it = by_id[sid]
                    y_low.append(luminance_means(it['low'])['y_mean'])
                    y_gt.append(luminance_means(it['normal'])['y_mean'])
                    y_ref.append(luminance_means(it['ref'])['y_mean'])
                brightness.append({
                    'dataset': label,
                    'epoch': epoch,
                    'n': len(last_ids),
                    'low_y_mean': sum(y_low) / len(y_low),
                    'gt_y_mean': sum(y_gt) / len(y_gt),
                    'ref_y_mean': sum(y_ref) / len(y_ref),
                    'definition': 'ITU-R BT.601-ish 0.299R+0.587G+0.114B on uint8 RGB, divided by 255, mean over last 10 batches',
                })
            # Cross-stage: geometry depends only on epoch+id+size+seed, not on stage.
            summary['datasets'][label] = {
                'n_train': len(items),
                'seeds_seen_in_packs': seeds,
                'batch_size': batch_size,
                'crop': crop,
                'epoch1_randperm_repeat_ok': order['_epoch1_repeat_ok'],
                'epochs': epochs,
                'cross_stage_geometry': (
                    'Identical across short_ft_5ep, short_ft_r2 and short_ft_w91 for the same local epoch, '
                    'because the formula does not read the stage name and every pack of this dataset carries the same seed.'
                    if len(seeds) == 1 else
                    'Pack seeds differ; cross-stage identity does not follow.'
                ),
            }
    summary['last_10_batch_brightness'] = brightness
    return summary


def per_image(eval_cache):
    csv_path = os.path.join(OUT, 'per_image_deltas.csv')
    fields = [
        'dataset', 'stage', 'split', 'transition', 'id',
        'psnr_a', 'psnr_b', 'delta_psnr', 'ssim_a', 'ssim_b', 'delta_ssim',
        'frac_gt1_a', 'frac_gt1_b', 'frac_lt0_a', 'frac_lt0_b',
        'raw_min_a', 'raw_min_b', 'raw_max_a', 'raw_max_b',
        'low', 'normal', 'ref',
    ]
    from dataset.block_amend_lolv2 import collect_triplets, load_manifest_items
    paths = {}
    for label, run, dataset_dir, test_ref in RUNS:
        val_items = load_manifest_items(os.path.join(EXP, run, 'splits', 'split_val.json'))
        for it in val_items:
            paths[(label, 'internal_val', str(it['id']))] = (it['low'], it['normal'], it['ref'])
        ok, _missing = collect_triplets(
            dataset_dir, 'Test', subset='real', ref_subdir=test_ref, require_all=True)
        for it in ok:
            paths[(label, 'official_test', str(it['id']))] = (it['low'], it['normal'], it['ref'])

    transitions = [('ep1', 'ep2', 1, 2), ('ep3', 'ep4', 3, 4), ('ep4', 'ep5', 4, 5)]
    summaries = []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, _run, _d, _t in RUNS:
            for stage in STAGES:
                for split in ('internal_val', 'official_test'):
                    docs = {}
                    for epoch in range(1, 6):
                        docs[epoch] = eval_cache.get((label, stage, epoch, split))
                    if any(docs[e] is None for e in range(1, 6)):
                        summaries.append({
                            'dataset': label, 'stage': stage, 'split': split,
                            'missing': True,
                        })
                        continue
                    id_sets = [set(docs[e]['ids']) for e in range(1, 6)]
                    ids_equal = all(s == id_sets[0] for s in id_sets)
                    for name_a, name_b, ea, eb in transitions:
                        deltas = []
                        for sid in sorted(id_sets[0]):
                            a = docs[ea]['by_id'][sid]
                            b = docs[eb]['by_id'][sid]
                            dp = float(b['psnr_rgb']) - float(a['psnr_rgb'])
                            ds = float(b['ssim_rgb']) - float(a['ssim_rgb'])
                            low, normal, ref = paths.get((label, split, sid), ('', '', ''))
                            writer.writerow({
                                'dataset': label, 'stage': stage, 'split': split,
                                'transition': '%s_to_%s' % (name_a, name_b),
                                'id': sid,
                                'psnr_a': a['psnr_rgb'], 'psnr_b': b['psnr_rgb'],
                                'delta_psnr': dp,
                                'ssim_a': a['ssim_rgb'], 'ssim_b': b['ssim_rgb'],
                                'delta_ssim': ds,
                                'frac_gt1_a': a.get('frac_gt1'), 'frac_gt1_b': b.get('frac_gt1'),
                                'frac_lt0_a': a.get('frac_lt0'), 'frac_lt0_b': b.get('frac_lt0'),
                                'raw_min_a': a.get('raw_min'), 'raw_min_b': b.get('raw_min'),
                                'raw_max_a': a.get('raw_max'), 'raw_max_b': b.get('raw_max'),
                                'low': low, 'normal': normal, 'ref': ref,
                            })
                            deltas.append((sid, dp, ds, float(a.get('frac_gt1') or 0), float(b.get('frac_gt1') or 0)))
                        dps = [dp for _s, dp, _ds, _fa, _fb in deltas]
                        n = len(dps)
                        worst = sorted(deltas, key=lambda x: x[1])[:10]
                        worst5 = worst[:5]
                        summaries.append({
                            'dataset': label,
                            'stage': stage,
                            'split': split,
                            'transition': '%s_to_%s' % (name_a, name_b),
                            'ids_equal_across_epochs': ids_equal,
                            'n': n,
                            'mean_delta_psnr': sum(dps) / n,
                            'median_delta_psnr': statistics.median(dps),
                            'q25_delta_psnr': quantile(dps, 0.25),
                            'q75_delta_psnr': quantile(dps, 0.75),
                            'n_down': sum(1 for v in dps if v < 0),
                            'n_up': sum(1 for v in dps if v > 0),
                            'n_equal': sum(1 for v in dps if v == 0),
                            'n_down_gt_0_5db': sum(1 for v in dps if v < -0.5),
                            'n_down_gt_1db': sum(1 for v in dps if v < -1.0),
                            'worst10': [{'id': s, 'delta_psnr': dp} for s, dp, _ds, _a, _b in worst],
                            'worst5_contribution': sum(dp for _s, dp, _ds, _a, _b in worst5) / n,
                            'worst10_contribution': sum(dp for _s, dp, _ds, _a, _b in worst) / n,
                            'mean_frac_gt1_a': sum(a for _s, _dp, _ds, a, _b in deltas) / n,
                            'mean_frac_gt1_b': sum(b for _s, _dp, _ds, _a, b in deltas) / n,
                        })
    return summaries


def nikon_audit():
    from dataset.block_amend_lolv2 import collect_triplets, load_manifest_items
    label_run = 'csfi_lfca_v1_data1_nikon_partial_20261119'
    view = os.path.join(EXP, 'views/data1_nikon')
    train_items = load_manifest_items(os.path.join(EXP, label_run, 'splits', 'split_train.json'))
    val_items = load_manifest_items(os.path.join(EXP, label_run, 'splits', 'split_val.json'))
    test_ok, test_missing = collect_triplets(
        view, 'Test', subset='real', ref_subdir='nanobanana_ref', require_all=False)
    train_ok, train_missing = collect_triplets(
        view, 'Train', subset='real', ref_subdir='nanobanana_ref', require_all=False)
    include_ids = {str(it['id']) for it in train_items} | {str(it['id']) for it in val_items}
    collected_ids = {str(it['id']) for it in train_ok}
    # symlink spot check
    ref_dir = os.path.join(view, 'Test', 'nanobanana_ref')
    link_rows = []
    if os.path.isdir(ref_dir):
        names = sorted(os.listdir(ref_dir))
        for name in names:
            path = os.path.join(ref_dir, name)
            link_rows.append({
                'name': name,
                'is_link': os.path.islink(path),
                'link_target': os.readlink(path) if os.path.islink(path) else None,
                'realpath': os.path.realpath(path),
                'suffix': os.path.splitext(name)[1].lower(),
                'target_suffix': os.path.splitext(os.path.realpath(path))[1].lower(),
            })
    groups = {
        'train': train_items,
        'internal_val': val_items,
        'official_test': test_ok,
    }
    csv_path = os.path.join(OUT, 'nikon_split_summary.csv')
    fields = [
        'split', 'id', 'low', 'normal', 'ref', 'low_realpath', 'normal_realpath', 'ref_realpath',
        'low_sha256', 'h', 'w', 'low_y', 'gt_y', 'ref_y',
        'low_y_p10', 'low_y_p50', 'low_y_p90',
        'log2_gt_over_low', 'log2_ref_over_gt', 'ref_rgb_minus_gt_rgb',
    ]
    per_split_y = defaultdict(list)
    content = defaultdict(list)  # sha -> [(split, id)]
    realpaths = defaultdict(list)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, items in groups.items():
            for it in items:
                low_s = luminance_means(it['low'])
                gt_s = luminance_means(it['normal'])
                ref_s = luminance_means(it['ref'])
                file_sha = sha256_file(it['low'])
                content[file_sha].append((split, str(it['id'])))
                lp, np_, rp = os.path.realpath(it['low']), os.path.realpath(it['normal']), os.path.realpath(it['ref'])
                realpaths[lp].append((split, str(it['id']), 'low'))
                import math
                log_gt_low = math.log2((gt_s['y_mean'] + 1e-3) / (low_s['y_mean'] + 1e-3))
                log_ref_gt = math.log2((ref_s['y_mean'] + 1e-3) / (gt_s['y_mean'] + 1e-3))
                rgb_delta = [ref_s['rgb_mean'][c] - gt_s['rgb_mean'][c] for c in range(3)]
                writer.writerow({
                    'split': split, 'id': it['id'],
                    'low': it['low'], 'normal': it['normal'], 'ref': it['ref'],
                    'low_realpath': lp, 'normal_realpath': np_, 'ref_realpath': rp,
                    'low_sha256': file_sha, 'h': low_s['h'], 'w': low_s['w'],
                    'low_y': low_s['y_mean'], 'gt_y': gt_s['y_mean'], 'ref_y': ref_s['y_mean'],
                    'low_y_p10': low_s['y_p10'], 'low_y_p50': low_s['y_p50'], 'low_y_p90': low_s['y_p90'],
                    'log2_gt_over_low': log_gt_low, 'log2_ref_over_gt': log_ref_gt,
                    'ref_rgb_minus_gt_rgb': json.dumps(rgb_delta),
                })
                per_split_y[split].append({
                    'low_y': low_s['y_mean'], 'gt_y': gt_s['y_mean'], 'ref_y': ref_s['y_mean'],
                    'log2_gt_over_low': log_gt_low, 'log2_ref_over_gt': log_ref_gt,
                    'h': low_s['h'], 'w': low_s['w'],
                })
    def pack_stats(rows_):
        def col(name):
            xs = [r[name] for r in rows_]
            return {
                'n': len(xs),
                'min': min(xs), 'p25': quantile(xs, 0.25), 'median': statistics.median(xs),
                'p75': quantile(xs, 0.75), 'max': max(xs), 'mean': sum(xs) / len(xs),
            }
        return {name: col(name) for name in ('low_y', 'gt_y', 'ref_y', 'log2_gt_over_low', 'log2_ref_over_gt')}

    shared_sha = {sha: locs for sha, locs in content.items() if len({s for s, _i in locs}) > 1}
    shared_path = {p: locs for p, locs in realpaths.items() if len({s for s, _i, _k in locs}) > 1}
    jpg_to_png = sum(1 for r in link_rows if r['suffix'] == '.jpg' and r['target_suffix'] == '.png')
    return {
        'train_manifest': len(train_items),
        'val_manifest': len(val_items),
        'collect_train_with_ref': len(train_ok),
        'collect_train_missing_ref': len(train_missing),
        'include_equals_all_with_ref': include_ids == collected_ids,
        'include_minus_collected': sorted(include_ids - collected_ids)[:20],
        'collected_minus_include': sorted(collected_ids - include_ids)[:20],
        'official_test_ok': len(test_ok),
        'official_test_missing_ref': len(test_missing),
        'id_overlap_train_val': sorted(
            {str(i['id']) for i in train_items} & {str(i['id']) for i in val_items})[:10],
        'id_overlap_train_test': sorted(
            {str(i['id']) for i in train_items} & {str(i['id']) for i in test_ok})[:10],
        'id_overlap_val_test': sorted(
            {str(i['id']) for i in val_items} & {str(i['id']) for i in test_ok})[:10],
        'test_ref_links': {
            'n': len(link_rows),
            'jpg_name_png_target': jpg_to_png,
            'examples': link_rows[:5],
        },
        'content_sha_shared_across_splits': [
            {'sha256': sha, 'where': locs} for sha, locs in list(shared_sha.items())[:20]
        ],
        'n_content_sha_shared_across_splits': len(shared_sha),
        'n_realpath_shared_across_splits': len(shared_path),
        'distribution': {split: pack_stats(rows_) for split, rows_ in per_split_y.items()},
        'selection_mechanism': (
            'collect_triplets keeps every Train Low that has a same-scene Normal and a reference file. '
            'It does not sample 500 out of a larger referenced set. split_train_val then shuffles those ids '
            'with random.Random(split_seed=42) and takes round(n*0.1) as val. '
            'The 2650 excluded names are Low files whose reference path is absent.'
        ),
    }


def log_lrs():
    found = []
    for _label, run, _d, _t in RUNS:
        for stage in STAGES:
            path = os.path.join(EXP, run, stage, 'block_amend.log')
            lrs = []
            if os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        if 'Epoch ' in line and ' lr=' in line:
                            lrs.append(line.strip().split('INFO: ', 1)[-1])
            found.append({'run': run, 'stage': stage, 'epoch_lr_lines': lrs})
    return found


def main():
    os.makedirs(OUT, exist_ok=True)
    print('inventory', flush=True)
    rows, _fingerprints, eval_cache = inventory_and_evals()
    # CSV without nested dicts dumped as json
    csv_fields = [
        'dataset', 'run', 'stage', 'local_epoch_filename', 'pack_epoch', 'global_step',
        'path', 'file_sha256', 'model_state_sha256', 'model_variant', 'schema',
        'init_checkpoint', 'rec_w', 'ms_ssim_w', 'optimizer', 'lr_config', 'lr_param_group',
        'gamma', 'scheduler_last_epoch', 'adam_steps', 'seed', 'crop', 'batch_size',
        'num_workers', 'train_hash', 'val_hash', 'train_ref_subdir', 'test_ref_subdir',
        'dataset_dir', 'metric_protocol', 'val_json', 'test_json',
    ]
    with open(os.path.join(OUT, 'artifact_inventory.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out['adam_steps'] = json.dumps(row.get('adam_steps'))
            writer.writerow(out)
    write_json('artifact_inventory.json', [
        {k: v for k, v in row.items() if k not in ('val_check', 'test_check')} | {
            'val_problems': (row.get('val_check') or {}).get('problems'),
            'test_problems': (row.get('test_check') or {}).get('problems'),
            'test_path_matches': (row.get('test_check') or {}).get('path_matches_this_file'),
            'test_epoch_matches': (row.get('test_check') or {}).get('epoch_matches_filename'),
            'test_source_guarantee': (row.get('test_check') or {}).get('source_guarantee'),
        }
        for row in rows
    ])
    checks, unique = parent_checks(rows)
    print('augmentation replay', flush=True)
    aug = augmentation(rows)
    write_json('augmentation_replay_summary.json', aug)
    print('per-image deltas', flush=True)
    deltas = per_image(eval_cache)
    write_json('per_image_delta_summary.json', deltas)
    print('nikon', flush=True)
    nikon = nikon_audit()
    write_json('nikon_audit.json', nikon)
    protocol = {
        'git_head_at_audit': os.popen('git -C %s rev-parse HEAD' % ROOT).read().strip(),
        'dataset_py_unchanged_since_amend4': True,
        'block_amend_main_delta_since_amend4': 'save_every_epoch snapshot only',
        'short_ft_driver_committed_after_the_runs': True,
        'weights_only_load': 'load_strict_state loads model_state_dict only; short_ft builds a new Adam',
        'eval_dir_guard': 'queue exits if the output directory exists; it does not validate an existing JSON',
        'parent_checks': checks,
        'model_fingerprint_uniqueness': unique,
        'logged_lrs': log_lrs(),
        'no_checkpoint_hash_inside_eval_json': True,
    }
    write_json('training_protocol_audit.json', protocol)
    print('done', OUT, flush=True)


if __name__ == '__main__':
    main()
