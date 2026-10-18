"""Train/val holdout for LOLv2 domain trainers.

Official Test stays in ``manifests_eval.csv`` and is not read unless
``eval_split=test`` and ``final_eval`` are both set. Checkpoint selection
uses the persisted validation split only.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from local_refine_runtime import read_manifest_csv

SCENE_NOTE = (
    'unconfirmed: manifest has no scene id and LOLv2 sample_id is an opaque '
    'token, so the holdout is per-image rather than per-scene'
)


def add_eval_split_args(ap):
    ap.add_argument('--eval_split', choices=['val', 'test'], default='val',
                    help='val selects checkpoints; test is a one-shot official eval')
    ap.add_argument('--final_eval', action='store_true',
                    help='Required together with --eval_split test')
    ap.add_argument('--val_ratio', type=float, default=0.1)


def scene_report(rows):
    """Group ids only when an explicit scene column exists."""
    scenes = []
    for row in rows:
        key = (row.get('scene_id') or row.get('scene') or '').strip()
        scenes.append(key)
    labeled = [s for s in scenes if s]
    if len(labeled) != len(rows):
        return {
            'scene_isolation': 'unconfirmed',
            'scene_note': SCENE_NOTE,
            'n_multi_image_scenes': 0,
        }
    from collections import Counter
    counts = Counter(labeled)
    multi = sum(1 for n in counts.values() if n > 1)
    return {
        'scene_isolation': 'scene_id',
        'scene_note': 'grouped by scene_id/scene column',
        'n_multi_image_scenes': int(multi),
    }


def _write_rows(path, rows):
    if not rows:
        raise SystemExit('refusing to write an empty split: ' + path)
    cols = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _group_ids(rows):
    """One group per scene when scene ids exist; otherwise one id per group."""
    groups = {}
    order = []
    for row in rows:
        key = (row.get('scene_id') or row.get('scene') or '').strip()
        if not key:
            key = 'id:' + row['sample_id']
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row['sample_id'])
    return [groups[k] for k in order]


def prepare_fit_val(rows, man_dir, seed, val_ratio=0.1):
    """Persist a fixed fit/val split. Reuse it only when seed and ratio match."""
    if not (0.0 < float(val_ratio) < 1.0):
        raise SystemExit('val_ratio must be in (0, 1)')
    os.makedirs(man_dir, exist_ok=True)
    meta_path = os.path.join(man_dir, 'split_meta.json')
    fit_path = os.path.join(man_dir, 'manifests_train_fit.csv')
    val_path = os.path.join(man_dir, 'manifests_val.csv')
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise SystemExit('duplicate sample_id in train manifest')

    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path, encoding='utf-8'))
        if int(meta['seed']) != int(seed) or abs(float(meta['val_ratio']) - float(val_ratio)) > 1e-12:
            raise SystemExit(
                'split_meta.json seed/ratio (%s, %s) != requested (%s, %s); '
                'refusing to silently reuse the old schedule split'
                % (meta.get('seed'), meta.get('val_ratio'), seed, val_ratio))
        fit = read_manifest_csv(fit_path)
        val = read_manifest_csv(val_path)
        fit_ids = {r['sample_id'] for r in fit}
        val_ids = {r['sample_id'] for r in val}
        if fit_ids & val_ids:
            raise SystemExit('persisted fit/val overlap')
        if fit_ids | val_ids != set(by_id):
            raise SystemExit('persisted fit/val does not match the train manifest')
        return fit, val, meta

    groups = _group_ids(rows)
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(groups))
    n_val_img = max(1, int(round(len(rows) * float(val_ratio))))
    if n_val_img >= len(rows):
        raise SystemExit('val split would consume the whole train set')
    val_ids = []
    for gi in order:
        if len(val_ids) >= n_val_img and val_ids:
            break
        # Keep scene members together. Skip a group that would empty the fit set.
        group = groups[int(gi)]
        if len(val_ids) + len(group) >= len(rows):
            continue
        val_ids.extend(group)
    if not val_ids:
        raise SystemExit('failed to hold out a validation group')
    val_ids = set(val_ids)
    fit_rows = [by_id[i] for i in sorted(set(by_id) - val_ids)]
    val_rows = [by_id[i] for i in sorted(val_ids)]
    report = scene_report(rows)
    meta = dict(
        seed=int(seed),
        val_ratio=float(val_ratio),
        n_train_manifest=len(rows),
        n_fit=len(fit_rows),
        n_val=len(val_rows),
        fit_ids=sorted(r['sample_id'] for r in fit_rows),
        val_ids=sorted(val_ids),
        **report,
    )
    _write_rows(fit_path, fit_rows)
    _write_rows(val_path, val_rows)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return fit_rows, val_rows, meta


def select_monitor_rows(man_dir, eval_split, final_eval, val_rows):
    """Rows for checkpoint selection. Official Test is not opened here."""
    if eval_split == 'val':
        return list(val_rows), 'val'
    if eval_split == 'test':
        raise SystemExit(
            'Official Test cannot drive early stopping (final_eval=%s). '
            'Train with --eval_split val; score the frozen checkpoint with '
            '--eval_split test --final_eval.' % bool(final_eval))
    raise SystemExit('eval_split must be val or test, got %r' % eval_split)


def load_official_test(man_dir, eval_split, final_eval):
    """Read Official Test only when both flags explicitly allow it."""
    if eval_split != 'test' or not final_eval:
        raise SystemExit(
            'Official Test is blocked. Pass --eval_split test --final_eval.')
    path = os.path.join(man_dir, 'manifests_eval.csv')
    rows = read_manifest_csv(path)
    return rows


def assert_schedule_in_fit(schedule_rows, fit_ids, val_ids):
    ids = {r['sample_id'] for r in schedule_rows}
    leaked = ids & set(val_ids)
    if leaked:
        raise SystemExit(
            'schedule includes validation ids (%d); refusing the old full-train schedule'
            % len(leaked))
    unknown = ids - set(fit_ids)
    if unknown:
        raise SystemExit(
            'schedule includes %d ids outside the fit split' % len(unknown))
