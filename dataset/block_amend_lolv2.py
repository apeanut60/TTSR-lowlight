"""Block-amend LOLv2-Real generated-reference dataset.

Strict Low / GT / Nano triplets. GT never fills missing Ref.
Paired geometric augmentation (shared crop / flip / rotate).
Train/val split via fixed manifests.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from imageio import imread
from torch.utils.data import Dataset
import torch


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
# v3: hash and imread share one relative path. v2 hashed *_rel and could
# read a different absolute path. A v2 hash is not reused or rewritten.
MANIFEST_HASH_VERSION = 'v3'
REFERENCE_SOURCE = 'generated_nano'
_PATH_KEYS = ('low', 'normal', 'ref')
# Compared for resume and for an independent internal-val eval.
SPLIT_IDENTITY_FIELDS = (
    'train_hash', 'val_hash', 'ref_subdir',
    'manifest_hash_version', 'reference_source',
)


def stable_aug_seed(base_seed: int, epoch: int, sample_id: str) -> int:
    """Stable integer seed from base_seed/epoch/sample_id (not Python hash())."""
    payload = ('%s|%s|%s' % (int(base_seed), int(epoch), str(sample_id))).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def normalize_rel_path(path: str, dataset_dir: str) -> str:
    path = os.path.normpath(path)
    root = os.path.normpath(dataset_dir)
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        rel = path
    return rel.replace('\\', '/')


def canonical_rel(rel: str) -> str:
    """One spelling for a dataset-relative path. Rejects absolute and '..'."""
    text = str(rel).strip().replace('\\', '/')
    if os.path.isabs(text):
        raise RuntimeError('manifest path is absolute, expected relative: %s' % text)
    parts = []
    for part in text.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            raise RuntimeError('manifest path escapes the dataset root: %s' % text)
        parts.append(part)
    if not parts:
        raise RuntimeError('empty manifest path')
    return '/'.join(parts)


def materialize_item(item: Dict, read_root: str) -> Dict:
    """Resolve ``low``/``normal``/``ref`` from the relative paths.

    The relative path is the only source of truth. When the record also
    carries an absolute path, it must equal ``recorded dataset_dir + rel``;
    otherwise this raises and names both paths. It does not keep whichever
    file happens to exist. ``read_root`` may be a new dataset root: every
    file is then rebuilt under that one root.
    """
    read_root = os.path.normpath(read_root)
    recorded = os.path.normpath(item.get('dataset_dir') or read_root)
    out = dict(item)
    mismatches = []
    sid = item.get('id')
    for key in _PATH_KEYS:
        rel_raw = item.get(key + '_rel')
        abs_raw = item.get(key)
        if rel_raw:
            rel = canonical_rel(rel_raw)
        elif abs_raw:
            rel = canonical_rel(normalize_rel_path(abs_raw, recorded))
        else:
            raise RuntimeError('id=%s has no %s path' % (sid, key))
        expected = os.path.normpath(os.path.join(recorded, *rel.split('/')))
        if abs_raw and os.path.normpath(abs_raw) != expected:
            mismatches.append((key, os.path.normpath(abs_raw), expected, rel))
        out[key + '_rel'] = rel
        out[key] = os.path.normpath(os.path.join(read_root, *rel.split('/')))
    if mismatches:
        detail = '; '.join(
            'id=%s field=%s absolute=%s recorded+rel=%s (rel=%s)'
            % (sid, key, actual, expected, rel)
            for key, actual, expected, rel in mismatches)
        raise RuntimeError(
            'Manifest absolute path and relative path disagree. Neither path '
            'is chosen automatically. %s' % detail)
    out['dataset_dir'] = read_root
    if 'reference_source' not in out:
        out['reference_source'] = REFERENCE_SOURCE
    return out


def materialize_items(items: Sequence[Dict], read_root: str) -> List[Dict]:
    return [materialize_item(it, read_root) for it in items]


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def sample_id_from_low_name(low_name: str) -> str:
    base = os.path.splitext(os.path.basename(low_name))[0]
    # lolv2real: low00001 -> 00001; syn: keep basename
    if base.startswith('low'):
        return base[3:]
    return base


def normal_name_from_low(low_name: str, subset: str = 'real') -> str:
    if subset == 'real':
        return low_name.replace('low', 'normal')
    return low_name


def collect_triplets(
    dataset_dir: str,
    split: str,
    subset: str = 'real',
    ref_subdir: str = 'nanobanana_ref_v2',
    require_all: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Map by sample id. Returns (ok_list, missing_ref_list).

    Each ok item: {id, low, normal, ref}.
    If ``require_all`` and any missing ref among Low∩Normal, raises after
    writing nothing — caller may use missing list for manifest export.
    """
    base = os.path.join(dataset_dir, split)
    low_dir = os.path.join(base, 'Low')
    normal_dir = os.path.join(base, 'Normal')
    ref_dir = os.path.join(base, ref_subdir)
    if not os.path.isdir(low_dir):
        raise FileNotFoundError('Missing Low dir: %s' % low_dir)
    if not os.path.isdir(normal_dir):
        raise FileNotFoundError('Missing Normal dir: %s' % normal_dir)
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(
            'Missing generated-reference dir: %s (ref_subdir=%s)'
            % (ref_dir, ref_subdir))

    low_files = sorted(
        f for f in os.listdir(low_dir)
        if f.endswith(IMG_EXTS) or f.lower().endswith(('.png', '.jpg', '.jpeg')))
    by_id = {}
    ambiguous = []
    for low_name in low_files:
        sid = sample_id_from_low_name(low_name)
        if sid in by_id:
            ambiguous.append((sid, by_id[sid]['low_name'], low_name))
            continue
        normal_name = normal_name_from_low(low_name, subset)
        low_path = os.path.join(low_dir, low_name)
        normal_path = os.path.join(normal_dir, normal_name)
        # Prefer same basename in ref; also try low_name / sid.png
        candidates = [
            os.path.join(ref_dir, low_name),
            os.path.join(ref_dir, normal_name),
            os.path.join(ref_dir, sid + os.path.splitext(low_name)[1]),
        ]
        ref_path = next((p for p in candidates if os.path.isfile(p)), None)
        by_id[sid] = dict(
            id=sid, low_name=low_name, low=low_path, normal=normal_path,
            ref=ref_path, has_normal=os.path.isfile(normal_path),
            has_ref=ref_path is not None)

    if ambiguous:
        raise RuntimeError(
            'Ambiguous sample id mapping: %s' % (ambiguous[:5],))

    ok, missing_ref, missing_gt = [], [], []
    for sid, rec in sorted(by_id.items()):
        if not rec['has_normal']:
            missing_gt.append(rec)
            continue
        if not rec['has_ref']:
            missing_ref.append(rec)
            continue
        rels = {
            'low_rel': canonical_rel(normalize_rel_path(rec['low'], dataset_dir)),
            'normal_rel': canonical_rel(normalize_rel_path(rec['normal'], dataset_dir)),
            'ref_rel': canonical_rel(normalize_rel_path(rec['ref'], dataset_dir)),
        }
        root = os.path.normpath(dataset_dir)
        ok.append(dict(
            id=sid,
            # Absolute paths are derived from the relative paths, not a second source.
            low=os.path.normpath(os.path.join(root, *rels['low_rel'].split('/'))),
            normal=os.path.normpath(os.path.join(root, *rels['normal_rel'].split('/'))),
            ref=os.path.normpath(os.path.join(root, *rels['ref_rel'].split('/'))),
            split=split,                 # source partition (Train / Test)
            ref_subdir=ref_subdir,
            reference_source=REFERENCE_SOURCE,
            dataset_dir=root,
            **rels,
        ))

    if require_all and missing_ref:
        raise RuntimeError(
            'Generated-reference mode: %d samples missing Nano ref under %s. '
            'First: %s. Use write_coverage_manifests(..., allow_partial=True) '
            'to export include/exclude lists.'
            % (len(missing_ref), ref_dir,
               [m['low_name'] for m in missing_ref[:5]]))
    if missing_gt:
        raise RuntimeError(
            'Missing GT/Normal for %d lows; first=%s'
            % (len(missing_gt), [m['low_name'] for m in missing_gt[:5]]))
    return ok, missing_ref


def file_sha256(path: str, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_identity_hash(
    items: Sequence[Dict],
    dataset_dir: str,
    split_name: str,
    ref_subdir: str,
    reference_source: str = REFERENCE_SOURCE,
) -> str:
    """Identity hash over the same relative paths imread will open.

    Locks the sample ids, the split role, the reference version and the
    relative paths. It is not a content fingerprint: replacing a file without
    changing its path does not change this hash, and file bytes are not
    re-hashed every epoch.

    ``split_name`` is the role (train / val / test), not the source partition.
    Source partition stays on the item (``split``, Train or Test). Internal
    val stays a role over the Train partition; it is never Official Test.
    """
    dataset_dir = os.path.normpath(dataset_dir)
    rows = []
    for x in sorted(items, key=lambda z: str(z['id'])):
        row = {
            'id': str(x['id']),
            'role': split_name,
            'source_split': str(x.get('split') or ''),
            'ref_subdir': ref_subdir,
            'reference_source': reference_source,
        }
        for key in _PATH_KEYS:
            rel_raw = x.get(key + '_rel')
            if not rel_raw:
                raise RuntimeError(
                    'id=%s has no %s_rel. The hash refuses to fall back to a '
                    'separate absolute path.' % (x.get('id'), key))
            rel = canonical_rel(rel_raw)
            resolved = os.path.normpath(os.path.join(dataset_dir, *rel.split('/')))
            actual = os.path.normpath(x[key])
            if actual != resolved:
                raise RuntimeError(
                    'id=%s field=%s hash path %s != read path %s'
                    % (x.get('id'), key, resolved, actual))
            row[key] = rel
        rows.append(row)
    payload = json.dumps(
        {'version': MANIFEST_HASH_VERSION, 'items': rows},
        sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def manifest_hash(items: Sequence[Dict], dataset_dir: str = '',
                  split_name: str = '', ref_subdir: str = '') -> str:
    """Backward-compatible wrapper; prefers identity hash when context given."""
    if dataset_dir and (split_name or ref_subdir or items):
        # Infer split/ref from items when possible
        sn = split_name or (items[0].get('split') if items else '')
        rs = ref_subdir or (items[0].get('ref_subdir') if items else '')
        dd = dataset_dir or (items[0].get('dataset_dir') if items else '')
        return manifest_identity_hash(items, dd, sn, rs)
    # Legacy basename-only (discouraged); still version-tagged
    payload = json.dumps(
        {'version': 'v1_basename',
         'items': [{'id': x['id'],
                    'low': os.path.basename(x['low']),
                    'normal': os.path.basename(x['normal']),
                    'ref': os.path.basename(x['ref'])} for x in items]},
        sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def validate_split_manifests(
    train_items: List[Dict],
    val_items: List[Dict],
    dataset_dir: str,
    ref_subdir: str,
    batch_size: int,
    drop_last: bool = True,
) -> Dict:
    """Shared validation for create/load/resume/eval. Raises on failure.

    Items are materialized in place: ``low``/``normal``/``ref`` become
    ``read_root + rel`` and the hash is taken from those same relative paths.
    """
    if not train_items:
        raise RuntimeError('train manifest is empty')
    if not val_items:
        raise RuntimeError('val manifest is empty')
    train_items[:] = materialize_items(train_items, dataset_dir)
    val_items[:] = materialize_items(val_items, dataset_dir)

    def _check_unique(items, name):
        ids = [x['id'] for x in items]
        if len(ids) != len(set(ids)):
            from collections import Counter
            dup = [i for i, c in Counter(ids).items() if c > 1]
            raise RuntimeError('%s has duplicate ids: %s' % (name, dup[:10]))
        return set(ids)

    train_ids = _check_unique(train_items, 'train')
    val_ids = _check_unique(val_items, 'val')
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(
            'train/val id overlap: %s' % sorted(overlap)[:10])

    # Every Low/GT/Ref file is owned by exactly one (role, id, field).
    path_owner = {}
    for split_name, items in (('train', train_items), ('val', val_items)):
        for it in items:
            for key in _PATH_KEYS:
                p = os.path.normpath(it[key])
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        '%s missing %s for id=%s path=%s'
                        % (split_name, key, it['id'], p))
                prev = path_owner.get(p)
                if prev is not None and prev != (split_name, it['id'], key):
                    raise RuntimeError(
                        'Path reused (%s): %s is %s id=%s %s and %s id=%s %s'
                        % (key, p, prev[0], prev[1], prev[2],
                           split_name, it['id'], key))
                path_owner[p] = (split_name, it['id'], key)
            if os.path.normpath(it['ref']) == os.path.normpath(it['normal']):
                raise RuntimeError(
                    'GT path used as the reference for id=%s' % it['id'])

    if drop_last and len(train_items) < int(batch_size):
        raise RuntimeError(
            'train size %d < batch_size %d with drop_last=True would yield '
            'zero optimizer updates' % (len(train_items), batch_size))

    train_hash = manifest_identity_hash(
        train_items, dataset_dir, 'train', ref_subdir)
    val_hash = manifest_identity_hash(
        val_items, dataset_dir, 'val', ref_subdir)
    return {
        'train_count': len(train_items),
        'val_count': len(val_items),
        'train_hash': train_hash,
        'val_hash': val_hash,
        'manifest_hash_version': MANIFEST_HASH_VERSION,
        'ref_subdir': ref_subdir,
        'reference_source': REFERENCE_SOURCE,
        'dataset_dir': os.path.normpath(dataset_dir),
        'source_partition': 'Train',
    }


def assert_split_identity(live: Dict, saved: Dict, context: str) -> None:
    """One check for resume and for an independent internal-val eval.

    Does not re-split and does not rewrite the checkpoint hash.
    """
    if not isinstance(saved, dict):
        raise RuntimeError('%s: checkpoint has no split_identity' % context)
    bad = []
    for key in SPLIT_IDENTITY_FIELDS:
        if saved.get(key) != live.get(key):
            bad.append('%s: checkpoint=%r live=%r'
                       % (key, saved.get(key), live.get(key)))
    if bad:
        raise RuntimeError(
            '%s split identity mismatch. Refusing to treat this manifest as '
            'the checkpoint\'s internal val, to re-split, or to update the '
            'checkpoint.\n%s' % (context, '\n'.join(bad)))


def write_coverage_manifests(
    dataset_dir: str,
    out_dir: str,
    subset: str = 'real',
    ref_subdir: str = 'nanobanana_ref_v2',
    split: str = 'Train',
    allow_partial: bool = True,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    # Missing references are the only gap this mode drops. Missing GT,
    # ambiguous ids and bad directories still raise from collect_triplets.
    ok, missing = collect_triplets(
        dataset_dir, split, subset=subset, ref_subdir=ref_subdir,
        require_all=not allow_partial)
    include_path = os.path.join(out_dir, 'train_include_manifest.json')
    exclude_path = os.path.join(out_dir, 'train_exclude_missing_ref.json')
    with open(include_path, 'w') as f:
        json.dump({
            'count': len(ok),
            'items': ok,
            'dataset_dir': os.path.normpath(dataset_dir),
            'ref_subdir': ref_subdir,
            'reference_source': REFERENCE_SOURCE,
            'source_split': split,
            'manifest_hash_version': MANIFEST_HASH_VERSION,
        }, f, indent=2)
    with open(exclude_path, 'w') as f:
        json.dump({
            'count': len(missing),
            'items': [{'id': m['id'], 'low_name': m['low_name']} for m in missing],
        }, f, indent=2)
    report = {
        'include_count': len(ok),
        'exclude_missing_ref': len(missing),
        'include_manifest': include_path,
        'exclude_manifest': exclude_path,
        'include_hash': manifest_identity_hash(
            ok, dataset_dir, 'train_include', ref_subdir),
        'manifest_hash_version': MANIFEST_HASH_VERSION,
    }
    with open(os.path.join(out_dir, 'coverage_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report


def split_train_val(
    items: List[Dict],
    val_ratio: float = 0.1,
    split_seed: int = 42,
    train_ids: Optional[Sequence[str]] = None,
    val_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    by_id = {x['id']: x for x in items}
    if train_ids is not None or val_ids is not None:
        if train_ids is None or val_ids is None:
            raise ValueError('Both train_ids and val_ids required when either set')
        train_set, val_set = set(train_ids), set(val_ids)
        if train_set & val_set:
            raise RuntimeError(
                'train/val id overlap: %s' % sorted(train_set & val_set)[:10])
        missing = (train_set | val_set) - set(by_id)
        if missing:
            raise RuntimeError('Manifest ids not in triplets: %s'
                               % sorted(missing)[:10])
        train = [by_id[i] for i in sorted(train_set)]
        val = [by_id[i] for i in sorted(val_set)]
        return train, val

    ids = sorted(by_id.keys())
    rng = random.Random(int(split_seed))
    rng.shuffle(ids)
    n_val = int(round(len(ids) * float(val_ratio)))
    n_val = max(1, n_val) if len(ids) >= 10 else max(0, n_val)
    val_ids_l = set(ids[:n_val])
    train_ids_l = [i for i in ids if i not in val_ids_l]
    if set(train_ids_l) & val_ids_l:
        raise RuntimeError('internal split overlap')
    train = [by_id[i] for i in sorted(train_ids_l)]
    val = [by_id[i] for i in sorted(val_ids_l)]
    return train, val


def save_split_manifests(
    train_items, val_items, out_dir: str,
    dataset_dir: str = '', ref_subdir: str = '',
    batch_size: int = 1,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    if not dataset_dir and train_items:
        dataset_dir = train_items[0].get('dataset_dir', '')
    if not ref_subdir and train_items:
        ref_subdir = train_items[0].get('ref_subdir', '')
    meta = validate_split_manifests(
        train_items, val_items, dataset_dir, ref_subdir,
        batch_size=batch_size, drop_last=True)
    meta['train_ids'] = [x['id'] for x in train_items]
    meta['val_ids'] = [x['id'] for x in val_items]
    train_path = os.path.join(out_dir, 'split_train.json')
    val_path = os.path.join(out_dir, 'split_val.json')
    with open(train_path, 'w') as f:
        json.dump({
            'items': train_items,
            'hash': meta['train_hash'],
            'manifest_hash_version': MANIFEST_HASH_VERSION,
            'split': 'train',
            'ref_subdir': ref_subdir,
            'dataset_dir': os.path.normpath(dataset_dir),
        }, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump({
            'items': val_items,
            'hash': meta['val_hash'],
            'manifest_hash_version': MANIFEST_HASH_VERSION,
            'split': 'val',
            'ref_subdir': ref_subdir,
            'dataset_dir': os.path.normpath(dataset_dir),
        }, f, indent=2)
    with open(os.path.join(out_dir, 'split_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def load_manifest_items(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'items' in data:
        return data['items']
    if isinstance(data, list):
        return data
    raise ValueError('Bad manifest format: %s' % path)


def _paired_pad_crop(
    lr: np.ndarray, hr: np.ndarray, ref: np.ndarray, crop_size: int, rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad each image independently with reflect, then shared crop coords."""
    def pad_to(img):
        h, w = img.shape[:2]
        ph = max(0, crop_size - h)
        pw = max(0, crop_size - w)
        if ph == 0 and pw == 0:
            return img
        return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')

    lr, hr, ref = pad_to(lr), pad_to(hr), pad_to(ref)
    # After pad, Low/GT should match; Ref was resized to Low before this.
    h = min(lr.shape[0], hr.shape[0], ref.shape[0])
    w = min(lr.shape[1], hr.shape[1], ref.shape[1])
    if h < crop_size or w < crop_size:
        raise RuntimeError(
            'After pad still too small for crop_size=%d: %sx%s'
            % (crop_size, h, w))
    iy = rng.randrange(0, h - crop_size + 1)
    ix = rng.randrange(0, w - crop_size + 1)
    sl = (slice(iy, iy + crop_size), slice(ix, ix + crop_size), slice(None))
    return lr[sl].copy(), hr[sl].copy(), ref[sl].copy()


def _paired_flip_rot(
    lr, hr, ref, rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = rng.randint(0, 3)
    if k:
        lr = np.rot90(lr, k).copy()
        hr = np.rot90(hr, k).copy()
        ref = np.rot90(ref, k).copy()
    if rng.randint(0, 1):
        lr = np.fliplr(lr).copy()
        hr = np.fliplr(hr).copy()
        ref = np.fliplr(ref).copy()
    if rng.randint(0, 1):
        lr = np.flipud(lr).copy()
        hr = np.flipud(hr).copy()
        ref = np.flipud(ref).copy()
    return lr, hr, ref


def _to_norm(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 127.5 - 1.


class BlockAmendTrainSet(Dataset):
    """Generated-ref training set with paired geometry. No GT-as-Ref.

    Call ``set_epoch(epoch)`` before each training epoch (and before creating
    the DataLoader iterator). Augmentation RNG depends on
    ``(base_seed, epoch, sample_id)`` via ``stable_aug_seed`` — not on GT pixels.
    """

    def __init__(self, items: List[Dict], crop_size: int = 160,
                 ref_degrade: bool = False, seed: int = 42):
        if not items:
            raise ValueError('BlockAmendTrainSet got empty item list')
        self.items = list(items)
        self.crop_size = int(crop_size)
        self.ref_degrade = bool(ref_degrade)
        self.seed = int(seed)
        self.epoch = 0
        if self.ref_degrade:
            from dataset.data1 import DegradeRef
            self.degrader = DegradeRef()
        else:
            self.degrader = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        # Fail hard if a manifested file disappeared.
        for key in ('low', 'normal', 'ref'):
            if not os.path.isfile(rec[key]):
                raise FileNotFoundError(
                    'Manifested %s missing for id=%s path=%s'
                    % (key, rec['id'], rec[key]))

        LR = _to_rgb_uint8(imread(rec['low']))
        HR = _to_rgb_uint8(imread(rec['normal']))
        Ref = _to_rgb_uint8(imread(rec['ref']))

        if LR.shape[:2] != HR.shape[:2]:
            raise RuntimeError(
                'Low/GT size mismatch id=%s: %s vs %s'
                % (rec['id'], LR.shape[:2], HR.shape[:2]))

        h, w = LR.shape[:2]
        if Ref.shape[:2] != (h, w):
            Ref = cv2.resize(Ref, (w, h), interpolation=cv2.INTER_LINEAR)

        # Seed from base_seed + epoch + stable sample_id (never GT content).
        rng = random.Random(stable_aug_seed(self.seed, self.epoch, rec['id']))
        LR, HR, Ref = _paired_pad_crop(LR, HR, Ref, self.crop_size, rng)
        LR, HR, Ref = _paired_flip_rot(LR, HR, Ref, rng)

        if self.degrader is not None:
            # DegradeRef uses its own RNG; seed from the same stream for pairing.
            Ref = self.degrader(Ref, seed=rng.randint(0, 2 ** 31 - 1))

        LR_sr, Ref_sr = LR.copy(), Ref.copy()
        sample = {
            'LR': torch.from_numpy(_to_norm(LR).transpose(2, 0, 1)).float(),
            'LR_sr': torch.from_numpy(_to_norm(LR_sr).transpose(2, 0, 1)).float(),
            'HR': torch.from_numpy(_to_norm(HR).transpose(2, 0, 1)).float(),
            'Ref': torch.from_numpy(_to_norm(Ref).transpose(2, 0, 1)).float(),
            'Ref_sr': torch.from_numpy(_to_norm(Ref_sr).transpose(2, 0, 1)).float(),
            'id': rec['id'],
        }
        return sample


class BlockAmendEvalSet(Dataset):
    """Full-image eval set (internal val or Official Test). No GT-as-Ref."""

    def __init__(self, items: List[Dict]):
        if not items:
            raise ValueError('BlockAmendEvalSet got empty item list')
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        for key in ('low', 'normal', 'ref'):
            if not os.path.isfile(rec[key]):
                raise FileNotFoundError(
                    'Manifested %s missing for id=%s path=%s'
                    % (key, rec['id'], rec[key]))
        LR = _to_rgb_uint8(imread(rec['low']))
        HR = _to_rgb_uint8(imread(rec['normal']))
        Ref = _to_rgb_uint8(imread(rec['ref']))
        if LR.shape[:2] != HR.shape[:2]:
            raise RuntimeError(
                'Low/GT size mismatch id=%s: %s vs %s'
                % (rec['id'], LR.shape[:2], HR.shape[:2]))
        h, w = LR.shape[:2]
        if Ref.shape[:2] != (h, w):
            Ref = cv2.resize(Ref, (w, h), interpolation=cv2.INTER_LINEAR)
        LR_sr, Ref_sr = LR.copy(), Ref.copy()
        return {
            'LR': torch.from_numpy(_to_norm(LR).transpose(2, 0, 1)).float(),
            'LR_sr': torch.from_numpy(_to_norm(LR_sr).transpose(2, 0, 1)).float(),
            'HR': torch.from_numpy(_to_norm(HR).transpose(2, 0, 1)).float(),
            'Ref': torch.from_numpy(_to_norm(Ref).transpose(2, 0, 1)).float(),
            'Ref_sr': torch.from_numpy(_to_norm(Ref_sr).transpose(2, 0, 1)).float(),
            'id': rec['id'],
        }


# Synthetic helpers for unit tests (coordinate grids)
def make_coord_triplet(h, w, crop_size=None):
    """Create Low/GT/Ref with unique spatial patterns for aug tests."""
    yy, xx = np.mgrid[0:h, 0:w]
    low = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    gt = np.stack([(xx * 2) % 256, (yy * 3) % 256, 128 * np.ones_like(xx)],
                  axis=-1).astype(np.uint8)
    ref = np.stack([(255 - xx) % 256, (xx + 2 * yy) % 256, (yy * 5) % 256],
                   axis=-1).astype(np.uint8)
    return low, gt, ref
