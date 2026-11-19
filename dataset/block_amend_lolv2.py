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
        ok.append(dict(
            id=sid, low=rec['low'], normal=rec['normal'], ref=rec['ref']))

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


def manifest_hash(items: Sequence[Dict]) -> str:
    payload = json.dumps(
        [{'id': x['id'], 'low': os.path.basename(x['low']),
          'normal': os.path.basename(x['normal']),
          'ref': os.path.basename(x['ref'])} for x in items],
        sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def write_coverage_manifests(
    dataset_dir: str,
    out_dir: str,
    subset: str = 'real',
    ref_subdir: str = 'nanobanana_ref_v2',
    split: str = 'Train',
    allow_partial: bool = True,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    try:
        ok, missing = collect_triplets(
            dataset_dir, split, subset=subset, ref_subdir=ref_subdir,
            require_all=not allow_partial)
    except RuntimeError:
        ok, missing = collect_triplets(
            dataset_dir, split, subset=subset, ref_subdir=ref_subdir,
            require_all=False)
    include_path = os.path.join(out_dir, 'train_include_manifest.json')
    exclude_path = os.path.join(out_dir, 'train_exclude_missing_ref.json')
    with open(include_path, 'w') as f:
        json.dump({'count': len(ok), 'items': ok}, f, indent=2)
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
        'include_hash': manifest_hash(ok),
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


def save_split_manifests(train_items, val_items, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'split_train.json')
    val_path = os.path.join(out_dir, 'split_val.json')
    meta = {
        'train_count': len(train_items),
        'val_count': len(val_items),
        'train_hash': manifest_hash(train_items),
        'val_hash': manifest_hash(val_items),
        'train_ids': [x['id'] for x in train_items],
        'val_ids': [x['id'] for x in val_items],
    }
    with open(train_path, 'w') as f:
        json.dump({'items': train_items, 'hash': meta['train_hash']}, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump({'items': val_items, 'hash': meta['val_hash']}, f, indent=2)
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
    """Generated-ref training set with paired geometry. No GT-as-Ref."""

    def __init__(self, items: List[Dict], crop_size: int = 160,
                 ref_degrade: bool = False, seed: int = 42):
        if not items:
            raise ValueError('BlockAmendTrainSet got empty item list')
        self.items = list(items)
        self.crop_size = int(crop_size)
        self.ref_degrade = bool(ref_degrade)
        self.seed = int(seed)
        if self.ref_degrade:
            from dataset.data1 import DegradeRef
            self.degrader = DegradeRef()
        else:
            self.degrader = None

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

        rng = random.Random(self.seed + int(idx) * 10007)
        # Use numpy RNG seeded from python rng for crop; keep deterministic-ish
        # per index while still varying across epochs if caller re-seeds.
        LR, HR, Ref = _paired_pad_crop(LR, HR, Ref, self.crop_size, rng)
        LR, HR, Ref = _paired_flip_rot(LR, HR, Ref, rng)

        if self.degrader is not None:
            Ref = self.degrader(Ref)

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
