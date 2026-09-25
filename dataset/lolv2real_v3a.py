"""Strict LOL-v2-real dataset for V3-A.

Why this exists instead of ``dataset/lolv2real.py``:

  ``lolv2real.TrainSet`` hard-codes ``Ref = HR.copy()`` and has no external
  -reference support at all. Training through it silently uses the ground
  truth as the reference -- the HR fallback that every audit in this project
  forbids -- and nothing errors. ``TestSet`` there does accept ``ref_dir``, but
  the two splits would then disagree about what a reference is.

This module makes the reference mandatory and explicit:

  * every sample must have a matching reference file, or the sample is an error;
  * the one exception is opt-in: ``require_ref=False`` (base training only),
    which is allowed **only** together with ``--no_reference True`` so the
    placeholder can never reach the model. The placeholder is the low input,
    never the GT;
  * value order is fixed: decode -> /127.5-1 -> resize -> shared geometry.

LOL-v2-real is uniform 600x400 in both splits, so the resize branch never runs
here; it is written in the canonical order anyway so that swapping in another
dataset cannot reintroduce the old "resize then skip normalisation" bug.
"""

import os
import sys
import csv

import numpy as np
import torch
from imageio import imread
from torch.utils.data import Dataset
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_files(d):
    """Files only -- six of the lol-v2-real directories hold .ipynb_checkpoints."""
    if not os.path.isdir(d):
        raise SystemExit('directory not found: %s' % d)
    return sorted(f for f in os.listdir(d)
                  if os.path.isfile(os.path.join(d, f))
                  and os.path.splitext(f)[1].lower() in IMG_EXT)


def gt_name_for(low_name):
    """lowNNNNN.png -> normalNNNNN.png (prefix replacement, not basename equality)."""
    base, ext = os.path.splitext(low_name)
    if not base.startswith('low'):
        raise SystemExit('unexpected low filename: %s' % low_name)
    return base.replace('low', 'normal', 1) + ext


def collect_pairs(dataset_dir, split):
    low_dir = os.path.join(dataset_dir, split, 'Low')
    high_dir = os.path.join(dataset_dir, split, 'Normal')
    pairs = []
    for name in list_files(low_dir):
        gt = os.path.join(high_dir, gt_name_for(name))
        if not os.path.isfile(gt):
            raise SystemExit('no GT for %s (expected %s)' % (name, gt))
        pairs.append((name, os.path.join(low_dir, name), gt))
    return pairs


def pairs_from_manifest(path):
    """(name, low, high) rows from a manifest CSV written by the V3-A builder."""
    if not os.path.isfile(path):
        raise SystemExit('manifest not found: %s' % path)
    out = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out.append((os.path.basename(r['low_path']),
                        r['low_path'], r['high_path']))
    return out


def _read_rgb(path):
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError('%s: unsupported shape %s' % (path, im.shape))
    return im


def _to_model_tensor(arr, what='image'):
    """uint8/uint16 HxWx3 -> [3,H,W] float32 in [-1,1] (range from source dtype)."""
    dt = np.dtype(arr.dtype)
    if dt == np.uint8:
        full = 255.0
    elif dt == np.uint16:
        full = 65535.0
    else:
        raise ValueError('%s: unsupported dtype %s; refusing to guess the range'
                         % (what, dt))
    a = arr.astype(np.float32) / (full / 2.0) - 1.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def _resize(t, h, w):
    if t.shape[-2:] == (h, w):
        return t
    return F.interpolate(t[None], size=(h, w), mode='bilinear',
                         align_corners=False)[0]


def read_model_image(path, size=None):
    """decode -> normalise -> (resize). One order for every caller."""
    return _resize(_to_model_tensor(_read_rgb(path), what=path), *size) \
        if size else _to_model_tensor(_read_rgb(path), what=path)


def _geometry(tensors, crop, rng):
    """One draw of (pad, crop origin, rot90, flips) shared by all tensors."""
    h, w = tensors[0].shape[-2:]
    pad_h, pad_w = max(0, crop - h), max(0, crop - w)
    if pad_h or pad_w:
        tensors = [F.pad(t, (0, pad_w, 0, pad_h), mode='reflect') for t in tensors]
        h, w = tensors[0].shape[-2:]
    oy = int(rng.integers(0, h - crop + 1)) if h > crop else 0
    ox = int(rng.integers(0, w - crop + 1)) if w > crop else 0
    tensors = [t[:, oy:oy + crop, ox:ox + crop] for t in tensors]
    k = int(rng.integers(0, 4))
    if k:
        tensors = [torch.rot90(t, k, dims=(1, 2)) for t in tensors]
    if rng.integers(0, 2):
        tensors = [torch.flip(t, dims=(1,)) for t in tensors]
    if rng.integers(0, 2):
        tensors = [torch.flip(t, dims=(2,)) for t in tensors]
    return [t.contiguous() for t in tensors]


class _Base(Dataset):
    crop_size = None
    require_ref = True
    seed = 42
    _data_pass = 0

    def __init__(self, args, split, crop_size=None, require_ref=True,
                 pairs=None, ref_dir=None, y0_cache=None, mismatch_map=None,
                 gen_corrupt=False):
        self.dataset_dir = args.dataset_dir
        # Reference directory resolution, in priority order:
        #   1. explicit argument
        #   2. <dataset_dir>/<split>/<variant>  (split-aware; the same args
        #      object builds both splits, so a single --ref_dir would otherwise
        #      point the TestSet at the Train references)
        #   3. --ref_dir (legacy, same dir for both splits)
        variant = getattr(args, 'v3a_ref_variant', '') or ''
        if ref_dir is not None:
            self.ref_dir = ref_dir
        elif variant:
            self.ref_dir = os.path.join(self.dataset_dir, split, variant)
        else:
            self.ref_dir = getattr(args, 'ref_dir', '') or ''
        self.crop_size = crop_size or getattr(args, 'train_crop_size', 128)
        self.require_ref = require_ref
        self.seed = getattr(args, 'seed', 42)
        # Optional frozen-base output cache; the same geometry draw is applied to
        # Y0 so the refiner never runs the base inside the training loop.
        self.y0_cache = y0_cache
        # name -> donor name, a fixed derangement written once per run. The
        # donor's reference is loaded and given the SAME geometry draw, so
        # "mismatch" differs from "correct" only in content.
        self.mismatch_map = mismatch_map or {}
        self.gen_corrupt = bool(gen_corrupt)
        # ``pairs`` lets the caller pin the exact sample list (base = all 689,
        # refiner = the 639 that have a reference) instead of always taking the
        # whole split.
        self.pairs = list(pairs) if pairs is not None \
            else collect_pairs(self.dataset_dir, split)
        self.ref_map = {}
        if self.ref_dir:
            if not os.path.isdir(self.ref_dir):
                raise SystemExit('ref_dir not found: %s' % self.ref_dir)
            for f in list_files(self.ref_dir):
                self.ref_map[f] = os.path.join(self.ref_dir, f)
        if require_ref and not self.ref_dir:
            raise SystemExit('require_ref=True but --ref_dir was not given')
        if not require_ref and not getattr(args, 'no_reference', False):
            raise SystemExit('require_ref=False is only allowed with '
                             '--no_reference True (the placeholder must be unused)')
        self.n_missing_ref = sum(1 for _, low, _ in self.pairs
                                 if os.path.basename(low) not in self.ref_map)
        if require_ref and self.n_missing_ref:
            names = [os.path.basename(l) for _, l, _ in self.pairs
                     if os.path.basename(l) not in self.ref_map][:5]
            raise SystemExit('%d/%d samples have no reference under %s (first: %s); '
                             'refusing to fall back to HR'
                             % (self.n_missing_ref, len(self.pairs), self.ref_dir, names))

    def set_data_pass(self, data_pass):
        self._data_pass = int(data_pass)
        return self._data_pass

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        name, low_path, high_path = self.pairs[idx]
        lr = _to_model_tensor(_read_rgb(low_path), what=low_path)
        hr = _to_model_tensor(_read_rgb(high_path), what=high_path)
        h, w = lr.shape[-2:]
        y0 = None
        if self.y0_cache:
            p = os.path.join(self.y0_cache, name + '.npy')
            if not os.path.isfile(p):
                raise SystemExit('Y0 cache miss: %s' % p)
            arr = np.load(p)
            if arr.shape[1:] != (h, w):
                raise SystemExit('Y0 cache %s has %s but low is %s'
                                 % (p, arr.shape[1:], (h, w)))
            y0 = torch.from_numpy(np.ascontiguousarray(arr))
        ref_path = self.ref_map.get(name)
        if ref_path:
            ref = read_model_image(ref_path, size=(h, w))
        elif self.require_ref:
            raise SystemExit('%s: reference missing at load time' % name)
        else:
            # Base training only. ``--no_reference True`` is asserted in __init__,
            # so this tensor never reaches the model. It is the LOW input, never
            # the GT, so it cannot masquerade as a reference even by accident.
            ref = lr.clone()
        mis = None
        if self.mismatch_map:
            donor = self.mismatch_map.get(name)
            if donor is None:
                raise SystemExit('%s: no mismatch donor' % name)
            mis = read_model_image(self.ref_map[donor], size=(h, w))
        return name, lr, hr, ref, y0, mis


class TrainSet(_Base):
    """Random-crop training set with a synchronised geometry draw."""

    def __init__(self, args, crop_size=None, require_ref=None, pairs=None,
                 ref_dir=None, y0_cache=None, mismatch_map=None,
                 gen_corrupt=False, split='Train'):
        if require_ref is None:
            require_ref = not getattr(args, 'no_reference', False)
        super().__init__(args, split, crop_size=crop_size,
                         require_ref=require_ref, pairs=pairs, ref_dir=ref_dir,
                         y0_cache=y0_cache, mismatch_map=mismatch_map,
                         gen_corrupt=gen_corrupt)

    def __getitem__(self, idx):
        name, lr, hr, ref, y0, mis = self._load(idx)
        if not self.crop_size:
            # full-image mode (used for evaluation): no geometry draw at all
            d = dict(LR=lr, LR_sr=lr.clone(), HR=hr, Ref=ref, Ref_sr=ref.clone())
            if mis is not None:
                d['Ref_mis'] = mis
                if self.gen_corrupt:
                    from v3a_runtime import splice_corrupt
                    d['Ref_cor'] = splice_corrupt(ref, mis,
                                                  np.random.default_rng(1234 + idx))
            if y0 is not None:
                d['Y0'] = y0
            return d
        rng = np.random.default_rng(
            (self.seed * 1000003 + self._data_pass * 9973 + idx) % (2 ** 32))
        tensors = [lr, hr, ref]
        if mis is not None:
            tensors.append(mis)
        if y0 is not None:
            tensors.append(y0)
        out = _geometry(tensors, self.crop_size, rng)
        lr, hr, ref = out[0], out[1], out[2]
        k = 3
        d = dict(LR=lr, LR_sr=lr.clone(), HR=hr, Ref=ref, Ref_sr=ref.clone())
        if mis is not None:
            d['Ref_mis'] = out[k]
            k += 1
            if self.gen_corrupt:
                from v3a_runtime import splice_corrupt
                d['Ref_cor'] = splice_corrupt(d['Ref'], d['Ref_mis'], rng)
        if y0 is not None:
            d['Y0'] = out[k]
        return d


class TestSet(_Base):
    """Full-image test set. No crop, no geometry randomisation."""

    def __init__(self, args, ref_level='1', pairs=None, y0_cache=None,
                 require_ref=None):
        # A reference-free run needs no references anywhere, including here --
        # without this, training a reference-free base with a held-out split
        # fails at loader construction.
        if require_ref is None:
            require_ref = not getattr(args, 'no_reference', False)
        super().__init__(args, 'Test', require_ref=require_ref, pairs=pairs,
                         y0_cache=y0_cache)

    def __getitem__(self, idx):
        name, lr, hr, ref, y0, _mis = self._load(idx)
        d = dict(LR=lr, LR_sr=lr.clone(), HR=hr, Ref=ref, Ref_sr=ref.clone())
        if y0 is not None:
            d['Y0'] = y0
        return d
