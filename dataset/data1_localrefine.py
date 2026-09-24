"""V2 dataset: one geometric draw applied to all four aligned images.

The whole point is that low / high / generated-reference / cached-Y0 share a
single coordinate system and a *single* augmentation draw, so the two arms
(``self`` and ``nano``) see identical samples and identical geometry -- the only
difference between them is which tensor is handed to the refiner as its second
input, and that switch happens outside the dataset on purpose.

Unlike ``data1_nanobanana`` there is **no** independent reference augmentation,
no colour jitter / blur / shift, and no ``Ref = HR.copy()`` fallback anywhere.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from imageio import imread
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_refine_runtime import load_cache                     # noqa: E402


def _read_rgb(path):
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    return im


def _to_tensor(arr):
    """HxWx3 uint8/float -> [3,H,W] float32 in [-1,1]."""
    if arr.dtype == np.uint8:
        a = arr.astype(np.float32) / 127.5 - 1.
    else:
        a = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def _resize_to(arr, h, w):
    """Nearest-free bilinear resize of an HxWx3 array (whole-image coords)."""
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    t = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1))[None]
    t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    return t[0].permute(1, 2, 0).numpy()


def _augment(tensors, crop, rng):
    """Apply ONE draw of (pad, crop origin, rot90, flips) to every tensor.

    tensors: list of [3,H,W] tensors that already share a coordinate system.
    """
    h, w = tensors[0].shape[-2:]
    pad_h = max(0, crop - h)
    pad_w = max(0, crop - w)
    if pad_h or pad_w:
        tensors = [F.pad(t, (0, pad_w, 0, pad_h), mode='replicate') for t in tensors]
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


class LocalRefineTrainSet(Dataset):
    """Returns aligned crops of (low, high, nano, y0)."""

    def __init__(self, rows, cache_dir, crop_size=128, seed=42):
        self.rows = rows
        self.cache_dir = cache_dir
        self.crop_size = crop_size
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        low = _read_rgb(row['low_path'])
        high = _read_rgb(row['high_path'])
        nano = _read_rgb(row['nano_path'])
        h, w = low.shape[:2]
        nano = _resize_to(nano, h, w)          # whole-image coords, then shared crop
        y0 = load_cache(self.cache_dir, row)   # [3,H,W] float32, no clamp
        if y0.shape[1:] != (h, w):
            raise SystemExit('Y0 cache size %s != low %s for %s'
                             % (y0.shape[1:], (h, w), row['sample_id']))
        y0_t = torch.from_numpy(np.ascontiguousarray(y0))
        # One draw per (worker, item); seeded the same in both arms because the
        # dataset body is identical and both runs use the same --seed.
        rng = np.random.default_rng(
            (self.seed * 1000003 + self._epoch * 9973 + idx) % (2 ** 32))
        low_t, high_t, nano_t, y0_t = _augment(
            [_to_tensor(low), _to_tensor(high), _to_tensor(nano), y0_t],
            self.crop_size, rng)
        return dict(low=low_t, high=high_t, nano=nano_t, y0=y0_t,
                    sample_id=row['sample_id'])


class LocalRefineEvalSet(Dataset):
    """Full images (no crop) for the 50 evaluation samples."""

    def __init__(self, rows, cache_dir):
        self.rows = rows
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        low = _read_rgb(row['low_path'])
        high = _read_rgb(row['high_path'])
        nano = _read_rgb(row['nano_path'])
        h, w = low.shape[:2]
        nano = _resize_to(nano, h, w)
        y0 = load_cache(self.cache_dir, row)
        return dict(low=_to_tensor(low)[None], high=_to_tensor(high)[None],
                    nano=_to_tensor(nano)[None],
                    y0=torch.from_numpy(
                        np.ascontiguousarray(y0))[None],
                    sample_id=row['sample_id'], camera=row['camera'])
