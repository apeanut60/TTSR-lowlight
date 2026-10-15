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


# Explicitly supported raster dtypes -> full-scale value. The previous code
# inferred "this is already in [-1,1]" from ``dtype == float``, so a *resized*
# 0..255 array silently entered the network around 128 instead of ~0. Do not
# reintroduce a dtype guess: an unknown dtype must raise.
_RASTER_FULL_SCALE = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}


def read_rgb(path):
    """Decode one raster image to an HxWx3 array. No value mapping here."""
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError('%s: unsupported image shape %s' % (path, im.shape))
    return im


def raster_to_model_tensor(arr, what='image'):
    """HxWx3 raster -> [3,H,W] float32 in [-1,1].

    The mapping is chosen from the *source* dtype only; ``what`` is used for
    error messages.
    """
    dt = np.dtype(arr.dtype)
    if dt not in _RASTER_FULL_SCALE:
        raise ValueError('%s: unsupported dtype %s (expected uint8/uint16); '
                         'refusing to guess the value range' % (what, dt))
    a = arr.astype(np.float32) / (_RASTER_FULL_SCALE[dt] / 2.0) - 1.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def resize_tensor(t, h, w):
    """Bilinear resize of an already model-scale [3,H,W] tensor."""
    if t.shape[-2:] == (h, w):
        return t
    return F.interpolate(t[None], size=(h, w), mode='bilinear',
                         align_corners=False)[0]


def read_rgb_model_tensor(path, size=None):
    """The single raster -> model tensor path: normalise first, then resize.

    Shared by the training set, the eval set and (via the same ordering)
    ``local_refine_runtime._read_eval_tensor``, so the three cannot drift.
    """
    t = raster_to_model_tensor(read_rgb(path), what=os.path.basename(path))
    if size is not None:
        t = resize_tensor(t, size[0], size[1])
    return t


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

    def set_data_pass(self, data_pass):
        """Advance the augmentation pass used to seed this sample's geometry.

        Geometry is seeded from ``(seed, pass, idx)``. Without this call every
        sample keeps the *same* crop / rotation / flips for the whole run, so a
        3000-update run would only ever see 500 distinct training crops. Must be
        called before the DataLoader iterator for that pass is created.
        """
        self._epoch = int(data_pass)
        return self._epoch

    def __getitem__(self, idx):
        row = self.rows[idx]
        low_t = read_rgb_model_tensor(row['low_path'])
        high_t = read_rgb_model_tensor(row['high_path'])
        h, w = low_t.shape[-2:]
        # Reference is normalised first, then resized in whole-image coords, so
        # it shares the crop grid with low/high/y0.
        nano_t = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        y0 = load_cache(self.cache_dir, row)   # [3,H,W] float32, no clamp
        if y0.shape[1:] != (h, w):
            raise SystemExit('Y0 cache size %s != low %s for %s'
                             % (y0.shape[1:], (h, w), row['sample_id']))
        y0_t = torch.from_numpy(np.ascontiguousarray(y0))   # already model scale
        # One draw per (worker, item); seeded the same in both arms because the
        # dataset body is identical and both runs use the same --seed.
        rng = np.random.default_rng(
            (self.seed * 1000003 + self._epoch * 9973 + idx) % (2 ** 32))
        low_t, high_t, nano_t, y0_t = _augment(
            [low_t, high_t, nano_t, y0_t],
            self.crop_size, rng)
        return dict(low=low_t, high=high_t, nano=nano_t, y0=y0_t,
                    sample_id=row['sample_id'])


# ``LocalRefineEvalSet`` was removed: it was unreferenced dead code and carried
# its own copy of the buggy "resize then sniff dtype" ordering. Evaluation goes
# through ``local_refine_runtime.evaluate_conditions``, which is the single
# implementation shared by training-time validation and the standalone eval.
