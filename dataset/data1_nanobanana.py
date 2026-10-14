"""data1_nanobanana dataset.

Uses the nanobanana-generated references stored inside the data1 training tree:

    Training data/{camera}/low/*.jpg
    Training data/{camera}/high/*.jpg
    Training data/{camera}/nanobanana_ref/*.jpg

The reference is mapped to each low input by exact basename, so it does not
depend on sorted-list index alignment and will never use the unrelated
`data2/ref` directory.
"""

import os

import cv2
import numpy as np
import torch
from imageio import imread
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.data1 import (DegradeRef, RandomFlip, RandomRotate, ToTensor,
                           get_patch, apply_ref_gamma)


def _collect_triplets(base_dir, camera_filter='all', ref_subdir='nanobanana_ref',
                      manifest_dir=''):
    """Collect (low, high, nanobanana_ref) triplets by identical basename."""
    triplets = []
    for camera in sorted(os.listdir(base_dir)):
        if camera_filter != 'all' and camera != camera_filter:
            continue
        cam_dir = os.path.join(base_dir, camera)
        if not os.path.isdir(cam_dir):
            continue
        low_dir = os.path.join(cam_dir, 'low')
        high_dir = os.path.join(cam_dir, 'high')
        ref_dir = os.path.join(cam_dir, ref_subdir)
        if not all(os.path.isdir(d) for d in (low_dir, high_dir, ref_dir)):
            continue

        allowed_basenames = None
        manifest_path = os.path.join(manifest_dir, f'{camera}.txt') if manifest_dir else ''
        if manifest_path and os.path.isfile(manifest_path):
            allowed_basenames = set()
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        allowed_basenames.add(os.path.basename(line))

        for low_path in sorted(
            p for p in os.listdir(low_dir)
            if p.lower().endswith(('.jpg', '.jpeg', '.png'))
        ):
            if allowed_basenames is not None and low_path not in allowed_basenames:
                continue
            high_path = os.path.join(high_dir, low_path)
            ref_path = os.path.join(ref_dir, low_path)
            if os.path.isfile(high_path) and os.path.isfile(ref_path):
                triplets.append((os.path.join(low_dir, low_path),
                                 high_path, ref_path))
    return triplets


class TrainSet(Dataset):
    """Training set using nanobanana_ref as Ref."""

    def __init__(self, args, transform=None):
        if transform is None:
            transform = transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])
        self.args = args
        self.crop_size = getattr(args, 'train_crop_size', 128)
        self.ref_degrade = getattr(args, 'ref_degrade', False)
        self.ref_gamma = getattr(args, 'ref_gamma', 1.0)
        self.ref_photo_aug = getattr(args, 'ref_photo_aug', False)
        self.ref_photo_aug_range = float(getattr(args, 'ref_photo_aug_range', 0.05))
        # Full-image query gamma is always computed for data1_nanobanana so FT
        # can pass query_gamma_override. Unused when query_illum_align=False.
        self.compute_query_gamma = True
        if self.ref_degrade:
            self.degrader = DegradeRef(
                color_jitter=getattr(args, 'ref_color_jitter', 0.2),
                shift_range=getattr(args, 'ref_shift_range', 4),
                blur_sigma=getattr(args, 'ref_blur_sigma', 2.0))
        self.transform = transform
        self.triplets = _collect_triplets(
            os.path.join(args.dataset_dir, 'Training data'),
            camera_filter=getattr(args, 'data1_camera', 'all'),
            ref_subdir=getattr(args, 'nanobanana_ref_subdir', 'nanobanana_ref'),
            manifest_dir=getattr(args, 'nanobanana_manifest_dir', ''))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        low_path, high_path, ref_path = self.triplets[idx]
        LR = imread(low_path)
        HR = imread(high_path)
        Ref = imread(ref_path)

        h_lr, w_lr = LR.shape[:2]
        h_hr, w_hr = HR.shape[:2]
        h, w = min(h_lr, h_hr), min(w_lr, w_hr)
        LR = LR[:h, :w, :]
        HR = HR[:h, :w, :]

        h_ref, w_ref = Ref.shape[:2]
        if h_ref != h or w_ref != w:
            Ref = cv2.resize(Ref, (w, h), interpolation=cv2.INTER_LINEAR)
        Ref = apply_ref_gamma(Ref, self.ref_gamma)

        # v1r6/v1r7: full-image Gamma (+ hist LUT) from Low + Nano Ref (never GT)
        # BEFORE random crop. Applied later to the Low Query crop only.
        # Independent Ref crop / geometric aug below is intentionally unchanged.
        query_gamma, query_hist_lut = self._full_image_query_maps(LR, Ref)

        if h >= self.crop_size and w >= self.crop_size:
            LR, HR, Ref = get_patch(LR, HR, Ref, patch_size=self.crop_size)
        else:
            # Pad all three; never fall back to HR as Ref (nano integrity).
            pad = ((0, max(0, self.crop_size - h)),
                   (0, max(0, self.crop_size - w)), (0, 0))
            LR = np.pad(LR, pad, mode='reflect')
            HR = np.pad(HR, pad, mode='reflect')
            Ref = np.pad(Ref, pad, mode='reflect')

        LR_sr, Ref_sr = LR.copy(), Ref.copy()

        # Ref-only photometric jitter (does not touch Low/GT).
        if self.ref_photo_aug and self.ref_photo_aug_range > 0:
            r = self.ref_photo_aug_range
            ref_f = Ref.astype(np.float32) / 255.0
            b = 1.0 + np.random.uniform(-r, r)
            ref_f = np.clip(ref_f * b, 0.0, 1.0)
            s = 1.0 + np.random.uniform(-r, r)
            gray = ref_f.mean(axis=2, keepdims=True)
            ref_f = np.clip(gray + s * (ref_f - gray), 0.0, 1.0)
            Ref = (ref_f * 255.0 + 0.5).astype(np.uint8)
            Ref_sr = Ref.copy()

        if self.ref_degrade:
            Ref = self.degrader(Ref)
            Ref_sr = Ref.copy()

        LR = LR.astype(np.float32) / 127.5 - 1.
        LR_sr = LR_sr.astype(np.float32) / 127.5 - 1.
        HR = HR.astype(np.float32) / 127.5 - 1.
        Ref = Ref.astype(np.float32) / 127.5 - 1.
        Ref_sr = Ref_sr.astype(np.float32) / 127.5 - 1.

        sample = {'LR': LR, 'LR_sr': LR_sr, 'HR': HR,
                  'Ref': Ref, 'Ref_sr': Ref_sr,
                  'query_gamma': query_gamma,
                  'query_hist_lut': query_hist_lut}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _full_image_query_maps(self, lr_u8, ref_u8):
        """Full-image gamma + hist LUT (CPU) from Low/Nano Ref uint8."""
        from model.query_illum_align import estimate_query_align_maps
        args = getattr(self, 'args', None)
        gmin = float(getattr(args, 'query_gamma_min', 0.3)) if args else 0.3
        gmax = float(getattr(args, 'query_gamma_max', 1.0)) if args else 1.0
        alpha = float(getattr(args, 'query_illum_alpha', 0.75)) if args else 0.75
        bins = int(getattr(args, 'query_hist_bins', 256)) if args else 256
        # If alignment disabled, still compute gamma with alpha for LUT identity;
        # trainer only passes overrides when align is on.
        if args is not None and not getattr(args, 'query_illum_align', False):
            alpha = 0.0
        low = torch.from_numpy(
            lr_u8.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.).unsqueeze(0)
        ref = torch.from_numpy(
            ref_u8.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.).unsqueeze(0)
        with torch.no_grad():
            g, lut, _ = estimate_query_align_maps(
                low, ref, alpha=alpha, gamma_min=gmin, gamma_max=gmax, bins=bins)
        return float(g.view(-1)[0].item()), lut.view(-1).cpu().numpy().astype(np.float32)


# Evaluation for this dataset can reuse data1.TestSet, which already supports
# an external --ref_dir for nanobanana-style references.
from dataset.data1 import TestSet
