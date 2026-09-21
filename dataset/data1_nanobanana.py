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
        self.crop_size = getattr(args, 'train_crop_size', 128)
        self.ref_degrade = getattr(args, 'ref_degrade', False)
        self.ref_gamma = getattr(args, 'ref_gamma', 1.0)
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

        if h >= self.crop_size and w >= self.crop_size:
            LR, HR, Ref = get_patch(LR, HR, Ref, patch_size=self.crop_size)
        else:
            LR = np.pad(
                LR,
                ((0, max(0, self.crop_size - h)), (0, max(0, self.crop_size - w)), (0, 0)),
                mode='reflect')
            HR = np.pad(
                HR,
                ((0, max(0, self.crop_size - h)), (0, max(0, self.crop_size - w)), (0, 0)),
                mode='reflect')
            Ref = HR.copy()

        LR_sr, Ref_sr = LR.copy(), Ref.copy()

        if self.ref_degrade:
            Ref = self.degrader(Ref)
            Ref_sr = Ref.copy()

        LR = LR.astype(np.float32) / 127.5 - 1.
        LR_sr = LR_sr.astype(np.float32) / 127.5 - 1.
        HR = HR.astype(np.float32) / 127.5 - 1.
        Ref = Ref.astype(np.float32) / 127.5 - 1.
        Ref_sr = Ref_sr.astype(np.float32) / 127.5 - 1.

        sample = {'LR': LR, 'LR_sr': LR_sr, 'HR': HR,
                  'Ref': Ref, 'Ref_sr': Ref_sr}
        if self.transform:
            sample = self.transform(sample)
        return sample


# Evaluation for this dataset can reuse data1.TestSet, which already supports
# an external --ref_dir for nanobanana-style references.
from dataset.data1 import TestSet
