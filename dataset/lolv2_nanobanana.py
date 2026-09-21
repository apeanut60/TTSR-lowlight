"""lolv2 nanobanana training dataset.

Uses the nanobanana-generated references stored inside each lolv2 training tree:

    Train/Low/{name}.png
    Train/Normal/{name}.png
    Train/nanobanana_ref/{name}.png

For lolv2real, low names are ``low00001.png`` and normal names are
``normal00001.png``; for lolv2syn, low and normal names are identical.

Reference mapping is done by basename after the appropriate low->normal
translation, so it is not index-based.
"""

import os

import cv2
import numpy as np
from imageio import imread
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.data1 import (DegradeRef, RandomFlip, RandomRotate, ToTensor,
                           get_patch, apply_ref_gamma)
from dataset import lolv2real, lolv2syn


def _normal_name(low_name, subset):
    if subset == 'real':
        return low_name.replace('low', 'normal')
    return low_name


def _collect_triplets(base_dir, subset, ref_subdir='nanobanana_ref'):
    low_dir = os.path.join(base_dir, 'Low')
    normal_dir = os.path.join(base_dir, 'Normal')
    ref_dir = os.path.join(base_dir, ref_subdir)

    triplets = []
    for low_name in sorted(os.listdir(low_dir)):
        if not low_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        normal_name = _normal_name(low_name, subset)
        low_path = os.path.join(low_dir, low_name)
        normal_path = os.path.join(normal_dir, normal_name)
        ref_path = os.path.join(ref_dir, low_name)
        if os.path.isfile(normal_path) and os.path.isfile(ref_path):
            triplets.append((low_path, normal_path, ref_path))
    return triplets


class TrainSet(Dataset):
    def __init__(self, args, transform=None):
        if transform is None:
            transform = transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])
        self.subset = getattr(args, 'lolv2_nanobanana_subset', 'real')
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
            os.path.join(args.dataset_dir, 'Train'), self.subset,
            ref_subdir=getattr(args, 'lolv2_nanobanana_ref_subdir',
                               'nanobanana_ref'))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        low_path, normal_path, ref_path = self.triplets[idx]
        LR = imread(low_path)
        HR = imread(normal_path)
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


class TestSet:
    """Dispatch eval to the matching lolv2 test set."""

    def __new__(cls, args, ref_level='1', transform=None, **kwargs):
        subset = getattr(args, 'lolv2_nanobanana_subset', 'real')
        if subset == 'real':
            return lolv2real.TestSet(args=args, ref_level=ref_level)
        return lolv2syn.TestSet(args=args, ref_level=ref_level)
