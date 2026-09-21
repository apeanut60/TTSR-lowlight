"""
LOL (LOw-Light) dataset for TTSR-lowlight enhancement.
Dataset structure:
    our485/low/  - low-light input images (training)
    our485/high/ - normal-light GT images (training)
    eval15/low/  - low-light input images (test)
    eval15/high/ - normal-light GT images (test)
Files are 1:1 paired by numeric filenames (e.g., 1.png, 100.png).
"""

import os
import numpy as np
import glob
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from imageio import imread
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
import cv2

from dataset.data1 import apply_ref_gamma


def get_patch(lr_img, hr_img, ref_img, patch_size=128, scale=1):
    """Random crop for training. scale=1 for 1:1 enhancement."""
    ih, iw = lr_img.shape[:2]
    ip = patch_size
    tp = patch_size  # target patch same size since scale=1

    ix = random.randrange(0, iw - ip + 1) if (iw - ip + 1) > 0 else 0
    iy = random.randrange(0, ih - ip + 1) if (ih - ip + 1) > 0 else 0
    tx, ty = ix, iy  # same position since 1:1

    lr_patch = lr_img[iy:iy + ip, ix:ix + ip, :]
    hr_patch = hr_img[ty:ty + tp, tx:tx + tp, :]

    # ref: also crop from the ref image (normal-light version)
    rh, rw = ref_img.shape[:2]
    if rh >= tp and rw >= tp:
        rx = random.randrange(0, rw - tp + 1) if (rw - tp + 1) > 0 else 0
        ry = random.randrange(0, rh - tp + 1) if (rh - tp + 1) > 0 else 0
        ref_patch = ref_img[ry:ry + tp, rx:rx + tp, :]
    else:
        ref_patch = ref_img

    return lr_patch, hr_patch, ref_patch


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if np.random.randint(0, 2) == 1:
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if np.random.randint(0, 2) == 1:
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if np.random.randint(0, 2) == 1:
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if np.random.randint(0, 2) == 1:
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample




class DegradeRef(object):
    """Apply realistic degradations to the reference image to simulate
    real-world mismatches between the low-light input and the reference.
    Works on uint8 [0, 255] numpy arrays before ToTensor.
    """
    def __init__(self, color_jitter=0.2, shift_range=4, blur_sigma=2.0):
        self.color_jitter = color_jitter
        self.shift_range = shift_range
        self.blur_sigma = blur_sigma

    def __call__(self, sample, seed=None):
        rng = np.random if seed is None else np.random.RandomState(int(seed))
        ref = sample['Ref']        # uint8, (H, W, 3)
        ref_sr = sample['Ref_sr']

        # ── 1. Color jitter ──
        if self.color_jitter > 0:
            ref_f = ref.astype(np.float32) / 255.0
            brightness = 1.0 + rng.uniform(-self.color_jitter, self.color_jitter)
            ref_f = np.clip(ref_f * brightness, 0, 1)
            contrast = 1.0 + rng.uniform(-self.color_jitter, self.color_jitter)
            ref_f = np.clip((ref_f - 0.5) * contrast + 0.5, 0, 1)
            saturation = 1.0 + rng.uniform(-self.color_jitter, self.color_jitter)
            gray = ref_f.mean(axis=2, keepdims=True)
            ref_f = np.clip(gray + saturation * (ref_f - gray), 0, 1)
            ref = (ref_f * 255).astype(np.uint8)

        # ── 2. Spatial shift ──
        if self.shift_range > 0:
            dx = rng.randint(-self.shift_range, self.shift_range + 1)
            dy = rng.randint(-self.shift_range, self.shift_range + 1)
            if dx != 0 or dy != 0:
                h, w = ref.shape[:2]
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                ref = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # ── 3. Gaussian blur ──
        if self.blur_sigma > 0:
            sigma = rng.uniform(0, self.blur_sigma)
            if sigma > 0.3:
                ksize = int(2 * np.ceil(3 * sigma) + 1)
                ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
                ref = cv2.GaussianBlur(ref, (ksize, ksize), sigmaX=sigma)

        sample['Ref'] = ref
        sample['Ref_sr'] = ref.copy()
        return sample
class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = (
            sample['LR'], sample['LR_sr'], sample['HR'],
            sample['Ref'], sample['Ref_sr']
        )
        LR = LR.transpose((2, 0, 1))
        LR_sr = LR_sr.transpose((2, 0, 1))
        HR = HR.transpose((2, 0, 1))
        Ref = Ref.transpose((2, 0, 1))
        Ref_sr = Ref_sr.transpose((2, 0, 1))
        return {
            'LR': torch.from_numpy(LR).float(),
            'LR_sr': torch.from_numpy(LR_sr).float(),
            'HR': torch.from_numpy(HR).float(),
            'Ref': torch.from_numpy(Ref).float(),
            'Ref_sr': torch.from_numpy(Ref_sr).float(),
        }


class TrainSet(Dataset):
    """LOL training dataset. Low-light as input, normal-light as both GT and Ref."""

    def __init__(self, args, transform=None):
        if transform is None:
            transform = transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])
        self.dataset_dir = args.dataset_dir
        self.crop_size = getattr(args, 'train_crop_size', 128)
        self.ref_degrade = getattr(args, 'ref_degrade', False)
        if self.ref_degrade:
            self.degrader = DegradeRef(
                color_jitter=getattr(args, 'ref_color_jitter', 0.2),
                shift_range=getattr(args, 'ref_shift_range', 4),
                blur_sigma=getattr(args, 'ref_blur_sigma', 2.0))
        self.transform = transform

        low_dir = os.path.join(self.dataset_dir, 'our485', 'low')
        high_dir = os.path.join(self.dataset_dir, 'our485', 'high')

        self.low_list = sorted([
            os.path.join(low_dir, name)
            for name in os.listdir(low_dir)
            if name.endswith('.png')
        ])
        self.high_list = sorted([
            os.path.join(high_dir, name)
            for name in os.listdir(high_dir)
            if name.endswith('.png')
        ])
        assert len(self.low_list) == len(self.high_list), \
            f"low({len(self.low_list)}) and high({len(self.high_list)}) mismatch!"

    def __len__(self):
        return len(self.low_list)

    def __getitem__(self, idx):
        # LR = low-light input, HR = normal-light GT, Ref = normal-light reference
        LR = imread(self.low_list[idx])
        HR = imread(self.high_list[idx])

        h_lr, w_lr = LR.shape[:2]
        h_hr, w_hr = HR.shape[:2]
        # ensure same size
        h, w = min(h_lr, h_hr), min(w_lr, h_hr)
        LR = LR[:h, :w, :]
        HR = HR[:h, :w, :]

        # Ref: use the same normal-light image as reference
        Ref = HR.copy()

        # Random crop for training
        if h >= self.crop_size and w >= self.crop_size:
            LR, HR, Ref = get_patch(LR, HR, Ref, patch_size=self.crop_size, scale=1)
        else:
            # pad if smaller than crop_size
            LR = np.pad(LR, ((0, max(0, self.crop_size - h)),
                             (0, max(0, self.crop_size - w)), (0, 0)),
                        mode='reflect')
            HR = np.pad(HR, ((0, max(0, self.crop_size - h)),
                             (0, max(0, self.crop_size - w)), (0, 0)),
                        mode='reflect')
            Ref = HR.copy()

        # 1:1 enhancement — LR_sr = LR, Ref_sr = Ref (same resolution, no upscaling)
        LR_sr = LR.copy()
        Ref_sr = Ref.copy()

        # Apply degradation to reference (v2: simulate real-world mismatch)
        # Must be done BEFORE float32 conversion and normalization (Ref is uint8 [0,255])
        if hasattr(self, 'ref_degrade') and self.ref_degrade:
            sample_tmp = {'Ref': Ref, 'Ref_sr': Ref_sr}
            sample_tmp = self.degrader(sample_tmp)
            Ref = sample_tmp['Ref']
            Ref_sr = sample_tmp['Ref_sr']

        # Change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # RGB range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {
            'LR': LR,
            'LR_sr': LR_sr,
            'HR': HR,
            'Ref': Ref,
            'Ref_sr': Ref_sr,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    """LOL test dataset. 15 paired images for evaluation."""

    def __init__(self, args, ref_level='1',
                 transform=transforms.Compose([ToTensor()])):
        self.dataset_dir = args.dataset_dir
        self.transform = transform
        self.ref_degrade = getattr(args, 'ref_degrade', False)
        self.ref_gamma = getattr(args, 'ref_gamma', 1.0)
        self.eval_degrade_seed = getattr(args, 'eval_degrade_seed', 1234)
        if self.ref_degrade:
            self.degrader = DegradeRef(
                color_jitter=getattr(args, 'ref_color_jitter', 0.2),
                shift_range=getattr(args, 'ref_shift_range', 4),
                blur_sigma=getattr(args, 'ref_blur_sigma', 2.0))

        low_dir = os.path.join(self.dataset_dir, 'eval15', 'low')
        high_dir = os.path.join(self.dataset_dir, 'eval15', 'high')

        self.low_list = sorted([
            os.path.join(low_dir, name)
            for name in os.listdir(low_dir)
            if name.endswith('.png')
        ])
        self.high_list = sorted([
            os.path.join(high_dir, name)
            for name in os.listdir(high_dir)
            if name.endswith('.png')
        ])
        assert len(self.low_list) == len(self.high_list), \
            f"low({len(self.low_list)}) and high({len(self.high_list)}) mismatch!"

        # Support external reference images (e.g. nanobanana-generated refs)
        self.ref_dir = getattr(args, 'ref_dir', '')
        self.ref_files = []
        if self.ref_dir and os.path.isdir(self.ref_dir):
            self.ref_files = sorted(glob.glob(os.path.join(self.ref_dir, '*.png')))
            if len(self.ref_files) == 0:
                self.ref_files = sorted(glob.glob(os.path.join(self.ref_dir, '*.jpg')))

    def __len__(self):
        return len(self.low_list)

    def __getitem__(self, idx):
        # LR = low-light input, HR = normal-light GT
        LR = imread(self.low_list[idx])
        HR = imread(self.high_list[idx])

        h, w = LR.shape[:2]
        # crop to multiple of 4 (for network compatibility)
        h, w = h // 4 * 4, w // 4 * 4
        LR = LR[:h, :w, :]
        HR = HR[:h, :w, :]

        # Load external reference image if available, otherwise use HR as Ref
        if self.ref_files and idx < len(self.ref_files):
            Ref = imread(self.ref_files[idx])
            # Handle RGBA → RGB
            if Ref.ndim == 3 and Ref.shape[2] == 4:
                Ref = Ref[:, :, :3]
            # Resize external ref to match LR spatial size
            rh, rw = Ref.shape[:2]
            if rh != h or rw != w:
                Ref = cv2.resize(Ref, (w, h), interpolation=cv2.INTER_LINEAR)
            # Gamma correction applies to external (e.g. nanobanana) refs only
            Ref = apply_ref_gamma(Ref, self.ref_gamma)
        else:
            Ref = HR.copy()
        h2, w2 = Ref.shape[:2]

        # 1:1 enhancement — no bicubic up/down-sampling
        LR_sr = LR.copy()
        Ref_sr = Ref.copy()

        # Apply degradation to reference (matching training config)
        if hasattr(self, 'ref_degrade') and self.ref_degrade:
            sample_tmp = {'Ref': Ref, 'Ref_sr': Ref_sr}
            sample_tmp = self.degrader(
                sample_tmp, seed=self.eval_degrade_seed + idx)
            Ref = sample_tmp['Ref']
            Ref_sr = sample_tmp['Ref_sr']

        # Change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # RGB range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {
            'LR': LR,
            'LR_sr': LR_sr,
            'HR': HR,
            'Ref': Ref,
            'Ref_sr': Ref_sr,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
