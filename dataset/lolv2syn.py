"""
Dataset for lol-v2-synthetic (paired low-light images).
Structure:
    Train/Low/*.png     — low-light input
    Train/Normal/*.png  — normal-light GT  (same filenames)
    Test/Low/*.png
    Test/Normal/*.png
Identical to lolv2real but uses same-filename matching (not low/normal prefix).
"""
import os, glob, random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imageio import imread
import cv2

from dataset.data1 import apply_ref_gamma


class DegradeRef:
    """Apply color jitter, spatial shift, and Gaussian blur to reference."""
    def __init__(self, color_jitter=0.2, shift_range=4, blur_sigma=2.0):
        self.color_jitter = color_jitter
        self.shift_range = shift_range
        self.blur_sigma = blur_sigma

    def __call__(self, ref, seed=None):
        rng = np.random if seed is None else np.random.RandomState(int(seed))
        if self.color_jitter > 0:
            ref_f = ref.astype(np.float32) / 255.0
            b = 1.0 + rng.uniform(-self.color_jitter, self.color_jitter)
            ref_f = np.clip(ref_f * b, 0, 1)
            c = 1.0 + rng.uniform(-self.color_jitter, self.color_jitter)
            ref_f = np.clip((ref_f - 0.5) * c + 0.5, 0, 1)
            s = 1.0 + rng.uniform(-self.color_jitter, self.color_jitter)
            gray = ref_f.mean(axis=2, keepdims=True)
            ref_f = np.clip(gray + s * (ref_f - gray), 0, 1)
            ref = (ref_f * 255).astype(np.uint8)
        if self.shift_range > 0:
            dx = rng.randint(-self.shift_range, self.shift_range + 1)
            dy = rng.randint(-self.shift_range, self.shift_range + 1)
            if dx != 0 or dy != 0:
                h, w = ref.shape[:2]
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                ref = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if self.blur_sigma > 0:
            sigma = rng.uniform(0, self.blur_sigma)
            if sigma > 0.3:
                ksize = int(2 * np.ceil(3 * sigma) + 1)
                ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
                ref = cv2.GaussianBlur(ref, (ksize, ksize), sigmaX=sigma)
        return ref


def get_patch(lr_img, hr_img, ref_img, patch_size=128, scale=1):
    ih, iw = lr_img.shape[:2]
    ip = patch_size
    ix = random.randrange(0, iw - ip + 1) if (iw - ip + 1) > 0 else 0
    iy = random.randrange(0, ih - ip + 1) if (ih - ip + 1) > 0 else 0
    lr_patch = lr_img[iy:iy + ip, ix:ix + ip, :]
    hr_patch = hr_img[iy:iy + ip, ix:ix + ip, :]
    rh, rw = ref_img.shape[:2]
    if rh >= ip and rw >= ip:
        rx = random.randrange(0, rw - ip + 1) if (rw - ip + 1) > 0 else 0
        ry = random.randrange(0, rh - ip + 1) if (rh - ip + 1) > 0 else 0
        ref_patch = ref_img[ry:ry + ip, rx:rx + ip, :]
    else:
        ref_patch = ref_img
    return lr_patch, hr_patch, ref_patch


class RandomRotate:
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip:
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


class ToTensor:
    def __call__(self, sample):
        for k in ['LR', 'LR_sr', 'HR', 'Ref', 'Ref_sr']:
            sample[k] = torch.from_numpy(sample[k].transpose(2, 0, 1)).float()
        return sample


def _collect_pairs(base_dir):
    """Collect (low_path, normal_path) pairs — filenames are identical."""
    low_dir = os.path.join(base_dir, 'Low')
    normal_dir = os.path.join(base_dir, 'Normal')
    pairs = []
    for lf in sorted(glob.glob(os.path.join(low_dir, '*.png'))):
        base = os.path.basename(lf)
        nf = os.path.join(normal_dir, base)
        if os.path.exists(nf):
            pairs.append((lf, nf))
    return pairs


class TrainSet(Dataset):
    def __init__(self, args, transform=None):
        if transform is None:
            transform = transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])
        self.crop_size = getattr(args, 'train_crop_size', 128)
        self.ref_degrade = getattr(args, 'ref_degrade', False)
        if self.ref_degrade:
            self.degrader = DegradeRef(
                color_jitter=getattr(args, 'ref_color_jitter', 0.2),
                shift_range=getattr(args, 'ref_shift_range', 4),
                blur_sigma=getattr(args, 'ref_blur_sigma', 2.0))
        self.transform = transform
        self.pairs = _collect_pairs(os.path.join(args.dataset_dir, 'Train'))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, normal_path = self.pairs[idx]
        LR = imread(low_path)
        HR = imread(normal_path)
        h_lr, w_lr = LR.shape[:2]
        h_hr, w_hr = HR.shape[:2]
        h, w = min(h_lr, h_hr), min(w_lr, w_hr)
        LR = LR[:h, :w, :]
        HR = HR[:h, :w, :]
        Ref = HR.copy()

        if h >= self.crop_size and w >= self.crop_size:
            LR, HR, Ref = get_patch(LR, HR, Ref, patch_size=self.crop_size, scale=1)
        else:
            LR = np.pad(LR, ((0, max(0, self.crop_size - h)), (0, max(0, self.crop_size - w)), (0, 0)), mode='reflect')
            HR = np.pad(HR, ((0, max(0, self.crop_size - h)), (0, max(0, self.crop_size - w)), (0, 0)), mode='reflect')
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

        sample = {'LR': LR, 'LR_sr': LR_sr, 'HR': HR, 'Ref': Ref, 'Ref_sr': Ref_sr}
        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()]),
                 max_size=960):
        self.transform = transform
        self.max_size = max_size
        self.ref_degrade = getattr(args, 'ref_degrade', False)
        self.ref_gamma = getattr(args, 'ref_gamma', 1.0)
        self.eval_degrade_seed = getattr(args, 'eval_degrade_seed', 1234)
        if self.ref_degrade:
            self.degrader = DegradeRef(
                color_jitter=getattr(args, 'ref_color_jitter', 0.2),
                shift_range=getattr(args, 'ref_shift_range', 4),
                blur_sigma=getattr(args, 'ref_blur_sigma', 2.0))
        self.pairs = _collect_pairs(os.path.join(args.dataset_dir, 'Test'))

        # Support external reference images (e.g. nanobanana refs).
        # Files are matched by basename to the low input (e.g. r00816405t.png).
        self.ref_dir = getattr(args, 'ref_dir', '')
        self.ref_map = {}
        if self.ref_dir and os.path.isdir(self.ref_dir):
            for f in sorted(glob.glob(os.path.join(self.ref_dir, '*.png'))):
                self.ref_map[os.path.basename(f)] = f
            for f in sorted(glob.glob(os.path.join(self.ref_dir, '*.jpg'))):
                self.ref_map.setdefault(os.path.basename(f), f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, normal_path = self.pairs[idx]
        LR = imread(low_path)
        HR = imread(normal_path)
        h, w = LR.shape[:2]
        if self.max_size and max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale) // 4 * 4, int(w * scale) // 4 * 4
            LR = cv2.resize(LR, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            HR = cv2.resize(HR, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        h, w = h // 4 * 4, w // 4 * 4
        LR, HR = LR[:h, :w, :], HR[:h, :w, :]

        # Load external reference image if available, otherwise use HR as Ref
        ref_path = self.ref_map.get(os.path.basename(low_path))
        if ref_path and os.path.exists(ref_path):
            Ref = imread(ref_path)
            # Handle RGBA → RGB
            if Ref.ndim == 3 and Ref.shape[2] == 4:
                Ref = Ref[:, :, :3]
            # Resize external ref to match LR spatial size
            rh, rw = Ref.shape[:2]
            if rh != h or rw != w:
                Ref = cv2.resize(Ref, (w, h), interpolation=cv2.INTER_LINEAR)
            Ref = apply_ref_gamma(Ref, self.ref_gamma)
        else:
            Ref = HR.copy()
        LR_sr, Ref_sr = LR.copy(), Ref.copy()

        if self.ref_degrade:
            Ref = self.degrader(Ref, seed=self.eval_degrade_seed + idx)
            Ref_sr = Ref.copy()

        LR = LR.astype(np.float32) / 127.5 - 1.
        LR_sr = LR_sr.astype(np.float32) / 127.5 - 1.
        HR = HR.astype(np.float32) / 127.5 - 1.
        Ref = Ref.astype(np.float32) / 127.5 - 1.
        Ref_sr = Ref_sr.astype(np.float32) / 127.5 - 1.

        sample = {'LR': LR, 'LR_sr': LR_sr, 'HR': HR, 'Ref': Ref, 'Ref_sr': Ref_sr}
        if self.transform:
            sample = self.transform(sample)
        return sample
