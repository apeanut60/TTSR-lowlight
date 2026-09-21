"""
Dataset for SICE (Smartphone Image Color Enhancement).
Structure:
    Dataset_Part1/Lowlight_img/*.jpg      — low-light multi-exposure inputs
    Dataset_Part1/Lowlight_img_Label/*.jpg — expert-retouched GT
    Dataset_Part2/Lowlight_img/*.jpg
    Dataset_Part2/Lowlight_img_Label/*.jpg

Pairing: low image "scene_exposure.jpg" → GT label "scene.jpg"
E.g., "100_1.jpg" → "100.jpg", "296_9.jpg" → "296.jpg"
"""
import os, glob, random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imageio import imread
import cv2


class DegradeRef:
    """Apply color jitter, spatial shift, and Gaussian blur to reference."""
    def __init__(self, color_jitter=0.2, shift_range=4, blur_sigma=2.0):
        self.color_jitter = color_jitter
        self.shift_range = shift_range
        self.blur_sigma = blur_sigma

    def __call__(self, ref):
        if self.color_jitter > 0:
            ref_f = ref.astype(np.float32) / 255.0
            b = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            ref_f = np.clip(ref_f * b, 0, 1)
            c = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            ref_f = np.clip((ref_f - 0.5) * c + 0.5, 0, 1)
            s = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            gray = ref_f.mean(axis=2, keepdims=True)
            ref_f = np.clip(gray + s * (ref_f - gray), 0, 1)
            ref = (ref_f * 255).astype(np.uint8)
        if self.shift_range > 0:
            dx = np.random.randint(-self.shift_range, self.shift_range + 1)
            dy = np.random.randint(-self.shift_range, self.shift_range + 1)
            if dx != 0 or dy != 0:
                h, w = ref.shape[:2]
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                ref = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if self.blur_sigma > 0:
            sigma = np.random.uniform(0, self.blur_sigma)
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


def _scene_id(filename):
    """Extract scene ID: '100_1.jpg' -> '100'"""
    base = os.path.splitext(filename)[0]
    if '_' in base:
        return base.split('_')[0]
    return base


def _collect_pairs(base_dir):
    """Collect (low_path, label_path) pairs.
    Matches low image 'scene_exposure.jpg' to label 'scene.jpg'.
    """
    low_dir = os.path.join(base_dir, 'Lowlight_img')
    label_dir = os.path.join(base_dir, 'Lowlight_img_Label')

    # Build label lookup by scene ID
    label_map = {}
    for f in sorted(os.listdir(label_dir)):
        if f.lower().endswith(('.jpg', '.png')):
            sid = _scene_id(f)
            label_map[sid] = os.path.join(label_dir, f)

    pairs = []
    for f in sorted(os.listdir(low_dir)):
        if not f.lower().endswith(('.jpg', '.png')):
            continue
        sid = _scene_id(f)
        if sid in label_map:
            pairs.append((os.path.join(low_dir, f), label_map[sid]))

    return pairs


class TrainSet(Dataset):
    """SICE has no official train split — use all as test."""
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
        self.pairs = []

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError('SICE is evaluation-only')


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()]),
                 max_size=960):
        self.transform = transform
        self.max_size = max_size
        self.ref_degrade = getattr(args, 'ref_degrade', False)
        if self.ref_degrade:
            self.degrader = DegradeRef(
                color_jitter=getattr(args, 'ref_color_jitter', 0.2),
                shift_range=getattr(args, 'ref_shift_range', 4),
                blur_sigma=getattr(args, 'ref_blur_sigma', 2.0))

        # Collect pairs from both Part1 and Part2
        dataset_dir = args.dataset_dir
        self.pairs = []
        if os.path.isdir(os.path.join(dataset_dir, 'Dataset_Part1')):
            self.pairs += _collect_pairs(os.path.join(dataset_dir, 'Dataset_Part1'))
        if os.path.isdir(os.path.join(dataset_dir, 'Dataset_Part2')):
            self.pairs += _collect_pairs(os.path.join(dataset_dir, 'Dataset_Part2'))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, label_path = self.pairs[idx]
        LR = imread(low_path)
        HR = imread(label_path)
        h, w = LR.shape[:2]
        if self.max_size and max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale) // 4 * 4, int(w * scale) // 4 * 4
            LR = cv2.resize(LR, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            HR = cv2.resize(HR, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        h, w = h // 4 * 4, w // 4 * 4
        LR, HR = LR[:h, :w, :], HR[:h, :w, :]
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
