"""Shared constants / helpers for V6 LLFormer LOLv2-Real Base."""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from model.LLFormerBridge import (BRIDGE_VERSION, D1_CHANNELS, D1_SCALE,  # noqa: E402
                                  D1_TENSOR_NAME, D2_CHANNELS, D2_SCALE,
                                  D2_TENSOR_NAME, LOL_ARCH, OFFICIAL_LOL_CKPT,
                                  PAD_MULTIPLE)
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402

ROOT = '/root/data/experiments/llformer_lolv2real'
DATA_DIR = '/root/data/datasets/lol-v2-real'
HIST_SPLIT = '/root/data/experiments/v3a4_lolv2real/splits/split.json'
STAGE = 'LLFORMER_LOLV2REAL_BASE'

SEED = 1234
PATCH = 128
BATCH = 8
EPOCHS = 1000
LR_INITIAL = 2e-5
LR_MIN = 1e-6
WARMUP_EPOCHS = 5
VAL_EVERY = 10
SAVE_EVERY = 25
LOSS_NAME = 'SmoothL1Loss'
OPTIMIZER_NAME = 'Adam'

# Official Test anchor (RetinexFormer LOL_v2_real)
RETINEX_TEST_PSNR = 20.707
GO_PSNR = 21.2
STRONG_GO_PSNR = 22.0


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_train_lows(data_dir: str = DATA_DIR) -> List[str]:
    low_dir = os.path.join(data_dir, 'Train', 'Low')
    names = sorted(
        f for f in os.listdir(low_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('low'))
    return names


def low_to_normal_name(low_name: str) -> str:
    """LOLv2-Real: low00001.png → normal00001.png."""
    if low_name.startswith('low'):
        return 'normal' + low_name[3:]
    return low_name


def build_train625_dev64(hist_split: str = HIST_SPLIT,
                         data_dir: str = DATA_DIR
                         ) -> Tuple[List[str], List[str]]:
    with open(hist_split) as f:
        hist = json.load(f)
    dev = list(hist['dev'])
    all_lows = list_train_lows(data_dir)
    all_set = set(all_lows)
    for n in dev:
        if n not in all_set:
            raise SystemExit('historical dev name missing from Train/Low: %s' % n)
    train = [n for n in all_lows if n not in set(dev)]
    if len(dev) != 64 or len(train) != 625:
        raise SystemExit('expected train625/dev64 got %d/%d'
                         % (len(train), len(dev)))
    if set(train) & set(dev):
        raise SystemExit('train/dev intersection non-empty')
    if set(train) | set(dev) != all_set:
        raise SystemExit('train∪dev != all 689')
    for n in all_lows:
        hi = os.path.join(data_dir, 'Train', 'Normal', low_to_normal_name(n))
        if not os.path.isfile(hi):
            raise SystemExit('missing Normal pair for %s' % n)
    return train, dev


def write_split_txt(path: str, names: Sequence[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for n in names:
            f.write(n + '\n')


def read_split_txt(path: str) -> List[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_rgb01(path: str) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def paired_augment(low: torch.Tensor, high: torch.Tensor, patch: int = PATCH):
    """Same crop + same flip/rot90 on both. Returns [3,patch,patch]."""
    _, h, w = low.shape
    if h < patch or w < patch:
        # reflect pad then crop
        pad_h = max(0, patch - h)
        pad_w = max(0, patch - w)
        low = F.pad(low[None], (0, pad_w, 0, pad_h), mode='reflect')[0]
        high = F.pad(high[None], (0, pad_w, 0, pad_h), mode='reflect')[0]
        _, h, w = low.shape
    y0 = random.randint(0, h - patch)
    x0 = random.randint(0, w - patch)
    low = low[:, y0:y0 + patch, x0:x0 + patch]
    high = high[:, y0:y0 + patch, x0:x0 + patch]
    if random.random() < 0.5:
        low = torch.flip(low, dims=[-1])
        high = torch.flip(high, dims=[-1])
    if random.random() < 0.5:
        low = torch.flip(low, dims=[-2])
        high = torch.flip(high, dims=[-2])
    k = random.randint(0, 3)
    if k:
        low = torch.rot90(low, k, dims=[-2, -1])
        high = torch.rot90(high, k, dims=[-2, -1])
    return low.contiguous(), high.contiguous()


class LOLv2PairDataset(torch.utils.data.Dataset):
    """Low/Normal pairs in [0,1]. Train: random patch+aug. Eval: full image."""

    def __init__(self, names: Sequence[str], data_dir: str = DATA_DIR,
                 split: str = 'Train', train: bool = True, patch: int = PATCH):
        self.names = list(names)
        self.data_dir = data_dir
        self.split = split  # 'Train' or 'Test'
        self.train = train
        self.patch = patch
        self.low_dir = os.path.join(data_dir, split, 'Low')
        self.high_dir = os.path.join(data_dir, split, 'Normal')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        high_name = low_to_normal_name(name)
        low = load_rgb01(os.path.join(self.low_dir, name))
        high = load_rgb01(os.path.join(self.high_dir, high_name))
        if self.train:
            low, high = paired_augment(low, high, self.patch)
        return dict(name=name, low=low, high=high)


def metrics_01(pred01: torch.Tensor, gt01: torch.Tensor):
    """PSNR/SSIM via project helper (expects [-1,1])."""
    from local_refine_runtime import metrics as _metrics
    p = pred01.clamp(0, 1) * 2 - 1
    g = gt01.clamp(0, 1) * 2 - 1
    if p.dim() == 3:
        p, g = p[None], g[None]
    psnr, ssim, mse = _metrics(p, g)
    return float(psnr), float(ssim), float(mse)


def cosine_lr(epoch: int, epochs: int = EPOCHS, warmup: int = WARMUP_EPOCHS,
              lr0: float = LR_INITIAL, lr_min: float = LR_MIN) -> float:
    if epoch < warmup:
        return lr0 * float(epoch + 1) / float(warmup)
    t = (epoch - warmup) / max(1, epochs - warmup)
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + np.cos(np.pi * t))


def lock_architecture_fields() -> Dict:
    return dict(
        stage=STAGE,
        bridge_version=BRIDGE_VERSION,
        model_arch='LLFormerBridge',
        model_config=LOL_ARCH,
        d2_tensor_name=D2_TENSOR_NAME,
        d2_channels=D2_CHANNELS,
        d2_scale=D2_SCALE,
        d1_tensor_name=D1_TENSOR_NAME,
        d1_channels=D1_CHANNELS,
        d1_scale=D1_SCALE,
        input_range=[0, 1],
        pad_multiple=PAD_MULTIPLE,
        official_lol_checkpoint_path=OFFICIAL_LOL_CKPT,
        patch_size=PATCH,
        batch_size=BATCH,
        optimizer=OPTIMIZER_NAME,
        lr_initial=LR_INITIAL,
        lr_min=LR_MIN,
        warmup_epochs=WARMUP_EPOCHS,
        epochs=EPOCHS,
        loss=LOSS_NAME,
        seed=SEED,
        reference_branch_enabled=False,
        vgg_enabled=False,
        official_test_allowed_during_selection=False,
        retinex_test_psnr_anchor=RETINEX_TEST_PSNR,
        repo_commit=git_head(),
    )


def decide_go(psnr: float) -> str:
    if psnr >= STRONG_GO_PSNR:
        return 'STRONG_GO'
    if psnr >= GO_PSNR:
        return 'GO'
    if psnr >= RETINEX_TEST_PSNR:
        return 'HOLD'
    return 'NO_GO'
