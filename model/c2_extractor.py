"""C2-Matching ContrasExtractorSep + pretrained feature_extraction.pth loader.

Vendored architecture from yumingj/C2-Matching @6d60149 (Apache-2.0).
VGG16 backbone is built with weights=None; C2 checkpoint overwrites all
conv weights (no ImageNet init required at runtime after first verify).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg

C2_FEATURE_CKPT = (
    '/root/projects/TTSR-lowlight/pretrain_model/C2matching/feature_extraction.pth')


def _vgg16_features_to_conv3_1():
    """Build VGG16.features truncated at conv3_1 without downloading weights."""
    try:
        backbone = vgg.vgg16(weights=None)
    except TypeError:
        backbone = vgg.vgg16(pretrained=False)
    vgg16_layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ]
    conv3_1_idx = vgg16_layers.index('conv3_1')
    features = backbone.features[:conv3_1_idx + 1]
    modified = OrderedDict()
    for k, v in zip(vgg16_layers, features):
        modified[k] = v
    return nn.Sequential(modified)


class ContrasExtractorLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _vgg16_features_to_conv3_1()
        self.register_buffer(
            'mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std
        return self.model(batch)


class ContrasExtractorSep(nn.Module):
    """Two-stream C2 contrastive extractor (image1=input, image2=ref)."""

    def __init__(self):
        super().__init__()
        self.feature_extraction_image1 = ContrasExtractorLayer()
        self.feature_extraction_image2 = ContrasExtractorLayer()

    def forward(self, image1, image2):
        return {
            'dense_features1': self.feature_extraction_image1(image1),
            'dense_features2': self.feature_extraction_image2(image2),
        }


def load_c2_feature_extractor(path: str = C2_FEATURE_CKPT,
                              device: str = 'cpu',
                              freeze: bool = True) -> ContrasExtractorSep:
    """Strict-load official feature_extraction.pth (24 tensors)."""
    blob = torch.load(path, map_location='cpu')
    if isinstance(blob, dict) and 'state_dict' in blob:
        sd = blob['state_dict']
    else:
        sd = blob
    clean = OrderedDict()
    for k, v in sd.items():
        nk = k
        for pref in ('module.', 'network_student.', 'net_student.', 'student.'):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        clean[nk] = v
    net = ContrasExtractorSep()
    msd = net.state_dict()
    miss = [k for k in msd if k not in clean]
    extra = [k for k in clean if k not in msd]
    shape = [k for k in msd if k in clean
             and tuple(msd[k].shape) != tuple(clean[k].shape)]
    if miss or extra or shape:
        raise SystemExit(
            'C2 feature_extraction incomplete: miss=%d extra=%d shape=%d first=%s'
            % (len(miss), len(extra), len(shape), (miss or shape or extra)[:4]))
    net.load_state_dict(clean, strict=True)
    net.to(device)
    if freeze:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    print('[C2] loaded %d tensors from %s (freeze=%s)'
          % (len(clean), path, freeze), flush=True)
    return net


class C2MatchFeatures(nn.Module):
    """Y0/Ref [0,1] → (q,k) dense features @ H/4, 256ch (C2 pretrained)."""

    def __init__(self, ckpt: str = C2_FEATURE_CKPT, freeze: bool = True):
        super().__init__()
        self.extractor = load_c2_feature_extractor(ckpt, device='cpu', freeze=freeze)
        self.is_pretrained = True

    def forward(self, y0_01: torch.Tensor, ref_01: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image1 = content (Y0), image2 = reference
        out = self.extractor(y0_01, ref_01)
        return out['dense_features1'], out['dense_features2']
