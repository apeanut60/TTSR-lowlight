"""V6 texture pyramid for LLFormer D2/D1: 16 / 32 / 64."""

from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

from model.V5TextureEncoder import ResidualBlock

TEXTURE_CHANNELS = (16, 32, 64)
BLOCKS_PER_LEVEL = 2


class V6TextureEncoder(nn.Module):
    """R[0,1] → {h: 16ch, h2: 32ch, h4: 64ch}."""

    def __init__(self, channels=TEXTURE_CHANNELS, blocks_per_level=BLOCKS_PER_LEVEL):
        super().__init__()
        c0, c1, c2 = channels
        self.channels = tuple(channels)
        self.stem = nn.Conv2d(3, c0, 3, 1, 1)
        self.level0 = nn.Sequential(*[ResidualBlock(c0) for _ in range(blocks_per_level)])
        self.down1 = nn.Conv2d(c0, c1, 3, 2, 1)
        self.level1 = nn.Sequential(*[ResidualBlock(c1) for _ in range(blocks_per_level)])
        self.down2 = nn.Conv2d(c1, c2, 3, 2, 1)
        self.level2 = nn.Sequential(*[ResidualBlock(c2) for _ in range(blocks_per_level)])

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        r0 = self.level0(self.stem(x))
        r1 = self.level1(self.down1(r0))
        r2 = self.level2(self.down2(r1))
        return {'h': r0, 'h2': r1, 'h4': r2}
