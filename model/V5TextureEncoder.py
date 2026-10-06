"""Multi-scale Ref texture pyramid. R0@H unused in V5.0; R1@H/2 and R2@H/4 used."""

from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

TEXTURE_CHANNELS = (40, 80, 160)
BLOCKS_PER_LEVEL = 2


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class V5TextureEncoder(nn.Module):
    """R → {h: R0 40ch, h2: R1 80ch, h4: R2 160ch}."""

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
