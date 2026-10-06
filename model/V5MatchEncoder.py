"""Shared MatchEncoder: Y0 and R → 96ch @ H/4. Correspondence only, not texture value."""

from torch import Tensor, nn

MATCH_CH = 96


class V5MatchEncoder(nn.Module):
    def __init__(self, out_ch=MATCH_CH):
        super().__init__()
        self.out_ch = int(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, out_ch, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
