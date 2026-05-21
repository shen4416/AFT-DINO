import torch
from torch import nn
from ultralytics.nn.modules import Conv


class PerChannelGeM(nn.Module):
    def __init__(self, c: int, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.p = nn.Parameter(torch.ones(c, 1, 1) * p_init)  # [C, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bound the pooling exponent for numerical stability.
        p = self.p.clamp(min=1.0, max=6.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=(-1, -2), keepdim=True)
        return x.pow(1.0 / p)


class PoolHead(nn.Module):
    def __init__(self, f: int, i: int, c1: int):
        super().__init__()
        self.f, self.i = f, i
        self.conv = Conv(c1, 1280, 1, 1, None, 1)
        # GroupNorm is more stable than BatchNorm for small SSL batch sizes.
        self.gn = nn.GroupNorm(32, 1280)
        self.pool = PerChannelGeM(1280, p_init=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.pool(x)
        return x
