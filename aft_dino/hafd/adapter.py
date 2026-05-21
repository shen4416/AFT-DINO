import torch
from torch import nn


class BackboneLayerAdapter(nn.Module):
    """Adapt intermediate backbone features to a fixed-dimensional vector.

    HAFD uses this adapter to map YOLO backbone features from different depths
    into a shared representation space before adaptive gated fusion.

    Supported inputs:
    - [B, C, H, W]: convolutional feature maps
    - [B, S, C]: sequence-like features
    - [B, D]: flattened features
    """

    def __init__(self, in_dim: int, target_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.linear = nn.Linear(in_dim, target_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature maps: spatial average pooling.
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        # Sequence-like features: average over the sequence dimension.
        elif x.dim() == 3:
            x = x.mean(dim=1)
        # Flattened features: keep unchanged.
        elif x.dim() == 2:
            pass
        # Fallback for uncommon feature layouts.
        else:
            x = x.view(x.size(0), -1)
        return self.linear(x)
