from typing import Tuple

import torch
from torch import nn


class GateBlock(nn.Module):
    """Gate block for adaptive layer-wise feature fusion.

    Given a previous aggregated representation and the current layer
    representation with the same shape [B, D], the block predicts a
    per-dimension gate value in [0, 1]. The gate controls how much information
    is introduced from the current layer and how much is retained from the
    accumulated representation.

    output = gate * current + (1 - gate) * previous

    Layer normalization is applied to stabilize the fused representation.
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.linear = nn.Linear(dimension, dimension)
        self.activation = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(dimension)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if previous.shape != current.shape:
            raise ValueError(
                f"Shape of two inputs doesn't match: {previous.shape} vs {current.shape}"
            )

        # Degenerate case: if the accumulated representation is all zeros,
        # use the current layer representation directly.
        if torch.all(torch.eq(previous, 0)):
            out = self.layernorm(current)
            gate_values = torch.ones_like(current)
            return out, gate_values

        # Predict adaptive fusion weights for current-layer information.
        gate_values = self.activation(self.linear(previous))
        out = current * gate_values + previous * (1.0 - gate_values)
        out = self.layernorm(out)
        return out, gate_values
