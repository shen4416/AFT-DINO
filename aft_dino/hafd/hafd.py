from typing import List, Tuple

import torch
from torch import nn

from .adapter import BackboneLayerAdapter
from .gate_block import GateBlock


class HAFD(nn.Module):
    """Hierarchical Adaptive Feature Fusion Distillation module.

    HAFD maps multi-level backbone features into a common representation space
    and recursively fuses them with GateBlock modules. The resulting fused
    representation is used as the self-supervised distillation feature.
    """

    def __init__(self, in_dims: List[int], target_dim: int):
        super().__init__()
        self.in_dims = in_dims
        self.target_dim = target_dim
        self.adapters = nn.ModuleList(
            [BackboneLayerAdapter(in_dim, target_dim) for in_dim in in_dims]
        )
        self.gates = nn.ModuleList(
            [GateBlock(target_dim) for _ in range(max(len(in_dims) - 1, 0))]
        )
        self.last_gate_means: List[float] = []

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float]]:
        if len(features) == 0:
            raise ValueError("HAFD expects at least one feature tensor.")
        if len(features) != len(self.adapters):
            raise ValueError(
                f"Expected {len(self.adapters)} features, but got {len(features)}."
            )

        fused = self.adapters[0](features[0])
        gate_means: List[float] = []

        for i in range(1, len(self.adapters)):
            current = self.adapters[i](features[i])
            if len(self.gates) >= i:
                fused, gate_values = self.gates[i - 1](fused, current)
                if gate_values is not None:
                    gate_means.append(gate_values.detach().mean().item())

        self.last_gate_means = gate_means
        return fused, gate_means
