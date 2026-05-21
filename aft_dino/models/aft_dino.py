import copy

import torch
from torch import nn
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad

from aft_dino.hafd import BackboneLayerAdapter, GateBlock


class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()

        # Student/teacher backbones and projection heads.
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)

        # Whether to use HAFD gated multi-level fusion.
        self.use_gates = False

        # Forward hooks store intermediate backbone features.
        self.student_features = []
        self.teacher_features = []
        self.student_hooks = []
        self.teacher_hooks = []

        # HAFD adapters and GateBlocks.
        self.num_backbone_layers = 0
        self.student_adapters = nn.ModuleList()
        self.teacher_adapters = nn.ModuleList()
        self.student_gates = nn.ModuleList()
        self.teacher_gates = nn.ModuleList()

        # Cached average gate values for training logs.
        self.last_student_gate_means = []
        self.last_teacher_gate_means = []

        # Build HAFD when the backbone exposes a YOLO-style ModuleList.
        if hasattr(self.student_backbone, "model"):
            try:
                self._init_gate_blocks(input_dim)
                self.use_gates = True
            except Exception as e:
                # Fallback to final-layer DINO when HAFD cannot be initialized.
                print(f"[DINO] HAFD initialization failed; fallback to single-layer DINO. Reason: {e}")
                self.use_gates = False

        # Teacher modules are updated by EMA and kept gradient-free.
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
        if self.use_gates:
            deactivate_requires_grad(self.teacher_adapters)
            deactivate_requires_grad(self.teacher_gates)

    def _init_gate_blocks(self, input_dim: int):
        """
        Register YOLO backbone hooks, infer intermediate feature dimensions,
        and build HAFD adapters and GateBlocks.
        """

        # Register hooks for collecting intermediate backbone outputs.
        def student_hook(module, inp, out):
            self.student_features.append(out)

        def teacher_hook(module, inp, out):
            self.teacher_features.append(out)

        for m in self.student_backbone.model:
            self.student_hooks.append(m.register_forward_hook(student_hook))
        for m in self.teacher_backbone.model:
            self.teacher_hooks.append(m.register_forward_hook(teacher_hook))

        # Infer feature dimensions with a dummy input.
        device = next(self.student_backbone.parameters()).device
        dummy = torch.randn(1, 3, 224, 224, device=device)
        self.student_features.clear()
        with torch.no_grad():
            _ = self.student_backbone(dummy)

        self.num_backbone_layers = len(self.student_features)
        if self.num_backbone_layers == 0:
            # Disable HAFD if intermediate features are unavailable.
            self.use_gates = False
            return

        # Build one adapter per intermediate feature level.
        self.student_adapters = nn.ModuleList()
        self.teacher_adapters = nn.ModuleList()
        for feat in self.student_features:
            if feat.dim() == 4:
                in_dim = feat.shape[1]  # [B, C, H, W] -> C
            elif feat.dim() == 3:
                in_dim = feat.shape[-1]  # [B, S, C] -> C
            elif feat.dim() == 2:
                in_dim = feat.shape[1]  # [B, D]
            else:
                in_dim = feat.view(feat.size(0), -1).shape[1]
            self.student_adapters.append(BackboneLayerAdapter(in_dim, input_dim))
            self.teacher_adapters.append(BackboneLayerAdapter(in_dim, input_dim))

        # Build one GateBlock for each adjacent feature-level pair.
        if self.num_backbone_layers > 1:
            self.student_gates = nn.ModuleList(
                [GateBlock(input_dim) for _ in range(self.num_backbone_layers - 1)]
            )
            # Initialize teacher-side GateBlocks from the student branch for EMA updates.
            self.teacher_gates = copy.deepcopy(self.student_gates)
        else:
            self.student_gates = nn.ModuleList()
            self.teacher_gates = nn.ModuleList()

        self.last_student_gate_means = [0.0 for _ in range(len(self.student_gates))]
        self.last_teacher_gate_means = [0.0 for _ in range(len(self.teacher_gates))]

    def _aggregate_with_gates(self, features, adapters, gates, track: str):
        """
        Adapt and fuse multi-level backbone features with GateBlocks.
        features: backbone outputs collected by forward hooks
        adapters: one BackboneLayerAdapter per feature level
        gates: GateBlocks between adjacent feature levels
        track: branch name used to cache average gate values
        """
        if len(adapters) == 0 or len(features) == 0:
            # Fallback: flatten the final available feature tensor.
            x = features[-1]
            return x.flatten(start_dim=1)

        # First level: adaptation only.
        agg = adapters[0](features[0])
        gate_means = []

        # Subsequent levels: fuse current and accumulated representations.
        for i in range(1, len(adapters)):
            cur = adapters[i](features[i])
            if len(gates) >= i:
                agg, gate_values = gates[i - 1](agg, cur)
                if gate_values is not None:
                    # Log the global average gate value for this GateBlock.
                    gate_means.append(gate_values.detach().mean().item())

        if track == "student":
            self.last_student_gate_means = gate_means
        else:
            self.last_teacher_gate_means = gate_means
        return agg

    def forward(self, x):
        # Student branch: gradient-enabled HAFD fusion.
        if self.use_gates and len(self.student_adapters) > 0:
            # Clear features collected from the previous forward pass.
            self.student_features.clear()
            _ = self.student_backbone(x)
            y = self._aggregate_with_gates(
                self.student_features,
                self.student_adapters,
                self.student_gates,
                track="student",
            )
        else:
            # Fallback: use the final backbone output.
            y = self.student_backbone(x).flatten(start_dim=1)

        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        # Teacher branch: no gradients; parameters are updated by EMA.
        if self.use_gates and len(self.teacher_adapters) > 0:
            self.teacher_features.clear()
            with torch.no_grad():
                _ = self.teacher_backbone(x)
                y = self._aggregate_with_gates(
                    self.teacher_features,
                    self.teacher_adapters,
                    self.teacher_gates,
                    track="teacher",
                )
        else:
            with torch.no_grad():
                y = self.teacher_backbone(x).flatten(start_dim=1)

        z = self.teacher_head(y)
        return z


# Paper-facing alias.
AFTDINO = DINO
