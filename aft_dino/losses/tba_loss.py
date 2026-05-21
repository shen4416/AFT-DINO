from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

from lightly.models.modules import center
from lightly.models.modules.center import CENTER_MODE_TO_FUNCTION


class DINOLoss(Module):
    """DINO loss with Optimal Transport Balanced Assignment (TBA Loss).

    This is adapted from the original DINO loss and replaces the teacher target
    generation with an OT-balanced Sinkhorn assignment.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        center_mode: str = "mean",
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
    ):
        """Initializes the TBA-DINO Loss module."""
        super().__init__()
        if center_mode not in CENTER_MODE_TO_FUNCTION:
            raise ValueError(
                f"Unknown mode '{center_mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_FUNCTION.keys())}."
            )
        self._center_fn = CENTER_MODE_TO_FUNCTION[center_mode]
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters

        self.center: Parameter
        self.register_buffer("center", torch.zeros(1, 1, output_dim))
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: List[Tensor],
        student_out: List[Tensor],
        epoch: int,
    ) -> Tensor:
        """Cross-entropy between balanced teacher targets and student outputs."""
        teacher_out_stacked = torch.stack(teacher_out)

        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.teacher_temp_schedule[epoch].to(
                device=teacher_out_stacked.device, dtype=teacher_out_stacked.dtype
            )
        else:
            teacher_temp = teacher_out_stacked.new_tensor(self.teacher_temp)

        # OT-Balanced teacher targets (Sinkhorn)
        t_logits: Tensor = (teacher_out_stacked - self.center) / teacher_temp
        t_out: Tensor = self._sinkhorn_balanced_assignment(
            t_logits, epsilon=self.sinkhorn_epsilon, n_iters=self.sinkhorn_iters
        )

        student_out_stacked = torch.stack(student_out)
        s_out = F.log_softmax(student_out_stacked / self.student_temp, dim=-1)

        # Calculate feature similarities, ignoring the diagonal
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        loss.fill_diagonal_(0)

        n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = teacher_out_stacked.shape[1]
        loss = loss.sum() / (n_terms * batch_size)

        self.update_center(teacher_out_stacked)
        return loss

    @staticmethod
    @torch.no_grad()
    def _sinkhorn_balanced_assignment(
        logits: Tensor, epsilon: float, n_iters: int
    ) -> Tensor:
        """Computes OT-balanced assignments using Sinkhorn-Knopp iterations."""
        if epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be > 0.")
        if n_iters < 1:
            raise ValueError("sinkhorn_iters must be >= 1.")

        t, b, k = logits.shape
        out = torch.empty_like(logits)

        # We solve for each teacher view independently (t is typically small, e.g., 2).
        for i in range(t):
            scores = logits[i]
            # Compute kernel matrix K = exp(scores / epsilon) with numerical stabilization.
            scores = scores - scores.max(dim=1, keepdim=True).values
            Q = torch.exp(scores / epsilon)  # (b, k)
            Q = Q / (Q.sum() + 1e-12)

            # Target marginals:
            # - each sample (row) sums to 1
            # - each prototype (col) sums to b / k (balanced usage)
            r = Q.new_ones(b)  # (b,)
            c = Q.new_full((k,), float(b) / float(k))  # (k,)

            for _ in range(n_iters):
                # Row normalization: enforce Q 1_k = r
                Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
                Q = Q * r.view(-1, 1)

                # Column normalization: enforce Q^T 1_b = c
                Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
                Q = Q * c.view(1, -1)

            # Ensure per-sample distribution sums to 1 (minor drift correction).
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
            out[i] = Q

        return out

    @torch.no_grad()
    def update_center(self, teacher_out: Tensor) -> None:
        """Moving average update of the center used for the teacher output."""
        batch_center = self._center_fn(x=teacher_out, dim=(0, 1))
        self.center.data = center.center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.center_momentum
        )


# Paper-facing alias.
TBADINOLoss = DINOLoss
