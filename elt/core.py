"""Shared Elastic Looped Transformer (ELT) core primitives and paper scale presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ArchSpec:
    d_model: int
    mlp_dim: int
    n_heads: int
    baseline_layers: int


# Table 6: Masked Generative Transformer Model Scaling (MaskGIT / MAGVIT)
MASKGIT_SCALES: dict[str, ArchSpec] = {
    "S": ArchSpec(d_model=384, mlp_dim=1536, n_heads=6, baseline_layers=12),
    "B": ArchSpec(d_model=768, mlp_dim=3072, n_heads=12, baseline_layers=12),
    "L": ArchSpec(d_model=1024, mlp_dim=4096, n_heads=16, baseline_layers=24),
    "XL": ArchSpec(d_model=1152, mlp_dim=4608, n_heads=16, baseline_layers=28),
    "H": ArchSpec(d_model=1280, mlp_dim=5120, n_heads=16, baseline_layers=32),
    "g": ArchSpec(d_model=1408, mlp_dim=6144, n_heads=16, baseline_layers=40),
    "G": ArchSpec(d_model=1664, mlp_dim=8192, n_heads=16, baseline_layers=48),
}

HIGH_WD_MASKGIT_SCALES = frozenset({"H", "g", "G"})

# Table 7: Diffusion Transformer Scaling (+ standard DiT scales for smaller hardware)
DIT_SCALES: dict[str, ArchSpec] = {
    "DiT-S": ArchSpec(d_model=384, mlp_dim=1536, n_heads=6, baseline_layers=12),
    "DiT-B": ArchSpec(d_model=768, mlp_dim=3072, n_heads=12, baseline_layers=12),
    "DiT-L": ArchSpec(d_model=1024, mlp_dim=4096, n_heads=16, baseline_layers=24),
    "DiT-XL": ArchSpec(d_model=1152, mlp_dim=4608, n_heads=16, baseline_layers=28),
    "Small": ArchSpec(d_model=2048, mlp_dim=8192, n_heads=16, baseline_layers=16),
    "Large": ArchSpec(d_model=2048, mlp_dim=8192, n_heads=16, baseline_layers=32),
}


def linear_curriculum(step: int, total_steps: int, start: float = 1.0, end: float = 0.0) -> float:
    """Linearly decay ILSD curriculum weight lambda from 1 -> 0 over training (Section 4)."""
    if total_steps <= 0:
        return end
    progress = min(max(step / total_steps, 0.0), 1.0)
    return start + (end - start) * progress


class LoopedCompositeBlock(nn.Module):
    """Weight-shared composite block g_Theta of N unique layers looped L times."""

    def __init__(self, layers: Sequence[nn.Module], grad_checkpoint: bool = False) -> None:
        super().__init__()
        if len(layers) < 1:
            raise ValueError("LoopedCompositeBlock requires at least 1 unique layer.")
        self.layers = nn.ModuleList(layers)
        self.grad_checkpoint = grad_checkpoint

    @property
    def n_unique_layers(self) -> int:
        return len(self.layers)

    def _run_once(self, h: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h) if cond is None else layer(h, cond)
        return h

    def step_once(self, h: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.grad_checkpoint and self.training and h.requires_grad:
            return checkpoint(self._run_once, h, cond, use_reentrant=False)
        return self._run_once(h, cond)

    def forward(
        self, h: torch.Tensor, loops: int, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Algorithm 2: Any-Time inference forward for arbitrary loop budget L >= 1."""
        if loops < 1:
            raise ValueError(f"loops must be >= 1, got {loops}")
        for _ in range(loops):
            h = self.step_once(h, cond)
        return h

    def forward_dual(
        self,
        h: torch.Tensor,
        l_max: int,
        l_min: int = 1,
        l_int: int | None = None,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Algorithm 1: Single forward pass yielding (F_max, F_int, L_int).

        Student state F_int (at L_int loops) is captured as a strict prefix of
        teacher trajectory F_max (at L_max loops) with zero extra trunk FLOPs.
        """
        if not (1 <= l_min < l_max):
            raise ValueError(f"Require 1 <= l_min < l_max, got l_min={l_min}, l_max={l_max}")
        if l_int is None:
            l_int = int(torch.randint(l_min, l_max, (1,), device=h.device).item())
        elif not (l_min <= l_int < l_max):
            raise ValueError(f"l_int={l_int} outside [{l_min}, {l_max})")

        f_curr = h
        f_int = h
        for step in range(l_max):
            f_curr = self.step_once(f_curr, cond)
            if step == l_int - 1:
                f_int = f_curr
        return f_curr, f_int, l_int
