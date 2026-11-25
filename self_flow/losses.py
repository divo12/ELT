"""Self-Flow Training Objective & EMA Teacher (arXiv:2603.06507 Sec 3.4, App A.3/D & BFL Issues #7, #10)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from self_flow.schedules import MODALITY_MASK_RATIOS, TimestepDist, dual_timestep_schedule


@dataclass(frozen=True)
class SelfFlowConfig:
    """Hyperparameters for Self-Flow training (Appendix A.2 & A.3)."""

    mask_ratio: float = 0.25
    gamma: float = 0.8
    student_layer_ratio: float = 0.3
    teacher_layer_ratio: float = 0.7
    ema_decay: float = 0.9999
    label_drop_prob: float = 0.1

    timestep_dist: TimestepDist = "uniform"
    train_shift: float = 1.0
    sample_shift: float = 1.0
    high_noise_prob: float = 0.0

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def layer_indices(self, depth: int) -> tuple[int, int]:
        l_idx = max(1, min(depth - 1, int(round(self.student_layer_ratio * depth))))
        k_idx = max(l_idx + 1, min(depth, int(round(self.teacher_layer_ratio * depth))))
        return l_idx, k_idx


class EMATeacher(nn.Module):
    """Momentum-updated Exponential Moving Average teacher network f_{theta'} (decay = 0.9999, no warmup)."""

    def __init__(self, student: nn.Module, decay: float = 0.9999) -> None:
        super().__init__()
        raw = student.module if hasattr(student, "module") else student
        self.model = copy.deepcopy(raw)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        raw = student.module if hasattr(student, "module") else student
        for p_ema, p_stu in zip(self.model.parameters(), raw.parameters()):
            p_ema.data.mul_(self.decay).add_(p_stu.data, alpha=1.0 - self.decay)
        for b_ema, b_stu in zip(self.model.buffers(), raw.buffers()):
            b_ema.data.copy_(b_stu.data)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def cosine_alignment_loss(student_proj: Tensor, teacher_raw: Tensor) -> Tensor:
    """Eq. 6 + BFL Issue #7: 1 - CosineSimilarity(dim=-1) averaged over tokens & batch."""
    cos_sim = F.cosine_similarity(
        student_proj.float(), teacher_raw.detach().float(), dim=-1, eps=1e-8
    )
    return (1.0 - cos_sim).mean()


def self_flow_train_step(
    student: nn.Module,
    ema_teacher: EMATeacher,
    x0_tokens: Tensor,
    labels: Tensor,
    cfg: SelfFlowConfig,
    null_class_id: int | None = None,
) -> dict[str, Tensor]:
    """Full Self-Flow step (Eq. 2, Eq. 6, Eq. 7) on patched token sequence (B, N, C).

    Per BFL Issue #10, condition dropout is sampled once per sample and shared identically
    between the student and the EMA teacher.
    """
    raw_student = student.module if hasattr(student, "module") else student
    l_layer, k_layer = cfg.layer_indices(raw_student.depth)

    if null_class_id is None:
        null_class_id = raw_student.y_embedder.null_class_id

    cond_labels = labels
    if student.training and cfg.label_drop_prob > 0.0:
        drop = torch.rand(labels.shape[0], device=labels.device) < cfg.label_drop_prob
        cond_labels = torch.where(drop, null_class_id, labels)

    x1 = torch.randn_like(x0_tokens)
    target_velocity = x1 - x0_tokens

    x_tau, x_tau_min, tau, tau_min, _ = dual_timestep_schedule(
        x0_tokens,
        x1=x1,
        mask_ratio=cfg.mask_ratio,
        dist=cfg.timestep_dist,
        shift_alpha=cfg.train_shift,
        high_noise_prob=cfg.high_noise_prob,
    )

    v_pred, z_student = student(
        x_tau, timesteps=tau, vector=cond_labels, return_features=l_layer
    )
    with torch.no_grad():
        _, z_teacher = ema_teacher(
            x_tau_min, timesteps=tau_min, vector=cond_labels, return_raw_features=k_layer
        )

    loss_gen = F.mse_loss(v_pred.float(), target_velocity.float())
    loss_rep = cosine_alignment_loss(z_student, z_teacher)
    loss_total = loss_gen + cfg.gamma * loss_rep
    return {
        "loss": loss_total,
        "loss_gen": loss_gen,
        "loss_rep": loss_rep,
    }


def multimodal_self_flow_step(
    student: nn.Module,
    ema_teacher: EMATeacher,
    x0_tokens: Tensor,
    modality: str,
    cfg: SelfFlowConfig,
    cond: Tensor | None = None,
    modality_weights: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    """Appendix D multi-modal Self-Flow step weighted by w_m in (w_I, w_V, w_A)."""
    raw_student = student.module if hasattr(student, "module") else student
    l_layer, k_layer = cfg.layer_indices(raw_student.depth)
    mask_ratio = MODALITY_MASK_RATIOS.get(modality, cfg.mask_ratio)
    high_noise = 0.05 if modality == "video" else cfg.high_noise_prob

    if cond is not None and student.training and cfg.label_drop_prob > 0.0:
        drop = (
            torch.rand(cond.shape[0], device=cond.device) < cfg.label_drop_prob
        ).view(cond.shape[0], *([1] * (cond.ndim - 1)))
        cond = torch.where(drop, torch.zeros_like(cond), cond)

    x1 = torch.randn_like(x0_tokens)
    target_velocity = x1 - x0_tokens
    x_tau, x_tau_min, tau, tau_min, _ = dual_timestep_schedule(
        x0_tokens,
        x1=x1,
        mask_ratio=mask_ratio,
        dist=cfg.timestep_dist,
        shift_alpha=cfg.train_shift,
        high_noise_prob=high_noise,
    )

    v_pred, z_student = student(
        x_tau, timesteps=tau, modality=modality, cond=cond, return_features=l_layer
    )
    with torch.no_grad():
        _, z_teacher = ema_teacher(
            x_tau_min, timesteps=tau_min, modality=modality, cond=cond, return_raw_features=k_layer
        )

    loss_gen = F.mse_loss(v_pred.float(), target_velocity.float())
    loss_rep = cosine_alignment_loss(z_student, z_teacher)
    w_m = 1.0 if modality_weights is None else modality_weights.get(modality, 1.0)
    loss_total = w_m * (loss_gen + cfg.gamma * loss_rep)
    return {
        "loss": loss_total,
        "loss_gen": loss_gen,
        "loss_rep": loss_rep,
    }
