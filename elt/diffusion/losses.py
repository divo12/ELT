import random
import torch
import torch.nn.functional as F

from elt.diffusion.config import DiffusionELTConfig


def diff_gt_loss(predicted_noise: torch.Tensor, actual_noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_noise, actual_noise)


def diff_distillation_loss(
    student_pred: torch.Tensor, teacher_pred: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(student_pred, teacher_pred)


def diff_ILSD_train_step(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    actual_noise: torch.Tensor,
    lambda_val: float,
    cfg: DiffusionELTConfig,
) -> tuple[torch.Tensor, int]:
    l_int = random.randint(cfg.L_min, cfg.L_max - 1)
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "forward_dual"):
        teacher_pred, student_pred = raw_model.forward_dual(x_t, t, cfg.L_max, l_int)
    else:
        teacher_pred = model(x_t, t, n_loops=cfg.L_max)
        student_pred = model(x_t, t, n_loops=l_int)

    loss_teacher = diff_gt_loss(teacher_pred, actual_noise)
    loss_student = diff_gt_loss(student_pred, actual_noise)
    loss_student_distillation = diff_distillation_loss(student_pred, teacher_pred.detach())
    loss_total = (
        loss_teacher
        + lambda_val * loss_student
        + (1.0 - lambda_val) * loss_student_distillation
    )
    return loss_total, l_int
