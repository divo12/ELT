import random
import torch
import torch.nn.functional as F

from elt.maskgit.config import ELTConfig


def gt_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[mask], targets[mask])


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits[mask], dim=-1)
    teacher_probs = F.softmax(teacher_logits[mask], dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


def ILSD_train_step(
    model: torch.nn.Module,
    head: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    lambda_val: float,
    cfg: ELTConfig,
) -> tuple[torch.Tensor, int]:
    l_int = random.randint(cfg.L_min, cfg.L_max - 1)
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "forward_dual"):
        teacher_hidden, student_hidden = raw_model.forward_dual(tokens, cfg.L_max, l_int)
    else:
        teacher_hidden = model(tokens, n_loops=cfg.L_max)
        student_hidden = model(tokens, n_loops=l_int)

    teacher_logits = head(teacher_hidden)
    student_logits = head(student_hidden)

    loss_teacher = gt_loss(teacher_logits, targets, mask)
    loss_student_gt = gt_loss(student_logits, targets, mask)
    loss_student_distillation = distillation_loss(
        student_logits, teacher_logits.detach(), mask
    )
    total = (
        loss_teacher
        + lambda_val * loss_student_gt
        + (1.0 - lambda_val) * loss_student_distillation
    )
    return total, l_int
