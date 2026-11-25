from __future__ import annotations

import torch
import torch.nn.functional as F

from elt.maskgit.config import ELTConfig


def gt_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[mask], targets[mask])


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits[mask], dim=-1)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits[mask].detach(), dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


def _head_logits(head: torch.nn.Module, hidden: torch.Tensor, raw_model: torch.nn.Module) -> torch.Tensor:
    raw_head = head.module if hasattr(head, "module") else head
    if getattr(raw_head, "decoder", None) is None and hasattr(raw_model, "token_embedding"):
        return head(hidden, embed_weight=raw_model.token_embedding.weight)
    return head(hidden)


def ILSD_train_step(
    model: torch.nn.Module,
    head: torch.nn.Module,
    masked_tokens: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    lambda_val: float,
    cfg: ELTConfig,
    labels: torch.Tensor | None = None,
    use_ilsd: bool = True,
) -> tuple[torch.Tensor, int]:
    """Algorithm 1 + Section 4 Masked Generative ILSD loss (MaskGIT & MAGVIT)."""
    b = masked_tokens.shape[0]
    if labels is None:
        cond_labels = torch.full(
            (b,), cfg.null_class_id, dtype=torch.long, device=masked_tokens.device
        )
    else:
        cond_labels = labels
        if model.training and cfg.label_drop_prob > 0.0:
            drop = torch.rand(b, device=masked_tokens.device) < cfg.label_drop_prob
            cond_labels = torch.where(drop, cfg.null_class_id, cond_labels)

    raw_model = model.module if hasattr(model, "module") else model

    if not use_ilsd or cfg.L_max == 1:
        teacher_hidden = model(masked_tokens, n_loops=cfg.L_max, labels=cond_labels)
        teacher_logits = _head_logits(head, teacher_hidden, raw_model)
        return gt_loss(teacher_logits, targets, mask), cfg.L_max

    if hasattr(raw_model, "forward_dual"):
        teacher_hidden, student_hidden, l_int = raw_model.forward_dual(
            masked_tokens, l_max=cfg.L_max, l_min=cfg.L_min, labels=cond_labels
        )
    else:
        l_int = int(torch.randint(cfg.L_min, cfg.L_max, (1,), device=masked_tokens.device).item())
        teacher_hidden = model(masked_tokens, n_loops=cfg.L_max, labels=cond_labels)
        student_hidden = model(masked_tokens, n_loops=l_int, labels=cond_labels)

    teacher_logits = _head_logits(head, teacher_hidden, raw_model)
    student_logits = _head_logits(head, student_hidden, raw_model)

    loss_teacher = gt_loss(teacher_logits, targets, mask)
    loss_student_gt = gt_loss(student_logits, targets, mask)
    loss_distill = distillation_loss(student_logits, teacher_logits, mask)
    total = loss_teacher + lambda_val * loss_student_gt + (1.0 - lambda_val) * loss_distill
    return total, l_int
