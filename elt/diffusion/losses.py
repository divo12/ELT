from __future__ import annotations

import torch

from elt.diffusion.config import DiffusionELTConfig
from elt.diffusion.inference import NoiseScheduler


def sigmoid_weighted_mse(
    pred_x0: torch.Tensor, target_x0: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return (weight * (pred_x0 - target_x0).pow(2)).mean()


def diff_ILSD_train_step(
    model: torch.nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor | None,
    lambda_val: float,
    cfg: DiffusionELTConfig,
    noise_scheduler: NoiseScheduler | None = None,
    use_ilsd: bool = True,
) -> tuple[torch.Tensor, int]:
    """Algorithm 1 + Section 4 Diffusion ILSD loss (shifted cosine + v-pred + Sigmoid-ELBO)."""
    if noise_scheduler is None:
        noise_scheduler = NoiseScheduler(cfg)

    b = x0.shape[0]
    t = torch.rand(b, device=x0.device)
    x_t, a_t, s_t, log_snr = noise_scheduler.add_noise(x0, t)
    w = noise_scheduler.sigmoid_weight(log_snr).view(b, 1, 1, 1).to(x0.dtype)

    if labels is None:
        cond_labels = torch.full((b,), cfg.null_class_id, dtype=torch.long, device=x0.device)
    else:
        cond_labels = labels
        if model.training and cfg.label_drop_prob > 0.0:
            drop = torch.rand(b, device=x0.device) < cfg.label_drop_prob
            cond_labels = torch.where(drop, cfg.null_class_id, cond_labels)

    raw_model = model.module if hasattr(model, "module") else model

    if not use_ilsd or cfg.L_max == 1:
        v_max = model(x_t, t, n_loops=cfg.L_max, labels=cond_labels)
        x0_max = noise_scheduler.x0_from_v(x_t, v_max, a_t, s_t)
        return sigmoid_weighted_mse(x0_max, x0, w), cfg.L_max

    if hasattr(raw_model, "forward_dual"):
        v_max, v_int, l_int = raw_model.forward_dual(
            x_t, t, l_max=cfg.L_max, l_min=cfg.L_min, labels=cond_labels
        )
    else:
        l_int = int(torch.randint(cfg.L_min, cfg.L_max, (1,), device=x0.device).item())
        v_max = model(x_t, t, n_loops=cfg.L_max, labels=cond_labels)
        v_int = model(x_t, t, n_loops=l_int, labels=cond_labels)

    x0_max = noise_scheduler.x0_from_v(x_t, v_max, a_t, s_t)
    x0_int = noise_scheduler.x0_from_v(x_t, v_int, a_t, s_t)

    loss_teacher = sigmoid_weighted_mse(x0_max, x0, w)
    loss_student_gt = sigmoid_weighted_mse(x0_int, x0, w)
    loss_distill = sigmoid_weighted_mse(x0_int, x0_max.detach(), w)

    loss_total = (
        loss_teacher + lambda_val * loss_student_gt + (1.0 - lambda_val) * loss_distill
    )
    return loss_total, l_int
