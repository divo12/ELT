from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from elt.maskgit.config import ELTConfig
from elt.maskgit.losses import _head_logits


@torch.no_grad()
def any_time_generate(
    model: torch.nn.Module,
    head: torch.nn.Module,
    cfg: ELTConfig,
    n_loops: int | Sequence[int],
    class_label: int | torch.Tensor | None = None,
    batch_size: int = 1,
    cfg_scale: float | None = None,
    init_tokens: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    reshape_grid: bool = False,
) -> torch.Tensor:
    """Iterative parallel MaskGIT/MAGVIT decoding with CFG, STemp (Eq. 2), and Any-Time loops."""
    model.eval()
    head.eval()
    if seed is not None:
        torch.manual_seed(seed)

    if isinstance(class_label, torch.Tensor):
        labels = class_label.to(device=device, dtype=torch.long)
        batch_size = labels.shape[0]
    elif class_label is not None:
        labels = torch.full((batch_size,), int(class_label), dtype=torch.long, device=device)
    else:
        labels = torch.full((batch_size,), cfg.null_class_id, dtype=torch.long, device=device)

    guidance = cfg.cfg_scale if cfg_scale is None else cfg_scale
    use_cfg = guidance != 1.0 and bool((labels != cfg.null_class_id).any().item())

    seq_len = cfg.max_seq_len
    if init_tokens is not None:
        ids = init_tokens.to(device=device, dtype=torch.long).reshape(batch_size, seq_len).clone()
    else:
        ids = torch.full((batch_size, seq_len), cfg.mask_token_id, dtype=torch.long, device=device)

    k_steps = cfg.sampling_steps
    loop_sched = [n_loops] * k_steps if isinstance(n_loops, int) else list(n_loops)
    if len(loop_sched) != k_steps:
        raise ValueError(f"Expected {k_steps} loop counts, got {len(loop_sched)}")

    null_labels = torch.full_like(labels, cfg.null_class_id)
    raw_model = model.module if hasattr(model, "module") else model

    for k in range(k_steps):
        l_k = loop_sched[k]
        if use_cfg:
            both_ids = torch.cat([ids, ids], dim=0)
            both_lbl = torch.cat([labels, null_labels], dim=0)
            both_hidden = model(both_ids, n_loops=l_k, labels=both_lbl)
            logits_cond, logits_uncond = _head_logits(head, both_hidden, raw_model).chunk(2, dim=0)
            logits = logits_uncond + guidance * (logits_cond - logits_uncond)
        else:
            hidden = model(ids, n_loops=l_k, labels=labels)
            logits = _head_logits(head, hidden, raw_model)

        logits = logits[..., : cfg.codebook_size]

        # Eq. 2: STemp = bias + scale * (1 - (k + 1) / K)
        stemp = max(cfg.temp_bias + cfg.temp_scale * (1.0 - (k + 1) / k_steps), 1e-5)
        sampled_ids = torch.distributions.Categorical(logits=logits / stemp).sample()

        is_masked = ids == cfg.mask_token_id
        ids = torch.where(is_masked, sampled_ids, ids)

        mask_ratio = math.cos(0.5 * math.pi * (k + 1) / k_steps)
        num_keep_masked = int(math.floor(seq_len * mask_ratio))
        if num_keep_masked <= 0:
            continue

        raw_probs = F.softmax(logits, dim=-1)
        selected_probs = raw_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        gumbel = -torch.log(-torch.log(torch.rand_like(selected_probs).clamp_min(1e-10)))
        confidence = torch.log(selected_probs.clamp_min(1e-10)) + stemp * gumbel
        confidence = torch.where(
            is_masked, confidence, torch.tensor(float("inf"), device=device, dtype=confidence.dtype)
        )

        lowest_indices = confidence.argsort(dim=-1)[:, :num_keep_masked]
        ids.scatter_(1, lowest_indices, cfg.mask_token_id)

    if reshape_grid:
        return ids.view(batch_size, *cfg.grid_shape)
    return ids
