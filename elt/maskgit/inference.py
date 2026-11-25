from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from elt.maskgit.config import ELTConfig


def any_time_generate(
    model: torch.nn.Module,
    head: torch.nn.Module,
    cfg: ELTConfig,
    n_loops: int,
    class_label: int | None = None,
    cfg_scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    model.eval()
    head.eval()
    with torch.no_grad():
        x_t = torch.full((1, cfg.max_seq_len), cfg.mask_token_id, dtype=torch.long, device=device)
        k_steps = cfg.sampling_steps
        seq_len = cfg.max_seq_len

        for k in range(k_steps):
            hidden_state = model(x_t, n_loops=n_loops)
            logits = head(hidden_state)[..., : cfg.vocab_size - 1]

            stemp = max(0.5 + 0.8 * (1.0 - (k + 1) / k_steps), 1e-5)
            probs = F.softmax(logits / stemp, dim=-1).squeeze(0)
            sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            is_masked = x_t.squeeze(0) == cfg.mask_token_id
            candidate = torch.where(is_masked, sampled_ids, x_t.squeeze(0))

            raw_probs = F.softmax(logits, dim=-1).squeeze(0)
            confidence = raw_probs[torch.arange(seq_len, device=device), candidate]
            confidence = torch.where(is_masked, confidence, torch.tensor(float("inf"), device=device))

            mask_ratio = math.cos(0.5 * math.pi * (k + 1) / k_steps)
            num_keep_masked = int(math.floor(seq_len * mask_ratio))
            x_t[0] = candidate
            if num_keep_masked > 0:
                _, lowest_indices = torch.topk(confidence, k=num_keep_masked, largest=False)
                x_t[0, lowest_indices] = cfg.mask_token_id

    return x_t
