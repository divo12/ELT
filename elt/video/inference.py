from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from elt.maskgit.config import ELTConfig
from elt.maskgit.inference import any_time_generate
from elt.video.tokenizer import MAGVITTokenizer, decode_videos


@torch.no_grad()
def generate_video(
    model: torch.nn.Module,
    head: torch.nn.Module,
    tokenizer: MAGVITTokenizer,
    cfg: ELTConfig,
    n_loops: int | Sequence[int],
    class_label: int | torch.Tensor | None = 0,
    batch_size: int = 1,
    init_tokens: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """Generate (B, 3, 16, 128, 128) video clips via MAGVIT ELT parallel decoding (Table 4/5)."""
    token_ids = any_time_generate(
        model,
        head,
        cfg,
        n_loops=n_loops,
        class_label=class_label,
        batch_size=batch_size,
        cfg_scale=1.0,  # Table 4/5: no CFG used on UCF-101
        init_tokens=init_tokens,
        device=device,
        seed=seed,
        reshape_grid=True,
    )
    return decode_videos(tokenizer, token_ids, grid_shape=cfg.grid_shape)


def video_any_time_inference(
    model: torch.nn.Module,
    head: torch.nn.Module,
    tokenizer: MAGVITTokenizer,
    cfg: ELTConfig,
    device: torch.device,
    loop_counts: tuple[int, ...] = (1, 2, 4, 6),
    class_label: int = 0,
    num_keyframes: int = 4,
    seed: int = 42,
):
    """Render filmstrips across loop budgets (including L > L_max extrapolation per Figure 7)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(loop_counts),
        num_keyframes,
        figsize=(3.2 * num_keyframes, 3.0 * len(loop_counts)),
        squeeze=False,
    )
    fig.suptitle(
        f"UCF-101 Any-Time Video Generation (class={class_label}, L_max={cfg.L_max})", fontsize=14
    )

    for row, n_loops in enumerate(loop_counts):
        clip = generate_video(
            model,
            head,
            tokenizer,
            cfg,
            n_loops=n_loops,
            class_label=class_label,
            batch_size=1,
            device=device,
            seed=seed,
        )[0]  # (3, T, H, W)
        t_total = clip.shape[1]
        frame_indices = torch.linspace(0, t_total - 1, num_keyframes).round().long().tolist()
        for col, f_idx in enumerate(frame_indices):
            frame = ((clip[:, f_idx].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
            axes[row, col].imshow(np.clip(frame, 0.0, 1.0))
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(f"L={n_loops} | t={f_idx}")
            else:
                axes[row, col].set_title(f"t={f_idx}")

    plt.tight_layout()
    return fig
