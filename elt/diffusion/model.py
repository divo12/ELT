from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from elt.core import LoopedCompositeBlock
from elt.diffusion.config import DiffusionELTConfig


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous-time sinusoidal timestep embedding for t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 == 1 else emb


def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sin-cos positional embedding matching Peebles & Xie (2023) DiT."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sincos embed.")
    quarter = embed_dim // 4
    omega = torch.exp(-math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / quarter)
    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(coords, coords, indexing="ij")
    out_h = grid_h.reshape(-1, 1) * omega.unsqueeze(0)
    out_w = grid_w.reshape(-1, 1) * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=-1)
    return emb.unsqueeze(0)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaptive LayerNorm-Zero (adaLN-Zero) conditioning."""

    def __init__(self, cfg: DiffusionELTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.norm1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.norm2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_model, 6 * cfg.d_model))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(
            6, dim=-1
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        b, n, c = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = self.out_proj(attn.transpose(1, 2).contiguous().view(b, n, c))
        x = x + gate_msa.unsqueeze(1) * attn
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class DiTFinalHead(nn.Module):
    """Shared adaLN-Zero final layer mapping hidden tokens back to latent patches."""

    def __init__(self, cfg: DiffusionELTConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_model, 2 * cfg.d_model))
        self.proj = nn.Linear(cfg.d_model, cfg.patch_dim)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


class ELT_DiT(nn.Module):
    """Paper-complete Elastic Looped Diffusion Transformer (Section 4, Tables 5 & 7)."""

    def __init__(self, cfg: DiffusionELTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(
            cfg.latent_channels, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        pos = build_2d_sincos_pos_embed(cfg.d_model, cfg.grid_size)
        self.register_buffer("pos_emb", pos, persistent=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.cls_emb = nn.Embedding(cfg.num_classes + 1, cfg.d_model)

        self.trunk = LoopedCompositeBlock(
            [DiTBlock(cfg) for _ in range(cfg.n_unique_layers)],
            grad_checkpoint=cfg.grad_checkpoint,
        )
        self.head = DiTFinalHead(cfg)
        self._init_weights()

    def _init_weights(self) -> None:
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_emb.weight, std=0.02)
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def unpatchify(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        b = patch_tokens.shape[0]
        g = self.cfg.grid_size
        p = self.cfg.patch_size
        c = self.cfg.latent_channels
        x = patch_tokens.view(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def embed_inputs(
        self, x_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = x_t.shape[0]
        if labels is None:
            labels = torch.full((b,), self.cfg.null_class_id, dtype=torch.long, device=x_t.device)
        tokens = self.patch_embed(x_t).flatten(2).transpose(1, 2) + self.pos_emb.to(x_t.dtype)
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.cfg.d_model).to(x_t.dtype))
        cond = t_emb + self.cls_emb(labels)
        return tokens, cond

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        n_loops: int,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Any-Time inference forward predicting velocity v_hat of shape (B, C, H, W)."""
        tokens, cond = self.embed_inputs(x_t, t, labels)
        f = self.trunk(tokens, loops=n_loops, cond=cond)
        return self.unpatchify(self.head(f, cond))

    def forward_dual(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        l_max: int,
        l_min: int = 1,
        l_int: int | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Single-pass ILSD forward yielding (v_max, v_int, l_int)."""
        tokens, cond = self.embed_inputs(x_t, t, labels)
        f_max, f_int, sampled_l_int = self.trunk.forward_dual(
            tokens, l_max=l_max, l_min=l_min, l_int=l_int, cond=cond
        )
        v_max = self.unpatchify(self.head(f_max, cond))
        v_int = self.unpatchify(self.head(f_int, cond))
        return v_max, v_int, sampled_l_int


DiTELT = ELT_DiT
