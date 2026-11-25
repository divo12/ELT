from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from elt.diffusion.config import DiffusionELTConfig


class LearnedPositionalEmbedding_Diffusion(nn.Module):
    def __init__(self, cfg: DiffusionELTConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(cfg.latent_seq_len, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device)
        pos = self.pos_embed(pos)
        return pos.expand(batch, -1, -1)


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: DiffusionELTConfig):
        super().__init__()
        self.projection_layer = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.pos_embedding = LearnedPositionalEmbedding_Diffusion(cfg)
        self.embedding_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.projection_layer(x)
        x = self.pos_embedding(x) + x
        return self.embedding_dropout(x)


class TimeStepEmbedding(nn.Module):
    def __init__(self, cfg: DiffusionELTConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.linear_1 = nn.Linear(self.d_model, self.d_model * 4)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(self.d_model * 4, self.d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = self.linear_1(embedding)
        embedding = self.activation(embedding)
        return self.linear_2(embedding)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DiffusionELTConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.layer_norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.attn_dropout = cfg.dropout
        self.ffn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.layer_norm(x)
        b, n, c = output.shape

        qkv = self.qkv(output)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, c)
        attn_output = self.ffn_dropout(self.out_proj(attn_output))
        x = x + attn_output

        output = self.ffn_dropout(self.ffn(self.layer_norm2(x)))
        return x + output


class ELT_DiT(nn.Module):
    def __init__(self, cfg: DiffusionELTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg)
        self.time_embed = TimeStepEmbedding(cfg)
        self.looped_transformer = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_unique_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.projection = nn.Linear(cfg.d_model, cfg.patch_dim)

    def g_theta(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for unique_layer in self.looped_transformer:
            x = unique_layer(x, padding_mask)
        return x

    def _decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        s = self.cfg.latent_size
        c = self.cfg.latent_channels
        x = self.projection(self.norm(x))
        return x.view(b, s, s, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, t: torch.Tensor, n_loops: int) -> torch.Tensor:
        h = self.patch_embed(x) + self.time_embed(t).unsqueeze(1)
        for _ in range(n_loops):
            h = self.g_theta(h, padding_mask=None)
        return self._decode_tokens(h)

    def forward_dual(
        self, x: torch.Tensor, t: torch.Tensor, l_max: int, l_int: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.patch_embed(x) + self.time_embed(t).unsqueeze(1)
        h_int = h
        for step in range(l_max):
            h = self.g_theta(h, padding_mask=None)
            if step == l_int - 1:
                h_int = h
        return self._decode_tokens(h), self._decode_tokens(h_int)
