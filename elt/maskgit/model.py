from __future__ import annotations

import torch
import torch.nn as nn

from elt.maskgit.config import ELTConfig


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, cfg: ELTConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        return self.pos_embed(pos).expand(batch, -1, -1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ELTConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, batch_first=True, dropout=cfg.dropout
        )
        self.layer_norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.ffn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.layer_norm(x)
        output, _ = self.attn(
            output, output, output, key_padding_mask=padding_mask, need_weights=False
        )
        x = x + self.ffn_dropout(output)
        output = self.ffn_dropout(self.ffn(self.layer_norm2(x)))
        return x + output


class ELTModel(nn.Module):
    def __init__(self, cfg: ELTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = LearnedPositionalEmbedding(cfg)
        self.unique_layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_unique_layers)]
        )
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def g_theta(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.unique_layers:
            x = block(x, padding_mask)
        return x

    def forward(
        self, x: torch.Tensor, n_loops: int, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.dropout(self.token_embedding(x) + self.pos_embedding(x))
        for _ in range(n_loops):
            h = self.g_theta(h, padding_mask)
        return self.layer_norm(h)

    def forward_dual(
        self,
        x: torch.Tensor,
        l_max: int,
        l_int: int,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(self.token_embedding(x) + self.pos_embedding(x))
        h_int = h
        for step in range(l_max):
            h = self.g_theta(h, padding_mask)
            if step == l_int - 1:
                h_int = h
        return self.layer_norm(h), self.layer_norm(h_int)


class MLMHead(nn.Module):
    def __init__(self, cfg: ELTConfig):
        super().__init__()
        self.linear_projection = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_projection(self.activation(self.layer_norm(x)))
