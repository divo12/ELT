from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from elt.core import LoopedCompositeBlock
from elt.maskgit.config import ELTConfig


class BERTBlock(nn.Module):
    """Bidirectional Pre-LN Transformer block used in MaskGIT and MAGVIT."""

    def __init__(self, cfg: ELTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        b, n, c = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = self.out_proj(attn.transpose(1, 2).contiguous().view(b, n, c))
        x = x + self.resid_dropout(attn)
        return x + self.ffn(self.norm2(x))


class ELTModel(nn.Module):
    """Paper-complete Elastic Looped Masked Generative Transformer (Image MaskGIT & Video MAGVIT)."""

    def __init__(self, cfg: ELTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.cls_embedding = nn.Embedding(cfg.num_classes + 1, cfg.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + cfg.max_seq_len, cfg.d_model) * 0.02)
        self.in_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.trunk = LoopedCompositeBlock(
            [BERTBlock(cfg) for _ in range(cfg.n_unique_layers)],
            grad_checkpoint=cfg.grad_checkpoint,
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.cls_embedding.weight, std=0.02)

    def embed(self, tokens: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        tokens_flat = tokens.reshape(tokens.shape[0], -1)
        b, seq_len = tokens_flat.shape
        if labels is None:
            labels = torch.full((b,), self.cfg.null_class_id, dtype=torch.long, device=tokens.device)
        cls_tok = self.cls_embedding(labels).unsqueeze(1)
        tok_emb = self.token_embedding(tokens_flat)
        seq = torch.cat([cls_tok, tok_emb], dim=1)
        return self.dropout(self.in_norm(seq + self.pos_embedding[:, : 1 + seq_len]))

    def forward(
        self,
        tokens: torch.Tensor,
        n_loops: int,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Any-Time inference forward returning visual token hidden states (B, seq_len, d_model)."""
        h = self.embed(tokens, labels)
        f = self.trunk(h, loops=n_loops)
        return self.out_norm(f[:, 1:])

    def forward_dual(
        self,
        tokens: torch.Tensor,
        l_max: int,
        l_min: int = 1,
        l_int: int | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Single-pass ILSD forward yielding (hidden_max, hidden_int, l_int) for visual tokens."""
        h = self.embed(tokens, labels)
        f_max, f_int, sampled_l_int = self.trunk.forward_dual(
            h, l_max=l_max, l_min=l_min, l_int=l_int
        )
        return self.out_norm(f_max[:, 1:]), self.out_norm(f_int[:, 1:]), sampled_l_int


class MLMHead(nn.Module):
    """Shared BERT-style Masked Language Modeling prediction head over codebook V."""

    def __init__(self, cfg: ELTConfig, embed_weight: nn.Parameter | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.bias = nn.Parameter(torch.zeros(cfg.codebook_size))
        if cfg.tie_embeddings and embed_weight is not None:
            self.decoder: nn.Linear | None = None
            self._tied_embed = embed_weight
        else:
            self.decoder = nn.Linear(cfg.d_model, cfg.codebook_size, bias=False)
            self._tied_embed = None

    def forward(
        self, x: torch.Tensor, embed_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm(F.gelu(self.dense(x)))
        if self.decoder is not None:
            return self.decoder(h) + self.bias
        weight = embed_weight if embed_weight is not None else self._tied_embed
        if weight is None:
            raise RuntimeError("MLMHead initialized with tie_embeddings=True requires embed_weight.")
        return F.linear(h, weight[: self.cfg.codebook_size], self.bias)


MaskGITELT = ELTModel
TransformerBlock = BERTBlock
