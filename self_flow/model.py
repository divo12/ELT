"""Self-Flow Transformer Models (adapted from black-forest-labs/Self-Flow, Apache-2.0,
extended with Multi-Modal modality adapters from arXiv:2603.06507 Appendix D).

100% state_dict compatible with official BFL `Hila/Self-Flow` (`selfflow_imagenet256.pt`)
without requiring `timm` or `einops`.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from itertools import repeat
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


@dataclass(frozen=True)
class SiTSpec:
    hidden_size: int
    depth: int
    num_heads: int


SIT_SCALES: dict[str, SiTSpec] = {
    "SiT-S": SiTSpec(hidden_size=384, depth=12, num_heads=6),
    "SiT-B": SiTSpec(hidden_size=768, depth=12, num_heads=12),
    "SiT-L": SiTSpec(hidden_size=1024, depth=24, num_heads=16),
    "SiT-XL": SiTSpec(hidden_size=1152, depth=28, num_heads=16),
}


class Attention(nn.Module):
    """Drop-in replacement for timm.models.vision_transformer.Attention with identical parameter names."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    """Drop-in replacement for timm Mlp with identical parameter names (fc1, fc2)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class PatchedPatchEmbed(nn.Module):
    """Sequence-to-patch linear embedding matching BFL Self-Flow."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_chans: int = 4,
        embed_dim: int = 1152,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        img_size_2d = to_2tuple(img_size)
        self.grid_size = (img_size_2d[0] // self.patch_size[0], img_size_2d[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        patch_dim = self.patch_size[0] * self.patch_size[1] * in_chans
        self.proj = nn.Linear(patch_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_per_token(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    @property
    def null_class_id(self) -> int:
        return self.num_classes if self.dropout_prob > 0 else self.num_classes - 1

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.null_class_id, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0.0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PerTokenDiTBlock(DiTBlock):
    """DiT block handling per-token Dual-Timestep conditioning (B, N, D)."""

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b, seq_len, hidden_dim = c.shape
        modulation = self.adaLN_modulation(c.reshape(-1, hidden_dim)).reshape(b, seq_len, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate_per_token(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate_per_token(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class PerTokenFinalLayer(FinalLayer):
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b, seq_len, hidden_dim = c.shape
        modulation = self.adaLN_modulation(c.reshape(-1, hidden_dim)).reshape(b, seq_len, -1)
        shift, scale = modulation.chunk(2, dim=-1)
        return self.linear(modulate_per_token(self.norm_final(x), shift, scale))


class SimpleHead(nn.Module):
    """Self-distillation projection head h_theta (Eq. 6)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class SelfFlowDiT(nn.Module):
    """Base Self-Flow SiT/DiT model."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1001,
        class_dropout_prob: float = 0.0,
        learn_sigma: bool = True,
        compatibility_mode: bool = True,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.compatibility_mode = compatibility_mode

        self.x_embedder = PatchedPatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.projector = SimpleHead(hidden_size, hidden_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def shufflechannel(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder (b, l, p*q*c) -> (b, l, c*p*q) and strip sigma channels if learn_sigma."""
        b, l_len, _ = x.shape
        p = self.x_embedder.patch_size[0]
        x = x.view(b, l_len, p, p, self.out_channels).permute(0, 1, 4, 2, 3).reshape(b, l_len, -1)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=2)
        return x

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_features: int | bool = False,
        return_raw_features: int | bool = False,
    ):
        assert not (return_raw_features and return_features)
        x = self.x_embedder(x) + self.pos_embed.to(x.dtype)
        c = self.t_embedder(t) + self.y_embedder(y, self.training)

        zs = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if return_features and (i + 1) == int(return_features):
                zs = self.projector(x)
            elif return_raw_features and (i + 1) == int(return_raw_features):
                zs = x

        x = self.final_layer(x, c)
        if return_features or return_raw_features:
            return x, zs
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        vector: torch.Tensor,
        x_ids: torch.Tensor | None = None,
        return_features: int | bool = False,
        return_raw_features: int | bool = False,
    ):
        """Forward pass accepting paper-convention timesteps (0=clean x_0, 1=noise x_1)
        and returning velocity v_hat = x_1 - x_0.
        """
        t_internal = 1.0 - timesteps
        if return_features or return_raw_features:
            out, zs = self._forward(
                x=x,
                t=t_internal,
                y=vector,
                return_features=return_features,
                return_raw_features=return_raw_features,
            )
            return -self.shufflechannel(out), zs
        out = self._forward(x=x, t=t_internal, y=vector)
        return -self.shufflechannel(out)


class SelfFlowPerTokenDiT(SelfFlowDiT):
    """Self-Flow DiT with per-token Dual-Timestep conditioning (main model in paper & BFL repo)."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1001,
        class_dropout_prob: float = 0.0,
        learn_sigma: bool = True,
        compatibility_mode: bool = True,
    ) -> None:
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            compatibility_mode=compatibility_mode,
        )
        out_channels = in_channels * 2 if learn_sigma else in_channels
        new_blocks = nn.ModuleList()
        for orig_block in self.blocks:
            blk = PerTokenDiTBlock(hidden_size, num_heads, mlp_ratio)
            blk.load_state_dict(orig_block.state_dict())
            new_blocks.append(blk)
        self.blocks = new_blocks

        new_final = PerTokenFinalLayer(hidden_size, patch_size, out_channels)
        new_final.load_state_dict(self.final_layer.state_dict())
        self.final_layer = new_final

    @classmethod
    def from_scale(cls, scale: str = "SiT-XL", **kwargs) -> SelfFlowPerTokenDiT:
        if scale not in SIT_SCALES:
            raise ValueError(f"Unknown SiT scale {scale!r}. Valid: {sorted(SIT_SCALES)}")
        spec = SIT_SCALES[scale]
        return cls(
            hidden_size=spec.hidden_size,
            depth=spec.depth,
            num_heads=spec.num_heads,
            **kwargs,
        )

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_features: int | bool = False,
        return_raw_features: int | bool = False,
    ):
        assert not (return_raw_features and return_features)
        x = self.x_embedder(x) + self.pos_embed[:, : x.shape[1]].to(x.dtype)
        b, seq_len, _ = x.shape

        if t.ndim == 1:
            t_emb = self.t_embedder(t).unsqueeze(1).expand(-1, seq_len, -1)
        elif t.ndim == 2:
            t_emb = self.t_embedder(t.reshape(-1)).reshape(b, seq_len, -1)
        else:
            raise ValueError(f"Timesteps must be 1D (B,) or 2D (B, N), got {t.shape}")

        y_emb = self.y_embedder(y, self.training).unsqueeze(1).expand(-1, seq_len, -1)
        c = t_emb + y_emb

        zs = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if return_features and (i + 1) == int(return_features):
                zs = self.projector(x)
            elif return_raw_features and (i + 1) == int(return_raw_features):
                zs = x

        x = self.final_layer(x, c)
        if return_features or return_raw_features:
            return x, zs
        return x


class ModalityStemHead(nn.Module):
    """Per-modality token input/output projections for Multi-Modal Self-Flow (Appendix D)."""

    def __init__(self, token_dim: int, max_tokens: int, hidden_size: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(token_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, hidden_size) * 0.02)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, token_dim)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.in_proj(x) + self.pos_embed[:, : x.shape[1]].to(x.dtype)

    def decode(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b, seq_len, d = c.shape
        shift, scale = self.adaLN_modulation(c.reshape(-1, d)).reshape(b, seq_len, -1).chunk(2, dim=-1)
        return self.out_proj(modulate_per_token(self.norm_final(h), shift, scale))


class MultiModalSelfFlow(nn.Module):
    """Unified Multi-Modal Self-Flow Transformer (Section 4.2 & Appendix D).

    Shares a single per-token transformer trunk and projector across modalities
    (e.g. image=16/128-d, video=48-d, audio=64-d) with lightweight per-modality stems/heads.
    """

    def __init__(
        self,
        modality_specs: dict[str, tuple[int, int]] | None = None,
        hidden_size: int = 1152,
        depth: int = 21,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        cond_dim: int = 1152,
    ) -> None:
        super().__init__()
        if modality_specs is None:
            # Paper Appendix A.2 defaults: (token_dim, max_seq_len)
            modality_specs = {
                "image": (16, 256),
                "video": (48, 3072),
                "audio": (64, 250),
            }
        self.hidden_size = hidden_size
        self.depth = depth
        self.adapters = nn.ModuleDict(
            {
                mod: ModalityStemHead(tok_dim, max_len, hidden_size)
                for mod, (tok_dim, max_len) in modality_specs.items()
            }
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_proj = nn.Linear(cond_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [PerTokenDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.projector = SimpleHead(hidden_size, hidden_size)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN_modulation[-1].weight)
            nn.init.zeros_(blk.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        modality: str,
        cond: torch.Tensor | None = None,
        return_features: int | bool = False,
        return_raw_features: int | bool = False,
    ):
        assert not (return_features and return_raw_features)
        adapter = self.adapters[modality]
        h = adapter.embed(x)
        b, seq_len, _ = h.shape

        t_internal = 1.0 - timesteps
        if t_internal.ndim == 1:
            t_emb = self.t_embedder(t_internal).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            t_emb = self.t_embedder(t_internal.reshape(-1)).reshape(b, seq_len, -1)

        if cond is not None:
            c_emb = self.cond_proj(cond)
            if c_emb.ndim == 2:
                c_emb = c_emb.unsqueeze(1).expand(-1, seq_len, -1)
            c = t_emb + c_emb
        else:
            c = t_emb

        zs = None
        for i, block in enumerate(self.blocks):
            h = block(h, c)
            if return_features and (i + 1) == int(return_features):
                zs = self.projector(h)
            elif return_raw_features and (i + 1) == int(return_raw_features):
                zs = h

        v_pred = -adapter.decode(h, c)
        if return_features or return_raw_features:
            return v_pred, zs
        return v_pred
