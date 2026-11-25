"""MAGVIT-style Causal 3D VQGAN video tokenizer (Yu et al., CVPR 2023; ELT Section 5.1/5.3).

Compresses (B, 3, 16, 128, 128) RGB videos by (4, 8, 8) spatiotemporally into
(B, 4, 16, 16) = 1024 discrete codebook tokens with codebook_size=1024.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 1):
        if channels % g == 0:
            return g
    return 1


class CausalConv3d(nn.Module):
    """Temporally causal 3D convolution (pads past frames only, symmetric spatial padding)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kt = kh = kw = kernel_size
        else:
            kt, kh, kw = kernel_size
        self.pad = (kw // 2, kw // 2, kh // 2, kh // 2, kt - 1, 0)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=(kt, kh, kw), stride=stride, padding=0, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.pad, mode="replicate"))


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, bias=False)
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, bias=False)
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + residual


class DownStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock3D(in_channels, out_channels), ResBlock3D(out_channels, out_channels)]
        )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.stride is not None:
            x = F.avg_pool3d(x, kernel_size=self.stride, stride=self.stride)
        return x


class UpStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock3D(in_channels, out_channels), ResBlock3D(out_channels, out_channels)]
        )
        self.scale_factor = scale_factor
        self.upsample_conv = (
            CausalConv3d(out_channels, out_channels, kernel_size=3)
            if scale_factor is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.scale_factor is not None and self.upsample_conv is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
            x = self.upsample_conv(x)
        return x


class VectorQuantizer3D(nn.Module):
    def __init__(self, codebook_size: int = 1024, embed_dim: int = 256) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, t, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, c)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * z_flat @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1).view(b, t, h, w)
        quantized = self.embedding(indices).permute(0, 4, 1, 2, 3).contiguous()
        if self.training:
            quantized = z + (quantized - z).detach()
        return indices, quantized


class MAGVITEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, embed_dim: int = 256) -> None:
        super().__init__()
        ch = base_channels
        self.conv_in = CausalConv3d(in_channels, ch, kernel_size=3, bias=False)
        self.down = nn.ModuleList(
            [
                DownStage3D(ch, ch, stride=(1, 2, 2)),
                DownStage3D(ch, ch * 2, stride=(2, 2, 2)),
                DownStage3D(ch * 2, ch * 4, stride=(2, 2, 2)),
                DownStage3D(ch * 4, ch * 4, stride=None),
            ]
        )
        self.mid = nn.ModuleList([ResBlock3D(ch * 4, ch * 4), ResBlock3D(ch * 4, ch * 4)])
        self.norm_out = nn.GroupNorm(_group_norm_groups(ch * 4), ch * 4)
        self.conv_out = nn.Conv3d(ch * 4, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for stage in self.down:
            x = stage(x)
        for block in self.mid:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class MAGVITDecoder(nn.Module):
    def __init__(
        self, out_channels: int = 3, base_channels: int = 64, embed_dim: int = 256
    ) -> None:
        super().__init__()
        ch = base_channels
        self.conv_in = CausalConv3d(embed_dim, ch * 4, kernel_size=3)
        self.mid = nn.ModuleList([ResBlock3D(ch * 4, ch * 4), ResBlock3D(ch * 4, ch * 4)])
        self.up = nn.ModuleList(
            [
                UpStage3D(ch * 4, ch * 4, scale_factor=None),
                UpStage3D(ch * 4, ch * 2, scale_factor=(2.0, 2.0, 2.0)),
                UpStage3D(ch * 2, ch, scale_factor=(2.0, 2.0, 2.0)),
                UpStage3D(ch, ch, scale_factor=(1.0, 2.0, 2.0)),
            ]
        )
        self.norm_out = nn.GroupNorm(_group_norm_groups(ch), ch)
        self.conv_out = CausalConv3d(ch, out_channels, kernel_size=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        for block in self.mid:
            x = block(x)
        for stage in self.up:
            x = stage(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class MAGVITTokenizer(nn.Module):
    """3D VQGAN Video Tokenizer mapping (B, 3, 16, 128, 128) <-> (B, 4, 16, 16) tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        embed_dim: int = 256,
        codebook_size: int = 1024,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.encoder = MAGVITEncoder(in_channels, base_channels, embed_dim)
        self.quantize = VectorQuantizer3D(codebook_size, embed_dim)
        self.decoder = MAGVITDecoder(in_channels, base_channels, embed_dim)

    def encode(self, videos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(self.encoder(videos))

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)

    def decode_tokens(
        self, indices: torch.Tensor, grid_shape: tuple[int, int, int] = (4, 16, 16)
    ) -> torch.Tensor:
        b = indices.shape[0]
        idx_3d = indices.clamp(0, self.codebook_size - 1).view(b, *grid_shape)
        quantized = self.quantize.embedding(idx_3d).permute(0, 4, 1, 2, 3).contiguous()
        return self.decode(quantized)

    def forward(self, videos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices, quantized = self.encode(videos)
        return self.decode(quantized), indices


def load_video_tokenizer(
    device: torch.device,
    ckpt_path: str | None = None,
    base_channels: int = 64,
) -> MAGVITTokenizer:
    tokenizer = MAGVITTokenizer(base_channels=base_channels)
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        tokenizer.load_state_dict(state_dict, strict=False)
    tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False
    return tokenizer


@torch.no_grad()
def encode_videos(tokenizer: MAGVITTokenizer, videos: torch.Tensor) -> torch.Tensor:
    indices, _ = tokenizer.encode(videos)
    return indices.reshape(videos.shape[0], -1)


@torch.no_grad()
def decode_videos(
    tokenizer: MAGVITTokenizer,
    indices: torch.Tensor,
    grid_shape: tuple[int, int, int] = (4, 16, 16),
) -> torch.Tensor:
    return tokenizer.decode_tokens(indices, grid_shape=grid_shape)
