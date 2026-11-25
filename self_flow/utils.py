"""Self-Flow token & coordinate utilities (adapted from black-forest-labs/Self-Flow, Apache-2.0)."""

from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch import Tensor

Axes = Tuple[Literal["t", "h", "w", "l"], ...]


def prc_vid(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    c, t, h, w = x.shape
    dev = x.device
    if t_coord is None:
        t_coord = torch.arange(t, device=dev)
    x_coords = {
        "t": t_coord.to(dev),
        "h": torch.arange(h, device=dev),
        "w": torch.arange(w, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x_tokens = x.permute(1, 2, 3, 0).reshape(t * h * w, c)
    return x_tokens, x_ids


def prc_img(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    c, h, w = x.shape
    dev = x.device
    x_coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(h, device=dev),
        "w": torch.arange(w, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x_tokens = x.permute(1, 2, 0).reshape(h * w, c)
    return x_tokens, x_ids


def prc_txt(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txts"
    l_len, _ = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(l_len, device=dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids


def prc_txts(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txts"
    t, l_len, c = x.shape
    dev = x.device
    if t_coord is None:
        t_coord = torch.arange(t, device=dev)
    coords = {
        "t": t_coord.to(dev),
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(l_len, device=dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x.reshape(t * l_len, c), x_ids


def batched_wrapper(fn):
    def batched_prc(
        x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        results = [
            fn(
                x[i],
                t_coord[i] if t_coord is not None else None,
                l_coord[i] if l_coord is not None else None,
            )
            for i in range(len(x))
        ]
        xs, ids = zip(*results)
        return torch.stack(xs), torch.stack(ids)

    return batched_prc


batched_prc_vid = batched_wrapper(prc_vid)
batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)
batched_prc_txts = batched_wrapper(prc_txts)


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = int(torch.max(t_ids).item())
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique = torch.unique(t_ids, sorted=True)
    t_remap[t_unique] = torch.arange(len(t_unique), device=t_ids.device, dtype=t_ids.dtype)
    return t_remap[t_ids]


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)
        t = int(torch.max(t_ids_cmpr).item()) + 1
        h = int(torch.max(h_ids).item()) + 1
        w = int(torch.max(w_ids).item()) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids
        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        # (t*h*w, c) -> (1, c, t, h, w)
        x_list.append(out.view(t, h, w, ch).permute(3, 0, 1, 2).unsqueeze(0).contiguous())
    return x_list


def scattercat(x: Tensor, x_ids: Tensor) -> Tensor:
    return torch.cat(scatter_ids(x, x_ids), dim=0).squeeze(2)


def patchify_latents(latents: Tensor, patch_size: int = 2) -> Tensor:
    """Convert (B, C, H, W) latents to BFL token sequence (B, (H/p)*(W/p), C*p*p)."""
    b, c, h, w = latents.shape
    p = patch_size
    gh, gw = h // p, w // p
    # Match rearrange(latents, "b c (h p1) (w p2) -> b (h w) (c p1 p2)")
    patched = latents.view(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).contiguous()
    return patched.view(b, gh * gw, c * p * p)


def unpatchify_latents(tokens: Tensor, patch_size: int = 2, out_channels: int = 4) -> Tensor:
    """Convert BFL token sequence (B, (H/p)*(W/p), C*p*p) back to (B, C, H, W)."""
    b, n, _ = tokens.shape
    p = patch_size
    g = int(round(n**0.5))
    x = tokens.view(b, g, g, out_channels, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(b, out_channels, g * p, g * p)
