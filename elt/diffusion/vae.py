from __future__ import annotations

from typing import Any
import torch


def load_vae(device: torch.device, model_id: str = "CompVis/stable-diffusion-v1-4") -> Any:
    from diffusers import AutoencoderKL

    vae_model = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae_model.to(device)
    for parameter in vae_model.parameters():
        parameter.requires_grad = False
    vae_model.eval()
    return vae_model


def vae_encode(vae: Any, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        encoding = vae.encode(images).latent_dist.sample()
        return encoding * 0.18215


def vae_decode(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return vae.decode(latents / 0.18215).sample
