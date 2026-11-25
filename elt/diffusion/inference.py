from __future__ import annotations

import numpy as np
import torch

from elt.diffusion.config import DiffusionELTConfig
from elt.diffusion.vae import vae_decode


class NoiseScheduler:
    def __init__(self, cfg: DiffusionELTConfig):
        self.cfg = cfg
        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        t_cpu = t.cpu()
        sqrt_ab = self.sqrt_alpha_bars[t_cpu].reshape(-1, 1, 1, 1).to(x0.device)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t_cpu].reshape(-1, 1, 1, 1).to(x0.device)
        return sqrt_ab * x0 + sqrt_omab * epsilon

    def remove_noise(
        self, x_t: torch.Tensor, t: int | torch.Tensor, predicted_epsilon: torch.Tensor
    ) -> torch.Tensor:
        t_cpu = t if isinstance(t, int) else int(t.cpu().item())
        recip = self.sqrt_recip_alphas[t_cpu].to(x_t.device)
        beta = self.betas[t_cpu].to(x_t.device)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t_cpu].to(x_t.device)
        x_previous = recip * (x_t - beta / sqrt_omab * predicted_epsilon)
        if t_cpu > 0:
            x_previous = x_previous + torch.sqrt(beta) * torch.randn_like(x_t)
        return x_previous


def ddpm_generate(
    model: torch.nn.Module,
    noise_scheduler: NoiseScheduler,
    dcfg: DiffusionELTConfig,
    n_loops: int,
    device: torch.device,
    vae,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(1, dcfg.latent_channels, dcfg.latent_size, dcfg.latent_size, device=device)
    with torch.no_grad():
        timesteps = torch.linspace(dcfg.T - 1, 0, dcfg.sampling_steps, dtype=torch.long).tolist()
        for t in timesteps:
            t_batch = torch.tensor([t], device=device)
            predicted_noise = model(x, t_batch, n_loops=n_loops)
            x = noise_scheduler.remove_noise(x, t, predicted_noise)
        image = vae_decode(vae, x)
    return image


def diffusion_any_time_inference(
    model: torch.nn.Module,
    noise_scheduler: NoiseScheduler,
    dcfg: DiffusionELTConfig,
    device: torch.device,
    vae,
    loop_counts: tuple[int, ...] = (1, 2, 3, 4),
):
    model.eval()
    generated_images = []
    for n_loops in loop_counts:
        image = ddpm_generate(model, noise_scheduler, dcfg, n_loops, device, vae)
        image = image.squeeze(0).permute(1, 2, 0)
        image = ((image + 1.0) / 2.0).cpu().numpy()
        generated_images.append(np.clip(image, 0.0, 1.0))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(loop_counts), figsize=(4 * len(loop_counts), 4))
    fig.suptitle("Any-Time Inference for Diffusion — more loops = better quality", fontsize=14)
    for ax, img, n_loops in zip(axes, generated_images, loop_counts):
        ax.imshow(img)
        ax.set_title(f"L={n_loops}")
        ax.axis("off")

    plt.tight_layout()
    return fig
