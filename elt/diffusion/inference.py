from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from elt.diffusion.config import DiffusionELTConfig
from elt.diffusion.vae import vae_decode


class NoiseScheduler:
    """Shifted cosine continuous-time schedule with v-prediction & Sigmoid-ELBO (Table 5)."""

    def __init__(self, cfg: DiffusionELTConfig) -> None:
        self.cfg = cfg

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = t.float().clamp(1e-4, 1.0 - 1e-4)
        log_cot = torch.log(torch.cos(0.5 * math.pi * t_clamped)) - torch.log(
            torch.sin(0.5 * math.pi * t_clamped)
        )
        return 2.0 * log_cot - 2.0 * math.log(self.cfg.noise_shift)

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lam = self.log_snr(t)
        alpha = torch.sqrt(torch.sigmoid(lam))
        sigma = torch.sqrt(torch.sigmoid(-lam))
        return alpha, sigma, lam

    def sigmoid_weight(self, log_snr: torch.Tensor) -> torch.Tensor:
        """Sigmoid-ELBO weighting w(t) in x0-MSE space (Kingma & Gao 2023; Hoogeboom et al. 2025)."""
        return torch.sigmoid(log_snr - self.cfg.sigmoid_bias)

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Corrupt x0 -> x_t = alpha_t * x0 + sigma_t * eps; return (x_t, alpha_t, sigma_t, log_snr)."""
        if eps is None:
            eps = torch.randn_like(x0)
        alpha, sigma, log_snr = self.alpha_sigma(t)
        b = x0.shape[0]
        a = alpha.view(b, 1, 1, 1).to(x0.dtype)
        s = sigma.view(b, 1, 1, 1).to(x0.dtype)
        return a * x0 + s * eps, a, s, log_snr

    def x0_from_v(
        self, x_t: torch.Tensor, v_pred: torch.Tensor, alpha_t: torch.Tensor, sigma_t: torch.Tensor
    ) -> torch.Tensor:
        return alpha_t * x_t - sigma_t * v_pred

    def ddpm_posterior_step(
        self, x_t: torch.Tensor, v_pred: torch.Tensor, t_curr: torch.Tensor, t_next: torch.Tensor
    ) -> torch.Tensor:
        """Exact Gaussian reverse step q(x_s | x_t, x0_hat) for s = t_next < t_curr."""
        b = x_t.shape[0]
        alpha_t, sigma_t, lam_t = self.alpha_sigma(t_curr)
        a_t = alpha_t.view(b, 1, 1, 1).to(x_t.dtype)
        s_t = sigma_t.view(b, 1, 1, 1).to(x_t.dtype)
        x0_hat = self.x0_from_v(x_t, v_pred, a_t, s_t)

        alpha_s, sigma_s, lam_s = self.alpha_sigma(t_next)
        a_s = alpha_s.view(b, 1, 1, 1).to(x_t.dtype)
        s_s = sigma_s.view(b, 1, 1, 1).to(x_t.dtype)

        expm1_term = (-torch.expm1(lam_t - lam_s)).clamp_min(0.0).view(b, 1, 1, 1).to(x_t.dtype)
        ratio = torch.exp(lam_t - lam_s).view(b, 1, 1, 1).to(x_t.dtype)

        mean = ratio * (a_s / a_t) * x_t + expm1_term * a_s * x0_hat
        std = s_s * torch.sqrt(expm1_term)
        return mean + std * torch.randn_like(x_t)


@torch.no_grad()
def ddpm_sample_latents(
    model: torch.nn.Module,
    noise_scheduler: NoiseScheduler,
    dcfg: DiffusionELTConfig,
    n_loops: int | Sequence[int],
    labels: torch.Tensor | None = None,
    batch_size: int = 1,
    cfg_scale: float | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """512-step DDPM v-prediction ancestral sampler with CFG and Any-Time / dynamic loop budget."""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    if labels is None:
        labels = torch.full((batch_size,), dcfg.null_class_id, dtype=torch.long, device=device)
    else:
        labels = labels.to(device)
        batch_size = labels.shape[0]

    guidance = dcfg.cfg_scale if cfg_scale is None else cfg_scale
    use_cfg = guidance != 1.0 and bool((labels != dcfg.null_class_id).any().item())

    steps = dcfg.sampling_steps
    loop_sched = [n_loops] * steps if isinstance(n_loops, int) else list(n_loops)
    if len(loop_sched) != steps:
        raise ValueError(f"Expected {steps} loop counts, got {len(loop_sched)}")

    x = torch.randn(
        batch_size, dcfg.latent_channels, dcfg.latent_size, dcfg.latent_size, device=device
    )
    ts = torch.linspace(1.0 - 1e-3, 1e-3, steps + 1, device=device)
    null_labels = torch.full_like(labels, dcfg.null_class_id)

    for i in range(steps):
        t_curr = ts[i].expand(batch_size)
        t_next = ts[i + 1].expand(batch_size)
        l_i = loop_sched[i]

        if use_cfg:
            both_x = torch.cat([x, x], dim=0)
            both_t = torch.cat([t_curr, t_curr], dim=0)
            both_lbl = torch.cat([labels, null_labels], dim=0)
            v_cond, v_uncond = model(both_x, both_t, n_loops=l_i, labels=both_lbl).chunk(2, dim=0)
            v_pred = v_uncond + guidance * (v_cond - v_uncond)
        else:
            v_pred = model(x, t_curr, n_loops=l_i, labels=labels)

        if i == steps - 1:
            alpha_t, sigma_t, _ = noise_scheduler.alpha_sigma(t_curr)
            a_t = alpha_t.view(batch_size, 1, 1, 1).to(x.dtype)
            s_t = sigma_t.view(batch_size, 1, 1, 1).to(x.dtype)
            return noise_scheduler.x0_from_v(x, v_pred, a_t, s_t)

        x = noise_scheduler.ddpm_posterior_step(x, v_pred, t_curr, t_next)

    return x


@torch.no_grad()
def ddpm_generate(
    model: torch.nn.Module,
    noise_scheduler: NoiseScheduler,
    dcfg: DiffusionELTConfig,
    n_loops: int | Sequence[int],
    device: torch.device,
    vae,
    class_label: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    labels = (
        torch.tensor([class_label], dtype=torch.long, device=device)
        if class_label is not None
        else None
    )
    latents = ddpm_sample_latents(
        model,
        noise_scheduler,
        dcfg,
        n_loops=n_loops,
        labels=labels,
        batch_size=1,
        cfg_scale=cfg_scale,
        device=device,
        seed=seed,
    )
    return vae_decode(vae, latents)


def diffusion_any_time_inference(
    model: torch.nn.Module,
    noise_scheduler: NoiseScheduler,
    dcfg: DiffusionELTConfig,
    device: torch.device,
    vae,
    loop_counts: tuple[int, ...] = (1, 2, 3, 4),
    class_label: int | None = 0,
    cfg_scale: float | None = None,
    seed: int = 42,
):
    model.eval()
    generated_images = []
    for n_loops in loop_counts:
        image = ddpm_generate(
            model,
            noise_scheduler,
            dcfg,
            n_loops=n_loops,
            device=device,
            vae=vae,
            class_label=class_label,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        image = image.squeeze(0).permute(1, 2, 0)
        image = ((image + 1.0) / 2.0).cpu().numpy()
        generated_images.append(np.clip(image, 0.0, 1.0))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(loop_counts), figsize=(4 * len(loop_counts), 4))
    if len(loop_counts) == 1:
        axes = [axes]
    fig.suptitle("Any-Time Inference for Diffusion — more loops = better quality", fontsize=14)
    for ax, img, n_loops in zip(axes, generated_images, loop_counts):
        ax.imshow(img)
        ax.set_title(f"L={n_loops}")
        ax.axis("off")

    plt.tight_layout()
    return fig
