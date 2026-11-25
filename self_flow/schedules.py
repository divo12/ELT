"""Self-Flow Timestep Distributions & Dual-Timestep Scheduling (arXiv:2603.06507 Sec 3.3 & App A.2/C)."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

TimestepDist = Literal["uniform", "shifted_uniform", "logit_normal", "plateau_logit_normal"]

# Paper Appendix A.3 modality mask ratios R_M
MODALITY_MASK_RATIOS: dict[str, float] = {
    "image": 0.25,
    "video": 0.10,
    "audio": 0.50,
}


def timeshift(t: Tensor, alpha: float = 1.0) -> Tensor:
    """Eq. 8: s(alpha, t) = (alpha * t) / (1 + (alpha - 1) * t)."""
    if alpha == 1.0:
        return t
    return (alpha * t) / (1.0 + (alpha - 1.0) * t)


def _plateau_logit_normal(
    shape: tuple[int, ...],
    mu: float = 0.0,
    sigma: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Appendix A.2 Fig. 12(d): Logit-Normal PDF held constant past its mode up to t=1."""
    # Mode of p_ln(t; mu, sigma) solves logit(t) = mu + sigma^2 * (2t - 1); for mu=0, sigma=1 mode is 0.5.
    t_grid = torch.linspace(1e-4, 1.0 - 1e-4, 2048, device=device)
    logit_t = torch.log(t_grid / (1.0 - t_grid))
    pdf = torch.exp(-0.5 * ((logit_t - mu) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi) * t_grid * (1.0 - t_grid)
    )
    mode_idx = int(torch.argmax(pdf).item())
    pdf[mode_idx:] = pdf[mode_idx]
    cdf = torch.cumsum(pdf, dim=0)
    cdf = cdf / cdf[-1]
    u = torch.rand(shape, device=device)
    idx = torch.searchsorted(cdf, u.reshape(-1)).clamp(0, len(t_grid) - 1)
    return t_grid[idx].view(shape)


def sample_timesteps(
    shape: tuple[int, ...],
    dist: TimestepDist = "uniform",
    shift_alpha: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    high_noise_prob: float = 0.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample training timesteps t ~ p(t) in (0, 1) per Appendix A.2 & Appendix C."""
    if dist == "uniform":
        t = torch.rand(shape, device=device)
    elif dist == "shifted_uniform":
        t = timeshift(torch.rand(shape, device=device), alpha=shift_alpha)
    elif dist == "logit_normal":
        z = torch.randn(shape, device=device) * sigma + (mu + math.log(shift_alpha))
        t = torch.sigmoid(z)
    elif dist == "plateau_logit_normal":
        t = timeshift(_plateau_logit_normal(shape, mu=mu, sigma=sigma, device=device), alpha=shift_alpha)
    else:
        raise ValueError(f"Unknown timestep distribution: {dist!r}")

    # Appendix C: optional 5% uniform sampling on [0.95, 1.0] for video low-SNR coverage
    if high_noise_prob > 0.0:
        high_mask = torch.rand(shape, device=device) < high_noise_prob
        high_t = 0.95 + 0.05 * torch.rand(shape, device=device)
        t = torch.where(high_mask, high_t, t)

    return t.clamp(1e-5, 1.0 - 1e-5)


def dual_timestep_schedule(
    x0: Tensor,
    x1: Tensor | None = None,
    mask_ratio: float = 0.25,
    dist: TimestepDist = "uniform",
    shift_alpha: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    high_noise_prob: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Section 3.3 (Eq. 4 & Eq. 5): Dual-Timestep Scheduling.

    Args:
        x0: Clean token sequence of shape (B, N, C).
        x1: Standard Gaussian noise of shape (B, N, C) (sampled if None).
        mask_ratio: R_M <= 0.5 (0.25 image, 0.10 video, 0.50 audio).

    Returns:
        x_tau:     Heterogeneously noised student input (B, N, C).
        x_tau_min: Cleaner homogeneously noised teacher input (B, N, C) at tau_min = min(t, s).
        tau:       Per-token dual-timestep vector (B, N).
        tau_min:   Cleaner scalar timestep per sample (B,).
        mask:      Boolean mask M of shape (B, N) where tau_i = s.
    """
    if not (0.0 <= mask_ratio <= 0.5):
        raise ValueError(f"Paper requires mask_ratio R_M in [0, 0.5], got {mask_ratio}")
    if x1 is None:
        x1 = torch.randn_like(x0)

    b, n, _ = x0.shape
    device = x0.device

    t = sample_timesteps(
        (b,), dist=dist, shift_alpha=shift_alpha, mu=mu, sigma=sigma,
        high_noise_prob=high_noise_prob, device=device,
    )
    s = sample_timesteps(
        (b,), dist=dist, shift_alpha=shift_alpha, mu=mu, sigma=sigma,
        high_noise_prob=high_noise_prob, device=device,
    )

    mask = torch.rand(b, n, device=device) < mask_ratio
    tau = torch.where(mask, s.unsqueeze(1).expand(b, n), t.unsqueeze(1).expand(b, n))
    tau_min = torch.minimum(t, s)

    tau_exp = tau.unsqueeze(-1).to(x0.dtype)
    tau_min_exp = tau_min.view(b, 1, 1).to(x0.dtype)

    x_tau = (1.0 - tau_exp) * x0 + tau_exp * x1
    x_tau_min = (1.0 - tau_min_exp) * x0 + tau_min_exp * x1
    return x_tau, x_tau_min, tau, tau_min, mask
