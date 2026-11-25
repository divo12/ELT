from __future__ import annotations

from dataclasses import dataclass

from elt.core import DIT_SCALES


@dataclass(frozen=True)
class DiffusionELTConfig:
    """Paper-complete Diffusion Transformer ELT configuration (Table 5 & Table 7)."""

    # Architecture hyperparameters (default: Table 7 Large / 8N x 4L)
    d_model: int = 2048
    n_heads: int = 16
    d_ff: int = 8192
    dropout: float = 0.0
    n_unique_layers: int = 8
    L_max: int = 4
    L_min: int = 1
    grad_checkpoint: bool = False

    # VAE latent space & patching (ImageNet 256x256 -> SD-v1.4 32x32x4)
    latent_channels: int = 4
    latent_size: int = 32
    patch_size: int = 2
    num_classes: int = 1000
    label_drop_prob: float = 0.1

    # Shifted cosine v-prediction + Sigmoid-ELBO schedule (Table 5, Simpler Diffusion)
    noise_shift: float = 1.0
    sigmoid_bias: float = 0.0

    # Training hyperparameters (Table 5)
    batch_size: int = 512
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    total_epochs: int = 200

    # Distillation curriculum
    lambda_start: float = 1.0
    lambda_end: float = 0.0

    # Sampling (Table 5)
    sampling_steps: int = 512
    cfg_scale: float = 3.0

    @property
    def grid_size(self) -> int:
        if self.latent_size % self.patch_size != 0:
            raise ValueError(
                f"latent_size ({self.latent_size}) must be divisible by patch_size ({self.patch_size})"
            )
        return self.latent_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.latent_channels

    @property
    def null_class_id(self) -> int:
        return self.num_classes

    @classmethod
    def from_scale(
        cls,
        scale: str = "Large",
        n_unique_layers: int = 8,
        L_max: int = 4,
        **kwargs,
    ) -> DiffusionELTConfig:
        if scale not in DIT_SCALES:
            raise ValueError(f"Unknown DiT scale {scale!r}. Valid: {sorted(DIT_SCALES)}")
        spec = DIT_SCALES[scale]
        return cls(
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            d_ff=spec.mlp_dim,
            n_unique_layers=n_unique_layers,
            L_max=L_max,
            **kwargs,
        )
