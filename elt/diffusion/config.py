from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionELTConfig:
    # Architecture hyperparameters
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    n_unique_layers: int = 3
    L_max: int = 4
    L_min: int = 1

    # Diffusion schedule
    T: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # VAE latent space
    latent_channels: int = 4
    latent_size: int = 32

    # Training hyperparameters
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 4.5e-2
    warmup_steps: int = 10000
    total_epochs: int = 500

    # Distillation curriculum
    lambda_start: float = 1.0
    lambda_end: float = 0.0

    # Sampling
    sampling_steps: int = 512

    @property
    def latent_seq_len(self) -> int:
        return self.latent_size * self.latent_size

    @property
    def patch_dim(self) -> int:
        return self.latent_channels
