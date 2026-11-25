from __future__ import annotations

from dataclasses import dataclass
import math

from elt.core import HIGH_WD_MASKGIT_SCALES, MASKGIT_SCALES


@dataclass(frozen=True)
class ELTConfig:
    """Paper-complete Masked Generative Transformer ELT config (Tables 5 & 6: MaskGIT & MAGVIT)."""

    # Architecture hyperparameters (default: Table 6 Scale L, 8N x 3L/4L)
    d_model: int = 1024
    n_heads: int = 16
    d_ff: int = 4096
    dropout: float = 0.0
    n_unique_layers: int = 8
    grad_checkpoint: bool = False
    tie_embeddings: bool = True

    # Discrete token space: (1, 16, 16) for ImageNet 256x256; (4, 16, 16) for UCF-101 16x128x128
    codebook_size: int = 1024
    grid_shape: tuple[int, int, int] = (1, 16, 16)
    num_classes: int = 1000

    # Loop hyperparameters
    L_max: int = 4
    L_min: int = 1

    # Distillation curriculum
    lambda_start: float = 1.0
    lambda_end: float = 0.0

    # Training hyperparameters (Table 5 ImageNet defaults)
    batch_size: int = 512
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.96)
    weight_decay: float = 4.5e-2
    warmup_steps: int = 15000
    total_epochs: int = 270
    label_drop_prob: float = 0.1

    # Inference hyperparameters (Table 5 & Eq. 2)
    sampling_steps: int = 24
    cfg_scale: float = 3.0
    temp_bias: float = 0.5
    temp_scale: float = 0.8

    @property
    def max_seq_len(self) -> int:
        return math.prod(self.grid_shape)

    @property
    def vocab_size(self) -> int:
        """Total input token embedding table size (codebook + [MASK])."""
        return self.codebook_size + 1

    @property
    def mask_token_id(self) -> int:
        return self.codebook_size

    @property
    def null_class_id(self) -> int:
        return self.num_classes

    @classmethod
    def from_scale(
        cls,
        scale: str = "L",
        n_unique_layers: int = 8,
        L_max: int = 4,
        **kwargs,
    ) -> ELTConfig:
        """Instantiate from Table 6 scale (S, B, L, XL, H, g, G).

        Per Appendix A, weight_decay is multiplied by 5 for scales >= H unless overridden.
        """
        if scale not in MASKGIT_SCALES:
            raise ValueError(f"Unknown MaskGIT scale {scale!r}. Valid: {sorted(MASKGIT_SCALES)}")
        spec = MASKGIT_SCALES[scale]
        if "weight_decay" not in kwargs and scale in HIGH_WD_MASKGIT_SCALES:
            kwargs["weight_decay"] = 4.5e-2 * 5.0
        return cls(
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            d_ff=spec.mlp_dim,
            n_unique_layers=n_unique_layers,
            L_max=L_max,
            **kwargs,
        )

    @classmethod
    def for_ucf101(
        cls,
        scale: str = "L",
        n_unique_layers: int = 6,
        L_max: int = 4,
        **kwargs,
    ) -> ELTConfig:
        """Instantiate UCF-101 MAGVIT video configuration (Table 4 & Table 5: 6N x 4L = ~76M params)."""
        defaults = dict(
            grid_shape=(4, 16, 16),
            num_classes=101,
            batch_size=256,
            total_epochs=2000,
            label_drop_prob=0.0,
            sampling_steps=12,
            cfg_scale=1.0,
        )
        defaults.update(kwargs)
        return cls.from_scale(scale=scale, n_unique_layers=n_unique_layers, L_max=L_max, **defaults)
