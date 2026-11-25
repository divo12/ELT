from dataclasses import dataclass


@dataclass(frozen=True)
class ELTConfig:
    # Architecture hyperparameters
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    n_unique_layers: int = 4
    max_seq_len: int = 256
    vocab_size: int = 1025

    # Loop hyperparameters
    L_max: int = 4
    L_min: int = 1

    # Distillation curriculum
    lambda_start: float = 1.0
    lambda_end: float = 0.0

    # Training hyperparameters
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 4.5e-2
    warmup_steps: int = 15000
    total_epochs: int = 270
    label_drop_prob: float = 0.1

    # Inference hyperparameters
    sampling_steps: int = 24
    cfg_scale: float = 3.0

    @property
    def mask_token_id(self) -> int:
        return self.vocab_size - 1
