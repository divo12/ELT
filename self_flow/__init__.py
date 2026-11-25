"""Self-Flow: Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis (arXiv:2603.06507).

Combines the official Black Forest Labs inference codebase (`black-forest-labs/Self-Flow`, Apache-2.0)
with the paper-complete Dual-Timestep Scheduling, EMA self-distillation objective, ODE/SDE timeshifted
samplers, and Multi-Modal adapter architecture.
"""

from self_flow.losses import (
    EMATeacher,
    SelfFlowConfig,
    cosine_alignment_loss,
    multimodal_self_flow_step,
    self_flow_train_step,
)
from self_flow.model import (
    SIT_SCALES,
    MultiModalSelfFlow,
    PerTokenDiTBlock,
    PerTokenFinalLayer,
    SelfFlowDiT,
    SelfFlowPerTokenDiT,
    SimpleHead,
)
from self_flow.sampling import denoise_loop, sample_ode
from self_flow.schedules import (
    MODALITY_MASK_RATIOS,
    dual_timestep_schedule,
    sample_timesteps,
    timeshift,
)
from self_flow.utils import patchify_latents, unpatchify_latents

__all__ = [
    "EMATeacher",
    "MODALITY_MASK_RATIOS",
    "MultiModalSelfFlow",
    "PerTokenDiTBlock",
    "PerTokenFinalLayer",
    "SIT_SCALES",
    "SelfFlowConfig",
    "SelfFlowDiT",
    "SelfFlowPerTokenDiT",
    "SimpleHead",
    "cosine_alignment_loss",
    "denoise_loop",
    "dual_timestep_schedule",
    "multimodal_self_flow_step",
    "patchify_latents",
    "sample_ode",
    "sample_timesteps",
    "self_flow_train_step",
    "timeshift",
    "unpatchify_latents",
]
