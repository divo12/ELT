from elt.maskgit.config import ELTConfig
from elt.video.dataset import CachedVideoTokenDataset, UCF101Dataset, VideoTokenDataset
from elt.video.inference import generate_video, video_any_time_inference
from elt.video.tokenizer import (
    MAGVITTokenizer,
    decode_videos,
    encode_videos,
    load_video_tokenizer,
)


def VideoELTConfig(
    scale: str = "L", n_unique_layers: int = 6, L_max: int = 4, **kwargs
) -> ELTConfig:
    """Table 4 & Table 5 UCF-101 MAGVIT ELT configuration factory."""
    return ELTConfig.for_ucf101(scale=scale, n_unique_layers=n_unique_layers, L_max=L_max, **kwargs)


__all__ = [
    "CachedVideoTokenDataset",
    "MAGVITTokenizer",
    "UCF101Dataset",
    "VideoELTConfig",
    "VideoTokenDataset",
    "decode_videos",
    "encode_videos",
    "generate_video",
    "load_video_tokenizer",
    "video_any_time_inference",
]
