from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from elt.video.tokenizer import MAGVITTokenizer, encode_videos


class UCF101Dataset(Dataset):
    """Class-conditional UCF-101 video dataset returning (video_3xTxHxW, label).

    Supports class subdirectories containing .avi/.mp4 videos (read via torchvision.io)
    or pre-extracted frame directories / .pt tensor clips.
    """

    VIDEO_EXTS = {".avi", ".mp4", ".webm", ".mkv", ".mov"}

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 16,
        frame_size: int = 128,
    ) -> None:
        self.root = Path(data_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        if not self.root.exists():
            raise FileNotFoundError(f"UCF-101 directory not found: {data_dir}")

        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.samples: list[tuple[Path, int]] = []
        for cls_dir in class_dirs:
            label = self.class_to_idx[cls_dir.name]
            for item in sorted(cls_dir.iterdir()):
                if item.suffix.lower() in self.VIDEO_EXTS or item.suffix == ".pt" or item.is_dir():
                    self.samples.append((item, label))
        if not self.samples:
            raise FileNotFoundError(f"No video files or frame folders found under {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_temporal_indices(self, total: int) -> torch.Tensor:
        if total >= self.num_frames:
            start = int(torch.randint(0, total - self.num_frames + 1, (1,)).item())
            return torch.arange(start, start + self.num_frames)
        return torch.linspace(0, max(total - 1, 0), self.num_frames).round().long()

    def _load_clip(self, path: Path) -> torch.Tensor:
        if path.suffix == ".pt":
            clip = torch.load(path, map_location="cpu").float()
            if clip.ndim == 4 and clip.shape[-1] == 3:
                clip = clip.permute(3, 0, 1, 2)
            return clip

        if path.is_dir():
            from PIL import Image
            import numpy as np

            frame_paths = sorted(
                p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            idx = self._sample_temporal_indices(len(frame_paths))
            frames = [
                torch.from_numpy(np.array(Image.open(frame_paths[int(i)]).convert("RGB")))
                for i in idx
            ]
            # (T, H, W, C) -> (C, T, H, W) in [-1, 1]
            clip = torch.stack(frames, dim=0).permute(3, 0, 1, 2).float() / 127.5 - 1.0
        else:
            from torchvision.io import read_video

            vframes, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
            idx = self._sample_temporal_indices(vframes.shape[0])
            clip = vframes[idx].permute(1, 0, 2, 3).float() / 127.5 - 1.0

        # Resize spatial dims to (frame_size, frame_size)
        c, t, h, w = clip.shape
        if h != self.frame_size or w != self.frame_size:
            resized = F.interpolate(
                clip.permute(1, 0, 2, 3),
                size=(self.frame_size, self.frame_size),
                mode="bilinear",
                align_corners=False,
            )
            clip = resized.permute(1, 0, 2, 3)
        return clip.contiguous()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        return self._load_clip(path), label


class VideoTokenDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: MAGVITTokenizer,
        device: torch.device,
        batch_size: int = 8,
    ) -> None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_tokens = []
        all_labels = []
        for clips, labels in tqdm(loader, desc="Encoding MAGVIT video tokens"):
            clips = clips.to(device)
            tokens = encode_videos(tokenizer, clips)
            all_tokens.append(tokens.cpu())
            all_labels.append(labels.cpu())
        self.tokens = torch.cat(all_tokens, dim=0)
        self.labels = torch.cat(all_labels, dim=0)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]


class CachedVideoTokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, labels: torch.Tensor) -> None:
        self.tokens = tokens
        self.labels = labels

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]
