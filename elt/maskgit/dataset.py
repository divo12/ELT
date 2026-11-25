from __future__ import annotations

import glob
import io
import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from elt.maskgit.config import ELTConfig
from elt.maskgit.tokenizer import encode


def random_masking(
    tokens: torch.Tensor, cfg: ELTConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact MaskGIT/MAGVIT per-sample cosine masking schedule."""
    tokens_flat = tokens.reshape(tokens.shape[0], -1)
    b, m = tokens_flat.shape
    u = torch.rand(b, device=tokens.device)
    mask_ratio = torch.cos(0.5 * math.pi * u)
    num_masked = (mask_ratio * m).ceil().clamp(1, m).long()

    rand_scores = torch.rand(b, m, device=tokens.device)
    ranks = rand_scores.argsort(dim=-1).argsort(dim=-1)
    mask = ranks < num_masked.unsqueeze(1)
    masked_tokens = torch.where(mask, cfg.mask_token_id, tokens_flat)
    return masked_tokens, tokens_flat, mask


class ImageNet100Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        import pandas as pd

        parquet_files = sorted(glob.glob(f"{data_dir}/train-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No train-*.parquet files found in {data_dir}")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)
        self.transform = transform
        print(f"loaded {len(self.df)} images")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.df.iloc[idx]
        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


class TokenDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer, device: torch.device, batch_size: int = 32):
        temporary_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_token_ids = []
        all_labels = []
        for images, labels in tqdm(temporary_dataloader, desc="Encoding VQGAN tokens"):
            images = images.to(device)
            indices = encode(tokenizer, images)
            all_token_ids.append(indices.cpu())
            all_labels.append(labels.cpu())
        self.tokens = torch.cat(all_token_ids, dim=0)
        self.labels = torch.cat(all_labels, dim=0)

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.tokens)


class CachedTokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, labels: torch.Tensor):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.tokens)
