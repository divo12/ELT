import glob
import io
import math
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from elt.maskgit.config import ELTConfig
from elt.maskgit.tokenizer import encode


def random_masking(
    tokens: torch.Tensor, cfg: ELTConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, m = tokens.shape
    u = torch.rand(b, 1, device=tokens.device)
    mask_prob = torch.cos(0.5 * math.pi * u)
    mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    # Guarantee at least one masked token per batch so cross_entropy is well-defined
    if not mask.any():
        mask[0, 0] = True
    masked_tokens = torch.where(mask, cfg.mask_token_id, tokens)
    return masked_tokens, tokens, mask


class ImageNet100Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
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
