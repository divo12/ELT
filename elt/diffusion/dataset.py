from __future__ import annotations

import glob
import io

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from elt.diffusion.vae import vae_encode


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


class LatentDataset(Dataset):
    def __init__(self, dataset: Dataset, vae, device: torch.device, batch_size: int = 32):
        temporary_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_latents = []
        all_labels = []
        for image, labels in tqdm(temporary_dataloader, desc="Encoding VAE latents"):
            image = image.to(device)
            latent = vae_encode(vae, image)
            all_latents.append(latent.cpu())
            all_labels.append(labels.cpu())
        self.latents = torch.cat(all_latents, dim=0)
        self.labels = torch.cat(all_labels, dim=0)

    def __getitem__(self, idx: int):
        return self.latents[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.latents)


class CachedLatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, labels: torch.Tensor):
        self.latents = latents
        self.labels = labels

    def __getitem__(self, idx: int):
        return self.latents[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.latents)
