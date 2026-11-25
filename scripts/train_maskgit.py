import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

from elt.maskgit.config import ELTConfig
from elt.maskgit.dataset import CachedTokenDataset, ImageNet100Dataset, TokenDataset, random_masking
from elt.maskgit.losses import ILSD_train_step
from elt.maskgit.model import ELTModel, MLMHead
from elt.maskgit.tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaskGIT-based ELT")
    p.add_argument("--data-dir", type=str, default="./data/imagenet100")
    p.add_argument("--vqgan-ckpt", type=str, default="./ckpts/maskgit-vqgan-imagenet-f16-256.bin")
    p.add_argument("--cache-path", type=str, default="./cache/imagenet100_tokens_train.pt")
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints/maskgit")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=270)
    p.add_argument("--n-unique-layers", type=int, default=3)
    p.add_argument("--l-max", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    cfg = ELTConfig(
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_unique_layers=args.n_unique_layers,
        L_max=args.l_max,
        batch_size=args.batch_size,
        total_epochs=args.epochs,
    )

    Path(args.cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    if rank == 0 and not os.path.exists(args.cache_path):
        print("rank 0: encoding dataset...")
        vqgan = load_tokenizer(device, ckpt_path=args.vqgan_ckpt)
        train_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = ImageNet100Dataset(args.data_dir, transform=train_transforms)
        token_dataset = TokenDataset(train_dataset, vqgan, device)
        torch.save(
            {"tokens": token_dataset.tokens, "labels": token_dataset.labels},
            args.cache_path,
        )
        print("rank 0: cache saved")

    if is_ddp:
        dist.barrier()

    cache = torch.load(args.cache_path, map_location="cpu")
    train_token_dataset = CachedTokenDataset(cache["tokens"], cache["labels"])
    train_sampler = (
        DistributedSampler(train_token_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    train_loader = DataLoader(
        train_token_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model: torch.nn.Module = ELTModel(cfg).to(device)
    head: torch.nn.Module = MLMHead(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        head = DDP(head, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.96),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: step / cfg.warmup_steps if step < cfg.warmup_steps else 1.0,
    )

    global_step = 0
    total_steps = max(1, cfg.total_epochs * len(train_loader))

    for epoch in range(cfg.total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        head.train()

        for batch_tokens, _ in train_loader:
            batch_tokens = batch_tokens.to(device)
            lambda_val = max(0.0, 1.0 - global_step / total_steps)
            masked_tokens, _, mask = random_masking(batch_tokens, cfg)

            loss, l_int = ILSD_train_step(
                model, head, masked_tokens, batch_tokens, mask, lambda_val, cfg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if rank == 0 and global_step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.total_epochs} "
                    f"Step {global_step} "
                    f"Loss {loss.item():.4f} "
                    f"lambda {lambda_val:.4f} "
                    f"L_int {l_int}"
                )

        if rank == 0 and epoch % 10 == 0:
            raw_model = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            raw_head = head.module.state_dict() if hasattr(head, "module") else head.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model": raw_model,
                    "head": raw_head,
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                },
                os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch}.pt"),
            )

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
