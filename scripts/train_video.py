from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from elt.core import MASKGIT_SCALES, linear_curriculum
from elt.maskgit.dataset import random_masking
from elt.maskgit.losses import ILSD_train_step
from elt.maskgit.model import ELTModel, MLMHead
from elt.video import (
    CachedVideoTokenDataset,
    UCF101Dataset,
    VideoELTConfig,
    VideoTokenDataset,
    load_video_tokenizer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MAGVIT Video ELT on UCF-101 (Table 4 & Table 5)")
    p.add_argument("--data-dir", type=str, default="./data/ucf101")
    p.add_argument("--magvit-ckpt", type=str, default="./ckpts/magvit-ucf101-f4x8x8.pt")
    p.add_argument("--cache-path", type=str, default="./cache/ucf101_tokens_train.pt")
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints/video")
    p.add_argument("--scale", type=str, default="L", choices=sorted(MASKGIT_SCALES))
    p.add_argument("--n-unique-layers", type=int, default=6)
    p.add_argument("--l-max", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-ilsd", action="store_true")
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
        local_rank = rank = 0
        world_size = 1
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    cfg = VideoELTConfig(
        scale=args.scale,
        n_unique_layers=args.n_unique_layers,
        L_max=args.l_max,
        batch_size=args.batch_size,
        total_epochs=args.epochs,
        grad_checkpoint=args.grad_checkpoint,
    )

    Path(args.cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    if rank == 0 and not os.path.exists(args.cache_path):
        print("rank 0: encoding UCF-101 videos with MAGVIT 3D tokenizer...")
        tokenizer = load_video_tokenizer(device, ckpt_path=args.magvit_ckpt)
        raw_dataset = UCF101Dataset(args.data_dir, num_frames=16, frame_size=128)
        tok_dataset = VideoTokenDataset(raw_dataset, tokenizer, device)
        torch.save({"tokens": tok_dataset.tokens, "labels": tok_dataset.labels}, args.cache_path)
        print("rank 0: video token cache saved")

    if is_ddp:
        dist.barrier()

    cache = torch.load(args.cache_path, map_location="cpu")
    train_dataset = CachedVideoTokenDataset(cache["tokens"], cache["labels"])
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    base_model = ELTModel(cfg).to(device)
    base_head = MLMHead(cfg, embed_weight=base_model.token_embedding.weight).to(device)
    model: torch.nn.Module = DDP(base_model, device_ids=[local_rank]) if is_ddp else base_model
    head: torch.nn.Module = DDP(base_head, device_ids=[local_rank]) if is_ddp else base_head

    seen = {id(p): p for p in list(base_model.parameters()) + list(base_head.parameters())}
    optimizer = torch.optim.AdamW(
        list(seen.values()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
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

        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device, dtype=torch.long)
            lambda_val = linear_curriculum(
                global_step, total_steps, cfg.lambda_start, cfg.lambda_end
            )
            masked_tokens, targets, mask = random_masking(batch_tokens, cfg)

            loss, l_int = ILSD_train_step(
                model,
                head,
                masked_tokens,
                targets,
                mask,
                lambda_val,
                cfg,
                labels=batch_labels,
                use_ilsd=(not args.no_ilsd),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if rank == 0 and global_step % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.total_epochs} "
                    f"Step {global_step} "
                    f"Loss {loss.item():.4f} "
                    f"lambda {lambda_val:.4f} "
                    f"L_int {l_int}"
                )

        if rank == 0 and epoch % 50 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "config": cfg,
                    "model": base_model.state_dict(),
                    "head": base_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                },
                os.path.join(args.ckpt_dir, f"checkpoint_video_epoch_{epoch}.pt"),
            )

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
