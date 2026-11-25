#!/usr/bin/env python3
"""Train Self-Flow (arXiv:2603.06507) with Dual-Timestep Scheduling & EMA Self-Distillation."""

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

from elt.diffusion.dataset import CachedLatentDataset, ImageNet100Dataset, LatentDataset
from elt.diffusion.vae import load_vae
from self_flow.losses import EMATeacher, SelfFlowConfig, self_flow_train_step
from self_flow.model import SIT_SCALES, SelfFlowPerTokenDiT
from self_flow.utils import patchify_latents


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Self-Flow (Dual-Timestep + EMA Self-Distillation)")
    p.add_argument("--data-dir", type=str, default="./data/imagenet100")
    p.add_argument("--cache-path", type=str, default="./cache/imagenet100_latents_train.pt")
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints/self_flow")
    p.add_argument("--scale", type=str, default="SiT-XL", choices=sorted(SIT_SCALES))
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument(
        "--timestep-dist",
        type=str,
        default="uniform",
        choices=["uniform", "shifted_uniform", "logit_normal", "plateau_logit_normal"],
    )
    p.add_argument("--train-shift", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
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

    sf_cfg = SelfFlowConfig(
        mask_ratio=args.mask_ratio,
        gamma=args.gamma,
        timestep_dist=args.timestep_dist,
        train_shift=args.train_shift,
        lr=args.lr,
    )

    Path(args.cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    if rank == 0 and not os.path.exists(args.cache_path):
        import torchvision.transforms as transforms

        vae = load_vae(device, model_id="stabilityai/sd-vae-ft-ema")
        tfm = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        raw_ds = ImageNet100Dataset(args.data_dir, transform=tfm)
        lat_ds = LatentDataset(raw_ds, vae, device)
        torch.save({"latents": lat_ds.latents, "labels": lat_ds.labels}, args.cache_path)

    if is_ddp:
        dist.barrier()

    cache = torch.load(args.cache_path, map_location="cpu")
    dataset = CachedLatentDataset(cache["latents"], cache["labels"])
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    base_student = SelfFlowPerTokenDiT.from_scale(
        args.scale,
        input_size=32,
        patch_size=2,
        in_channels=4,
        num_classes=args.num_classes,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ).to(device)
    ema_teacher = EMATeacher(base_student, decay=sf_cfg.ema_decay).to(device)
    student: torch.nn.Module = DDP(base_student, device_ids=[local_rank]) if is_ddp else base_student

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=sf_cfg.lr, weight_decay=sf_cfg.weight_decay
    )
    use_bf16 = device.type == "cuda"
    global_step = 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        student.train()
        for batch_latents, batch_labels in loader:
            batch_latents = batch_latents.to(device)
            batch_labels = batch_labels.to(device, dtype=torch.long)
            x0_tokens = patchify_latents(batch_latents, patch_size=2)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = self_flow_train_step(student, ema_teacher, x0_tokens, batch_labels, sf_cfg)

            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=sf_cfg.max_grad_norm)
            optimizer.step()
            ema_teacher.update(base_student)
            global_step += 1

            if rank == 0 and global_step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} Step {global_step} "
                    f"Loss {out['loss'].item():.4f} "
                    f"(gen={out['loss_gen'].item():.4f}, rep={out['loss_rep'].item():.4f})"
                )

        if rank == 0 and epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "model": base_student.state_dict(),
                    "ema": ema_teacher.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"selfflow_epoch_{epoch}.pt"),
            )

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
