from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from elt.core import DIT_SCALES, linear_curriculum
from elt.diffusion.config import DiffusionELTConfig
from elt.diffusion.dataset import CachedLatentDataset, ImageNet100Dataset, LatentDataset
from elt.diffusion.inference import NoiseScheduler
from elt.diffusion.losses import diff_ILSD_train_step
from elt.diffusion.model import ELT_DiT
from elt.diffusion.vae import load_vae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Paper-Complete Diffusion ELT (DiT + ILSD)")
    p.add_argument("--data-dir", type=str, default="./data/imagenet100")
    p.add_argument("--cache-path", type=str, default="./cache/imagenet100_latents_train.pt")
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints/diffusion")
    p.add_argument("--scale", type=str, default="Large", choices=sorted(DIT_SCALES))
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n-unique-layers", type=int, default=8)
    p.add_argument("--l-max", type=int, default=4)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-ilsd", action="store_true", help="Disable ILSD (vanilla looping ablation)")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
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

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dcfg = DiffusionELTConfig.from_scale(
        scale=args.scale,
        n_unique_layers=args.n_unique_layers,
        L_max=args.l_max,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        total_epochs=args.epochs,
        grad_checkpoint=args.grad_checkpoint,
    )

    Path(args.cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    if rank == 0 and not os.path.exists(args.cache_path):
        import torchvision.transforms as transforms

        print("rank 0: encoding dataset...")
        vae = load_vae(device)
        train_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = ImageNet100Dataset(args.data_dir, transform=train_transforms)
        latent_dataset = LatentDataset(train_dataset, vae, device)
        torch.save(
            {"latents": latent_dataset.latents, "labels": latent_dataset.labels},
            args.cache_path,
        )
        print("rank 0: cache saved")

    if is_ddp:
        dist.barrier()

    cache = torch.load(args.cache_path, map_location="cpu")
    train_dataset = CachedLatentDataset(cache["latents"], cache["labels"])
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=dcfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model: torch.nn.Module = ELT_DiT(dcfg).to(device)
    if args.compile:
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=dcfg.lr,
        weight_decay=dcfg.weight_decay,
        betas=dcfg.betas,
        fused=(device.type == "cuda"),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    noise_scheduler = NoiseScheduler(dcfg)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: step / dcfg.warmup_steps if step < dcfg.warmup_steps else 1.0,
    )

    global_step = 0
    total_steps = max(1, dcfg.total_epochs * len(train_loader))

    for epoch in range(dcfg.total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_latents, batch_labels in train_loader:
            batch_latents = batch_latents.to(device)
            batch_labels = batch_labels.to(device, dtype=torch.long)
            lambda_val = linear_curriculum(
                global_step, total_steps, dcfg.lambda_start, dcfg.lambda_end
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss_total, l_int = diff_ILSD_train_step(
                    model,
                    batch_latents,
                    batch_labels,
                    lambda_val,
                    dcfg,
                    noise_scheduler=noise_scheduler,
                    use_ilsd=(not args.no_ilsd),
                )

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if rank == 0 and global_step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{dcfg.total_epochs} "
                    f"Step {global_step} "
                    f"Loss {loss_total.item():.4f} "
                    f"lambda {lambda_val:.4f} "
                    f"L_int {l_int}"
                )

        if rank == 0 and epoch % 10 == 0:
            raw_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "config": dcfg,
                    "model": raw_state,
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                },
                os.path.join(args.ckpt_dir, f"checkpoint_epoch_diffusion{epoch}.pt"),
            )

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
