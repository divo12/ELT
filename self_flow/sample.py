#!/usr/bin/env python3
"""Sample images from a trained Self-Flow model (adapted from black-forest-labs/Self-Flow, Apache-2.0).

Compatible with official HuggingFace checkpoint `Hila/Self-Flow` (`selfflow_imagenet256.pt`)
as well as checkpoints produced by `self_flow/train.py`.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from self_flow.model import SIT_SCALES, SelfFlowPerTokenDiT
from self_flow.sampling import denoise_loop
from self_flow.utils import batched_prc_img, scattercat


def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_dev = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_dev)
        return rank, world_size, torch.device(f"cuda:{local_dev}")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return 0, 1, device


def load_vae(device: torch.device, dtype: torch.dtype = torch.float32):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae, 0.18215, 0.0


def decode_latents(vae, latents: torch.Tensor, scale_factor: float, shift_factor: float) -> torch.Tensor:
    latents = latents / scale_factor + shift_factor
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        images = vae.decode(latents.to(dtype=vae_dtype)).sample
    return ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def load_model(ckpt_path: str, device: torch.device, scale: str = "SiT-XL") -> SelfFlowPerTokenDiT:
    spec = SIT_SCALES[scale]
    model = SelfFlowPerTokenDiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=spec.hidden_size,
        depth=spec.depth,
        num_heads=spec.num_heads,
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "ema" in ckpt:
        state_dict = ckpt["ema"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device=device)
    model.eval()
    return model


@torch.no_grad()
def sample_batch(
    model: SelfFlowPerTokenDiT,
    vae,
    scale_factor: float,
    shift_factor: float,
    batch_size: int,
    class_labels: torch.Tensor,
    num_steps: int = 250,
    cfg_scale: float = 1.0,
    guidance_low: float = 0.0,
    guidance_high: float = 1.0,
    mode: str = "SDE",
    sample_shift: float = 1.0,
    device: torch.device | str = "cuda",
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device)
    use_bf16 = dev.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    noise = torch.randn(batch_size, 4, 32, 32, device=dev, dtype=dtype)
    p = 2
    # (B, C, H, W) -> (B, C*p*p, H/p, W/p)
    noise_patched = (
        noise.view(batch_size, 4, 16, p, 16, p)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch_size, 4 * p * p, 16, 16)
    )
    x, x_ids = batched_prc_img(noise_patched.cpu())
    x = x.to(device=dev)
    x_ids = x_ids.to(device=dev)

    if cfg_scale > 1.0:
        x = torch.cat([x, x], dim=0)
        x_ids = torch.cat([x_ids, x_ids], dim=0)
        null_labels = torch.full_like(class_labels, 1000)
        class_labels = torch.cat([null_labels, class_labels], dim=0)

    with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=use_bf16):
        samples = denoise_loop(
            model=model,
            num_steps=num_steps,
            cfg_scale=cfg_scale if cfg_scale > 1.0 else None,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            mode=mode,
            sample_shift=sample_shift,
            x=x,
            x_ids=x_ids,
            vector=class_labels,
        )

    if cfg_scale > 1.0:
        samples = samples[batch_size:]
        x_ids = x_ids[batch_size:]

    samples_spatial = scattercat(samples, x_ids)
    latents = (
        samples_spatial.view(batch_size, 4, p, p, 16, 16)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch_size, 4, 32, 32)
    )
    images = decode_latents(vae, latents, scale_factor, shift_factor)
    return (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Sample images from Self-Flow model")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--scale", type=str, default="SiT-XL", choices=sorted(SIT_SCALES))
    p.add_argument("--output-dir", type=str, default="./samples")
    p.add_argument("--num-fid-samples", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--mode", type=str, default="SDE", choices=["SDE", "ODE"])
    p.add_argument("--sample-shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--guidance-low", type=float, default=0.0)
    p.add_argument("--guidance-high", type=float, default=0.7)
    p.add_argument("--no-save-images", action="store_true")
    args = p.parse_args()

    from PIL import Image

    rank, world_size, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_save_images:
            (output_dir / "images").mkdir(exist_ok=True)
    if world_size > 1:
        dist.barrier()

    model = load_model(args.ckpt, device, scale=args.scale)
    vae_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae, scale_factor, shift_factor = load_vae(device, dtype=vae_dtype)

    samples_per_gpu = math.ceil(args.num_fid_samples / world_size)
    start_idx = rank * samples_per_gpu
    my_samples = max(0, min(start_idx + samples_per_gpu, args.num_fid_samples) - start_idx)
    num_batches = math.ceil(my_samples / args.batch_size)

    all_samples = []
    for batch_idx in tqdm(range(num_batches), desc=f"Rank {rank}", disable=(rank != 0)):
        b_start = batch_idx * args.batch_size
        bs = min(b_start + args.batch_size, my_samples) - b_start
        class_labels = torch.randint(0, 1000, (bs,), device=device)
        images = sample_batch(
            model=model,
            vae=vae,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
            batch_size=bs,
            class_labels=class_labels,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            mode=args.mode,
            sample_shift=args.sample_shift,
            device=device,
            seed=args.seed + rank * 100000 + batch_idx,
        )
        all_samples.append(images)
        if not args.no_save_images and rank == 0:
            for i, img in enumerate(images):
                Image.fromarray(img).save(output_dir / "images" / f"{start_idx + b_start + i:06d}.png")

    gathered = np.concatenate(all_samples, axis=0)[: args.num_fid_samples]
    if rank == 0:
        npz_path = output_dir / f"samples_{len(gathered)}.npz"
        np.savez(npz_path, arr_0=gathered)
        print(f"Saved {len(gathered)} samples to {npz_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
