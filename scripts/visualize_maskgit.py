from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from elt.core import MASKGIT_SCALES
from elt.maskgit.config import ELTConfig
from elt.maskgit.inference import any_time_generate
from elt.maskgit.model import ELTModel, MLMHead
from elt.maskgit.tokenizer import decode, load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Any-Time MaskGIT ELT generations")
    p.add_argument("--ckpt", type=str, default="checkpoint_epoch_0.pt")
    p.add_argument("--vqgan-ckpt", type=str, default="ckpts/maskgit-vqgan-imagenet-f16-256.bin")
    p.add_argument("--scale", type=str, default="L", choices=sorted(MASKGIT_SCALES))
    p.add_argument("--n-unique-layers", type=int, default=8)
    p.add_argument("--l-max", type=int, default=4)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--class-label", type=int, default=0)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--out", type=str, default="generated_maskgit.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = checkpoint.get(
        "config",
        ELTConfig.from_scale(
            scale=args.scale, n_unique_layers=args.n_unique_layers, L_max=args.l_max
        ),
    )
    print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")

    model = ELTModel(cfg).to(device)
    head = MLMHead(cfg, embed_weight=model.token_embedding.weight).to(device)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["head"])
    model.eval()
    head.eval()

    vqgan = load_tokenizer(device, ckpt_path=args.vqgan_ckpt)

    generated_images = []
    for n_loops in args.loops:
        token_ids = any_time_generate(
            model,
            head,
            cfg,
            n_loops=n_loops,
            class_label=args.class_label,
            cfg_scale=args.cfg_scale,
            device=device,
            seed=42,
        )
        image = decode(vqgan, token_ids)
        image = image.squeeze(0).permute(1, 2, 0)
        image = ((image + 1.0) / 2.0).cpu().detach().numpy()
        generated_images.append(np.clip(image, 0.0, 1.0))

    fig, axes = plt.subplots(1, len(args.loops), figsize=(4 * len(args.loops), 4))
    if len(args.loops) == 1:
        axes = [axes]
    fig.suptitle(f"Any-Time MaskGIT Inference (class={args.class_label}, CFG={args.cfg_scale})", fontsize=14)
    for ax, img, n_loops in zip(axes, generated_images, args.loops):
        ax.imshow(img)
        ax.set_title(f"L={n_loops}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
