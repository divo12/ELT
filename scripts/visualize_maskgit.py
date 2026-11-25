import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from elt.maskgit.config import ELTConfig
from elt.maskgit.inference import any_time_generate
from elt.maskgit.model import ELTModel, MLMHead
from elt.maskgit.tokenizer import decode, load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize Any-Time MaskGIT ELT generations")
    p.add_argument("--ckpt", type=str, default="checkpoint_epoch_10.pt")
    p.add_argument("--vqgan-ckpt", type=str, default="ckpts/maskgit-vqgan-imagenet-f16-256.bin")
    p.add_argument("--out", type=str, default="generated_maskgit.png")
    args = p.parse_args()

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
        n_unique_layers=3,
        L_max=4,
        batch_size=32,
    )

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")

    model = ELTModel(cfg).to(device)
    head = MLMHead(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["head"])
    model.eval()
    head.eval()

    vqgan = load_tokenizer(device, ckpt_path=args.vqgan_ckpt)

    loop_counts = [1, 2, 3, 4]
    generated_images = []
    for n_loops in loop_counts:
        token_ids = any_time_generate(
            model, head, cfg, n_loops=n_loops, class_label=None, cfg_scale=1.0, device=device
        )
        image = decode(vqgan, token_ids)
        image = image.squeeze(0).permute(1, 2, 0)
        image = ((image + 1.0) / 2.0).cpu().detach().numpy()
        generated_images.append(np.clip(image, 0.0, 1.0))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Any-Time Inference", fontsize=14)
    for ax, img, n_loops in zip(axes, generated_images, loop_counts):
        ax.imshow(img)
        ax.set_title(f"L={n_loops}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
