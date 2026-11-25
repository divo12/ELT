from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from elt.core import MASKGIT_SCALES
from elt.maskgit.model import ELTModel, MLMHead
from elt.video import VideoELTConfig, load_video_tokenizer, video_any_time_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize UCF-101 Any-Time MAGVIT Video ELT")
    p.add_argument("--ckpt", type=str, default="checkpoint_video_epoch_0.pt")
    p.add_argument("--magvit-ckpt", type=str, default="ckpts/magvit-ucf101-f4x8x8.pt")
    p.add_argument("--scale", type=str, default="L", choices=sorted(MASKGIT_SCALES))
    p.add_argument("--n-unique-layers", type=int, default=6)
    p.add_argument("--l-max", type=int, default=4)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4, 6])
    p.add_argument("--sampling-steps", type=int, default=12)
    p.add_argument("--class-label", type=int, default=0)
    p.add_argument("--out", type=str, default="generated_video_filmstrip.png")
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
        VideoELTConfig(
            scale=args.scale,
            n_unique_layers=args.n_unique_layers,
            L_max=args.l_max,
            sampling_steps=args.sampling_steps,
        ),
    )
    print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")

    model = ELTModel(cfg).to(device)
    head = MLMHead(cfg, embed_weight=model.token_embedding.weight).to(device)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["head"])
    model.eval()
    head.eval()

    tokenizer = load_video_tokenizer(device, ckpt_path=args.magvit_ckpt)
    fig = video_any_time_inference(
        model,
        head,
        tokenizer,
        cfg,
        device=device,
        loop_counts=tuple(args.loops),
        class_label=args.class_label,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
