from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from elt.core import DIT_SCALES
from elt.diffusion.config import DiffusionELTConfig
from elt.diffusion.inference import NoiseScheduler, diffusion_any_time_inference
from elt.diffusion.model import ELT_DiT
from elt.diffusion.vae import load_vae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Any-Time Diffusion ELT generations")
    p.add_argument("--ckpt", type=str, default="checkpoint_epoch_diffusion0.pt")
    p.add_argument("--scale", type=str, default="Large", choices=sorted(DIT_SCALES))
    p.add_argument("--n-unique-layers", type=int, default=8)
    p.add_argument("--l-max", type=int, default=4)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--class-label", type=int, default=0)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--out", type=str, default="generated_diffusion.png")
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
    dcfg = checkpoint.get(
        "config",
        DiffusionELTConfig.from_scale(
            scale=args.scale, n_unique_layers=args.n_unique_layers, L_max=args.l_max
        ),
    )
    state_dict = checkpoint["model"]
    if next(iter(state_dict)).startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")
    model = ELT_DiT(dcfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    vae = load_vae(device)
    noise_scheduler = NoiseScheduler(dcfg)

    fig = diffusion_any_time_inference(
        model,
        noise_scheduler,
        dcfg,
        device=device,
        vae=vae,
        loop_counts=tuple(args.loops),
        class_label=args.class_label,
        cfg_scale=args.cfg_scale,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
