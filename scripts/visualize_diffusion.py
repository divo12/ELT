import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from elt.diffusion.config import DiffusionELTConfig
from elt.diffusion.inference import NoiseScheduler, diffusion_any_time_inference
from elt.diffusion.model import ELT_DiT
from elt.diffusion.vae import load_vae


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize Any-Time Diffusion ELT generations")
    p.add_argument("--ckpt", type=str, default="checkpoint_epoch_diffusion0.pt")
    p.add_argument("--out", type=str, default="generated_diffusion.png")
    args = p.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dcfg = DiffusionELTConfig(
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_unique_layers=3,
        L_max=4,
        batch_size=32,
    )

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["model"]
    if next(iter(state_dict)).startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")
    model = ELT_DiT(dcfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    vae = load_vae(device)
    noise_scheduler = NoiseScheduler(dcfg)

    fig = diffusion_any_time_inference(model, noise_scheduler, dcfg, device=device, vae=vae)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
