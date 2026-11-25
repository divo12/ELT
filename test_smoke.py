from __future__ import annotations

import torch

from elt.diffusion import DiffusionELTConfig, ELT_DiT, NoiseScheduler, diff_ILSD_train_step
from elt.maskgit import ELTConfig, ELTModel, ILSD_train_step, MLMHead, any_time_generate
from elt.maskgit.dataset import random_masking
from elt.maskgit.tokenizer import VQGAN


def test_smoke() -> None:
    dcfg = DiffusionELTConfig(
        d_model=64, n_heads=4, d_ff=128, n_unique_layers=2, L_max=4, latent_size=8, sampling_steps=4
    )
    dit = ELT_DiT(dcfg)
    ns = NoiseScheduler(dcfg)
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, dcfg.T, (2,))
    eps = torch.randn_like(x0)
    xt = ns.add_noise(x0, t, eps)
    loss_d, _ = diff_ILSD_train_step(dit, xt, t, eps, 0.5, dcfg)
    loss_d.backward()
    assert torch.isfinite(loss_d)

    mcfg = ELTConfig(
        d_model=64, n_heads=4, d_ff=128, n_unique_layers=2, max_seq_len=16, vocab_size=65, L_max=4, sampling_steps=4
    )
    model = ELTModel(mcfg)
    head = MLMHead(mcfg)
    toks = torch.randint(0, 64, (2, 16))
    masked, _, mask = random_masking(toks, mcfg)
    loss_m, _ = ILSD_train_step(model, head, masked, toks, mask, 0.5, mcfg)
    loss_m.backward()
    assert torch.isfinite(loss_m)

    gen = any_time_generate(model, head, mcfg, n_loops=2)
    assert gen.shape == (1, 16) and int(gen.max()) < 64

    vq = VQGAN()
    rec, idx = vq(torch.randn(1, 3, 256, 256))
    assert rec.shape == (1, 3, 256, 256) and idx.shape == (1, 16, 16)


if __name__ == "__main__":
    test_smoke()
    print("Smoke checks passed.")
