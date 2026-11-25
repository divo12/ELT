from __future__ import annotations

import torch
import torch.nn.functional as F

from elt import (
    DIT_SCALES,
    MASKGIT_SCALES,
    DiffusionELTConfig,
    ELTConfig,
    ELTModel,
    ELT_DiT,
    MAGVITTokenizer,
    MLMHead,
    VideoELTConfig,
    linear_curriculum,
)
from elt.diffusion import NoiseScheduler, ddpm_sample_latents, diff_ILSD_train_step
from elt.maskgit import ILSD_train_step, any_time_generate
from elt.maskgit.dataset import random_masking
from elt.maskgit.tokenizer import VQGAN
from elt.video import generate_video


def test_paper_complete() -> None:
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Verify Table 6 & Table 7 paper scaling presets + Appendix A 5x weight decay for >= H
    assert set(MASKGIT_SCALES) == {"S", "B", "L", "XL", "H", "g", "G"}
    assert {"Small", "Large"}.issubset(DIT_SCALES)
    cfg_l = ELTConfig.from_scale("L", n_unique_layers=8, L_max=3)
    cfg_h = ELTConfig.from_scale("H", n_unique_layers=8, L_max=4)
    assert abs(cfg_h.weight_decay - 5.0 * cfg_l.weight_decay) < 1e-9

    vcfg_paper = VideoELTConfig(scale="L", n_unique_layers=6, L_max=4)
    assert vcfg_paper.grid_shape == (4, 16, 16) and vcfg_paper.max_seq_len == 1024
    assert vcfg_paper.num_classes == 101 and vcfg_paper.label_drop_prob == 0.0

    # 2. Paper-complete DiT Diffusion ELT (adaLN-Zero + 2x2 patch + shifted cosine + v-pred + Sigmoid-ELBO + CFG)
    dcfg = DiffusionELTConfig(
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_unique_layers=2,
        L_max=4,
        latent_channels=4,
        latent_size=8,
        patch_size=2,
        num_classes=4,
        sampling_steps=6,
        cfg_scale=3.0,
    )
    dit = ELT_DiT(dcfg).to(device)
    ns = NoiseScheduler(dcfg)
    opt_d = torch.optim.AdamW(dit.parameters(), lr=2e-3, betas=dcfg.betas, weight_decay=dcfg.weight_decay)

    proto = torch.randn(4, 4, 8, 8, device=device)
    d_losses = []
    for step in range(25):
        lbl = torch.randint(0, 4, (8,), device=device)
        x0 = proto[lbl] + 0.05 * torch.randn(8, 4, 8, 8, device=device)
        loss_d, l_int = diff_ILSD_train_step(
            dit, x0, lbl, linear_curriculum(step, 25), dcfg, noise_scheduler=ns
        )
        assert 1 <= l_int < dcfg.L_max
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        if step == 2:
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in dit.trunk.parameters())
        opt_d.step()
        d_losses.append(float(loss_d.item()))
    assert sum(d_losses[-5:]) < sum(d_losses[:5]), "DiT ELT loss failed to decrease"

    # Verify single-pass dual prefix exactness & DDPM CFG + dynamic loop schedule sampling
    dit.eval()
    xt_test = torch.randn(2, 4, 8, 8, device=device)
    t_test = torch.tensor([0.25, 0.75], device=device)
    lbl_test = torch.tensor([0, 2], device=device)
    v_max, v_int, _ = dit.forward_dual(xt_test, t_test, l_max=4, l_min=1, l_int=2, labels=lbl_test)
    assert torch.allclose(v_int, dit(xt_test, t_test, n_loops=2, labels=lbl_test), atol=1e-5)
    assert torch.allclose(v_max, dit(xt_test, t_test, n_loops=4, labels=lbl_test), atol=1e-5)

    sampled_latents = ddpm_sample_latents(
        dit, ns, dcfg, n_loops=[4, 4, 3, 3, 2, 1], labels=lbl_test, cfg_scale=3.0, device=device
    )
    assert sampled_latents.shape == (2, 4, 8, 8) and torch.isfinite(sampled_latents).all()

    # 3. Paper-complete MaskGIT Image ELT (class token + tied MLMHead + ILSD elasticity + CFG)
    patterns = torch.stack([(torch.arange(16, device=device) * (c + 3) + c * 7) % 32 for c in range(4)])

    def train_maskgit(use_ilsd: bool, steps: int = 160) -> tuple[ELTModel, MLMHead, ELTConfig]:
        torch.manual_seed(7)
        mcfg = ELTConfig(
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_unique_layers=2,
            codebook_size=32,
            grid_shape=(1, 4, 4),
            num_classes=4,
            L_max=4,
            sampling_steps=6,
            cfg_scale=2.5,
        )
        m = ELTModel(mcfg).to(device)
        h = MLMHead(mcfg, embed_weight=m.token_embedding.weight).to(device)
        seen = {id(p): p for p in list(m.parameters()) + list(h.parameters())}
        opt = torch.optim.AdamW(list(seen.values()), lr=3e-3, weight_decay=1e-2)
        for s in range(steps):
            c = torch.randint(0, 4, (32,), device=device)
            y = patterns[c]
            masked, targets, mask = random_masking(y, mcfg)
            loss_m, _ = ILSD_train_step(
                m, h, masked, targets, mask, linear_curriculum(s, steps), mcfg, labels=c, use_ilsd=use_ilsd
            )
            opt.zero_grad(set_to_none=True)
            loss_m.backward()
            opt.step()
        m.eval()
        h.eval()
        return m, h, mcfg

    van_m, van_h, mcfg = train_maskgit(use_ilsd=False)
    elt_m, elt_h, _ = train_maskgit(use_ilsd=True)

    torch.manual_seed(99)
    eval_c = torch.arange(4, device=device).repeat(16)
    eval_y = patterns[eval_c]
    eval_masked, eval_targets, eval_mask = random_masking(eval_y, mcfg)
    with torch.no_grad():
        for l_eval in (1, 2, 3):
            van_logits = van_h(van_m(eval_masked, n_loops=l_eval, labels=eval_c), van_m.token_embedding.weight)
            elt_logits = elt_h(elt_m(eval_masked, n_loops=l_eval, labels=eval_c), elt_m.token_embedding.weight)
            assert F.cross_entropy(elt_logits[eval_mask], eval_targets[eval_mask]) < F.cross_entropy(
                van_logits[eval_mask], eval_targets[eval_mask]
            )

    gen_tokens = any_time_generate(
        elt_m, elt_h, mcfg, n_loops=2, class_label=torch.tensor([0, 1], device=device), cfg_scale=2.5, device=device, reshape_grid=True
    )
    assert gen_tokens.shape == (2, 1, 4, 4) and int(gen_tokens.max()) < 32

    vq2d = VQGAN().to(device)
    with torch.no_grad():
        rec_2d, idx_2d = vq2d(torch.randn(1, 3, 256, 256, device=device))
    assert rec_2d.shape == (1, 3, 256, 256) and idx_2d.shape == (1, 16, 16)

    # 4. Paper-complete MAGVIT 3D Video UCF-101 ELT (Causal 3D VQGAN 16x128x128 <-> 4x16x16 + L > L_max extrapolation)
    magvit = MAGVITTokenizer(base_channels=16, embed_dim=32, codebook_size=64).to(device)
    magvit.eval()
    with torch.no_grad():
        vid_in = torch.randn(1, 3, 16, 128, 128, device=device)
        vid_rec, vid_idx = magvit(vid_in)
    assert vid_idx.shape == (1, 4, 16, 16) and vid_rec.shape == (1, 3, 16, 128, 128)

    vcfg = VideoELTConfig(
        scale="S",
        n_unique_layers=2,
        L_max=4,
        codebook_size=64,
        grid_shape=(4, 16, 16),
        sampling_steps=4,
    )
    v_model = ELTModel(vcfg).to(device)
    v_head = MLMHead(vcfg, embed_weight=v_model.token_embedding.weight).to(device)
    v_masked, v_targets, v_mask = random_masking(vid_idx.reshape(1, -1), vcfg)
    v_loss, _ = ILSD_train_step(
        v_model, v_head, v_masked, v_targets, v_mask, 0.5, vcfg, labels=torch.tensor([3], device=device)
    )
    v_loss.backward()
    assert torch.isfinite(v_loss)

    # Test L = 6 (> L_max = 4) extrapolation per Figure 7 / Table 4
    gen_vid = generate_video(
        v_model, v_head, magvit, vcfg, n_loops=6, class_label=3, device=device
    )
    assert gen_vid.shape == (1, 3, 16, 128, 128) and torch.isfinite(gen_vid).all()
    print(f"All paper-complete ELT checks passed on {device}!")


if __name__ == "__main__":
    test_paper_complete()
