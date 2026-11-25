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

    # 5. Self-Flow (arXiv:2603.06507 + BFL official repo): Dual-Timestep + EMA + SDE/ODE + Multi-Modal
    from self_flow import (
        EMATeacher,
        MultiModalSelfFlow,
        SelfFlowConfig,
        SelfFlowPerTokenDiT,
        denoise_loop,
        dual_timestep_schedule,
        multimodal_self_flow_step,
        patchify_latents,
        sample_timesteps,
        self_flow_train_step,
        unpatchify_latents,
    )

    for dist_name in ("uniform", "shifted_uniform", "logit_normal", "plateau_logit_normal"):
        ts_sample = sample_timesteps((16,), dist=dist_name, shift_alpha=2.95, high_noise_prob=0.05, device=device)
        assert ts_sample.shape == (16,) and bool(((ts_sample > 0.0) & (ts_sample < 1.0)).all())

    sf_student = SelfFlowPerTokenDiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=6,
        num_heads=4,
        mlp_ratio=2.0,
        num_classes=4,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ).to(device)
    # Verify exact BFL checkpoint key compatibility
    sd_keys = set(sf_student.state_dict())
    for expected_key in (
        "x_embedder.proj.weight",
        "t_embedder.mlp.0.weight",
        "y_embedder.embedding_table.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
        "blocks.0.adaLN_modulation.1.weight",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.linear.weight",
        "projector.linear1.weight",
        "projector.linear2.weight",
    ):
        assert expected_key in sd_keys, f"Missing BFL key: {expected_key}"

    sf_ema = EMATeacher(sf_student, decay=0.99).to(device)
    sf_cfg = SelfFlowConfig(mask_ratio=0.25, gamma=0.8, lr=3e-3)
    sf_opt = torch.optim.AdamW(sf_student.parameters(), lr=sf_cfg.lr)

    x_lat = torch.randn(4, 4, 8, 8, device=device)
    assert torch.allclose(unpatchify_latents(patchify_latents(x_lat, 2), 2, 4), x_lat)
    x_tok_proto = patchify_latents(x_lat, 2)

    x_tau, x_tau_min, tau_vec, tau_min_vec, _ = dual_timestep_schedule(x_tok_proto, mask_ratio=0.25)
    assert bool((tau_vec >= tau_min_vec.unsqueeze(1) - 1e-7).all()) and x_tau.shape == x_tau_min.shape

    sf_losses = []
    for _ in range(20):
        lbl_sf = torch.randint(0, 4, (8,), device=device)
        x0_sf = x_tok_proto[lbl_sf] + 0.02 * torch.randn(8, 16, 16, device=device)
        out_sf = self_flow_train_step(sf_student, sf_ema, x0_sf, lbl_sf, sf_cfg)
        sf_opt.zero_grad(set_to_none=True)
        out_sf["loss"].backward()
        sf_opt.step()
        sf_ema.update(sf_student)
        sf_losses.append(float(out_sf["loss"].item()))
    assert sum(sf_losses[-4:]) < sum(sf_losses[:4]), "Self-Flow training loss failed to decrease"

    sf_student.eval()
    init_noise = torch.randn(4, 16, 16, device=device)
    cfg_labels = torch.tensor([4, 4, 0, 1], device=device)  # [null, null, cond, cond]
    sde_out = denoise_loop(
        model=sf_student, num_steps=6, cfg_scale=1.5, mode="SDE", x=init_noise, vector=cfg_labels
    )
    ode_out = denoise_loop(
        model=sf_student,
        num_steps=6,
        cfg_scale=1.5,
        mode="ODE",
        sampling_method="heun2",
        sample_shift=1.78,
        x=init_noise,
        vector=cfg_labels,
    )
    assert sde_out.shape == (4, 16, 16) and torch.isfinite(sde_out).all()
    assert ode_out.shape == (4, 16, 16) and torch.isfinite(ode_out).all()

    # Multi-Modal Self-Flow (Appendix D: image + video + audio shared trunk)
    mm_student = MultiModalSelfFlow(
        modality_specs={"image": (16, 16), "video": (48, 32), "audio": (64, 20)},
        hidden_size=64,
        depth=6,
        num_heads=4,
        cond_dim=32,
    ).to(device)
    mm_ema = EMATeacher(mm_student, decay=0.99).to(device)
    weights = {"image": 0.31, "video": 0.12, "audio": 0.57}
    for mod, (dim_m, len_m) in (("image", (16, 16)), ("video", (48, 32)), ("audio", (64, 20))):
        tok_m = torch.randn(2, len_m, dim_m, device=device)
        cond_m = torch.randn(2, 32, device=device)
        mm_out = multimodal_self_flow_step(
            mm_student, mm_ema, tok_m, mod, sf_cfg, cond=cond_m, modality_weights=weights
        )
        mm_out["loss"].backward()
        assert torch.isfinite(mm_out["loss"])

    print(f"All paper-complete ELT & Self-Flow checks passed on {device}!")


if __name__ == "__main__":
    test_paper_complete()
