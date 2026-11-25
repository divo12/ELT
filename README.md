# Elastic Looped Transformers (ELT) — Paper-Complete PyTorch Implementation

Paper-complete PyTorch implementation of **[ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/abs/2604.09168)** covering all three visual generation settings in the paper:

1. **Class-Conditional Diffusion Transformers (`elt.diffusion`)** — `adaLN-Zero` DiT blocks & final head, $2 \times 2$ latent patching over SD-v1.4 VAE $(4 \times 32 \times 32)$, shifted cosine continuous-time noise schedule, $v$-prediction, Sigmoid-ELBO weighted $\hat{\mathbf{x}}_0$ MSE + ILSD, Classifier-Free Guidance (`cfg_scale = 3.0`), and 512-step DDPM Any-Time / per-step adaptive loop sampling (`Table 5`, `Table 7`: `Small`, `Large`).
2. **Class-Conditional Masked Image Transformers (`elt.maskgit`)** — BERT-style bidirectional Pre-LN transformer with prepended `[CLS]` class token, weight-tied `MLMHead`, VQGAN $16 \times 16$ discrete tokenizer, cosine masking, single-pass ILSD + $S^3$ + linear $\lambda: 1 \to 0$ curriculum, `STemp` annealing (`Eq. 2`), and batched CFG iterative parallel decoding (`Table 5`, `Table 6`: `S`, `B`, `L`, `XL`, `H`, `g`, `G`, including $5\times$ weight decay for scales $\ge \text{H}$).
3. **Class-Conditional Masked Video Transformers on UCF-101 (`elt.video`)** — MAGVIT-style Causal 3D VQGAN compressing $(3 \times 16 \times 128 \times 128)$ clips by $(4 \times 8 \times 8)$ into $(4 \times 16 \times 16) = 1024$ discrete tokens + `6N × 4L` / `6N × 6L` (`~76M` params) parallel video decoding with loop extrapolation ($L > L_{\max}$, `Table 4` & `Figure 7`).

## Repository Structure

```text
ELT/
├── elt/
│   ├── core.py             # LoopedCompositeBlock (single-pass ILSD), Table 6/7 scales, curriculum
│   ├── diffusion/          # Paper-complete DiT + adaLN-Zero + shifted cosine + v-pred + Sigmoid-ELBO
│   ├── maskgit/            # Paper-complete class-conditional MaskGIT + VQGAN + CFG + STemp
│   └── video/              # MAGVIT Causal 3D VQGAN + UCF-101 16x128x128 video ELT pipeline
├── self_flow/              # Self-Flow (arXiv:2603.06507): official BFL inference + Dual-Timestep/EMA training
│   ├── model.py            # SelfFlowPerTokenDiT (BFL checkpoint-compatible) & MultiModalSelfFlow
│   ├── schedules.py        # Dual-Timestep Scheduling + uniform / shifted / (plateau-)logit-normal
│   ├── losses.py           # Self-Flow L_gen + gamma * L_rep cosine alignment + EMATeacher
│   ├── sampling.py         # SDE (250-step) & ODE (Euler/Heun + timeshift) samplers
│   ├── sample.py           # Sampling CLI compatible with Hila/Self-Flow checkpoint
│   └── train.py            # Self-Flow training CLI
├── scripts/
│   ├── train_diffusion.py
│   ├── visualize_diffusion.py
│   ├── train_maskgit.py
│   ├── visualize_maskgit.py
│   ├── train_video.py
│   └── visualize_video.py
├── test_smoke.py           # End-to-end verification of all 3 architectures + ILSD elasticity
└── requirements.txt
```

## Quick Verification

```bash
python3 test_smoke.py
```

## Training & Any-Time Inference

### 1. Diffusion ELT (ImageNet $256 \times 256$, Table 2 & Table 7)

```bash
# Paper 8N x 4L Large config (539M params, effective depth D = 32)
torchrun --nproc_per_node=8 scripts/train_diffusion.py \
  --scale Large --n-unique-layers 8 --l-max 4 \
  --data-dir /path/to/imagenet \
  --cache-path ./cache/imagenet_latents.pt \
  --ckpt-dir ./checkpoints/diffusion

python3 scripts/visualize_diffusion.py \
  --ckpt ./checkpoints/diffusion/checkpoint_epoch_diffusion0.pt \
  --loops 1 2 3 4 --class-label 207 --cfg-scale 3.0
```

### 2. MaskGIT Image ELT (ImageNet $256 \times 256$, Table 1 & Table 6)

```bash
# Paper ELT-XL (7N x 4L, 111M params, FID 2.0) or ELT-L (8N x 3L, 101M params)
torchrun --nproc_per_node=8 scripts/train_maskgit.py \
  --scale XL --n-unique-layers 7 --l-max 4 \
  --data-dir /path/to/imagenet \
  --vqgan-ckpt ./ckpts/maskgit-vqgan-imagenet-f16-256.bin \
  --cache-path ./cache/imagenet_tokens.pt \
  --ckpt-dir ./checkpoints/maskgit

python3 scripts/visualize_maskgit.py \
  --ckpt ./checkpoints/maskgit/checkpoint_epoch_0.pt \
  --loops 1 2 3 4 --class-label 207 --cfg-scale 3.0
```

### 3. MAGVIT Video ELT (UCF-101 $16 \times 128 \times 128$, Table 4 & Figure 7)

```bash
# Paper 6N x 4L config (~76M params, FVD 72.8 at 12 steps / 60.8 at 6N x 6L, 24 steps)
torchrun --nproc_per_node=8 scripts/train_video.py \
  --scale L --n-unique-layers 6 --l-max 4 \
  --data-dir /path/to/ucf101 \
  --magvit-ckpt ./ckpts/magvit-ucf101-f4x8x8.pt \
  --cache-path ./cache/ucf101_tokens.pt \
  --ckpt-dir ./checkpoints/video

python3 scripts/visualize_video.py \
  --ckpt ./checkpoints/video/checkpoint_video_epoch_0.pt \
  --loops 1 2 4 6 --sampling-steps 12 --class-label 0
```
