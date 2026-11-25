# Elastic Looped Transformers (ELT) — PyTorch

Organized PyTorch implementation of **[ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/abs/2604.09168)**, adapted and restructured from [`shakhsm123/Elastic-Looped-Transformers-for-Visual-Generation-PyTorch-Implementation`](https://github.com/shakhsm123/Elastic-Looped-Transformers-for-Visual-Generation-PyTorch-Implementation).

## Repository Structure

```text
ELT/
├── elt/
│   ├── diffusion/          # Looped DiT + SD-v1.4 VAE + DDPM ILSD training/sampling
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── inference.py
│   │   ├── losses.py
│   │   ├── model.py
│   │   └── vae.py
│   └── maskgit/            # Looped MaskGIT + VQGAN tokenizer + MLM ILSD training/sampling
│       ├── config.py
│       ├── dataset.py
│       ├── inference.py
│       ├── losses.py
│       ├── model.py
│       └── tokenizer.py
├── scripts/
│   ├── train_diffusion.py
│   ├── train_maskgit.py
│   ├── visualize_diffusion.py
│   └── visualize_maskgit.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Diffusion-based ELT

```bash
# Single GPU or multi-GPU via torchrun
torchrun --nproc_per_node=1 scripts/train_diffusion.py \
  --data-dir /path/to/imagenet100/data \
  --cache-path ./cache/imagenet100_latents_train.pt \
  --ckpt-dir ./checkpoints/diffusion

# Any-Time inference across loop budgets L = 1..4
python scripts/visualize_diffusion.py \
  --ckpt ./checkpoints/diffusion/checkpoint_epoch_diffusion0.pt \
  --out generated_diffusion.png
```

### 2. MaskGIT-based ELT

```bash
torchrun --nproc_per_node=1 scripts/train_maskgit.py \
  --data-dir /path/to/imagenet100/data \
  --vqgan-ckpt ./ckpts/maskgit-vqgan-imagenet-f16-256.bin \
  --cache-path ./cache/imagenet100_tokens_train.pt \
  --ckpt-dir ./checkpoints/maskgit

python scripts/visualize_maskgit.py \
  --ckpt ./checkpoints/maskgit/checkpoint_epoch_0.pt \
  --vqgan-ckpt ./ckpts/maskgit-vqgan-imagenet-f16-256.bin \
  --out generated_maskgit.png
```
