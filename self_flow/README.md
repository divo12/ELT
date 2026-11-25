# Self-Flow: Self-Supervised Flow Matching (`arXiv:2603.06507`)

Implementation of **[Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://arxiv.org/abs/2603.06507)** (Chefer, Esser et al., Black Forest Labs, ICML 2026):

- **Official Open-Source Inference (`model.py`, `sampling.py`, `utils.py`, `sample.py`):** Adapted from [`black-forest-labs/Self-Flow`](https://github.com/black-forest-labs/Self-Flow) (Apache-2.0), preserving 100% `state_dict` compatibility with the official `Hila/Self-Flow` (`selfflow_imagenet256.pt`) checkpoint without requiring `timm` or `einops`, and adding full `ODE` (`euler` / `heun2` + `sampleshift`) support alongside `SDE` (`250`-step).
- **Paper-Complete Training & Multi-Modal Extensions (`schedules.py`, `losses.py`, `train.py`):** Implements the unreleased training components directly from `arXiv:2603.06507`:
  - **Dual-Timestep Scheduling (`Section 3.3`, `Eq. 4–5`):** Per-token heterogeneous timestep vector $\boldsymbol{\tau} \in [0, 1]^N$ ($\mathcal{R}_M = 0.25$ image, $0.10$ video, $0.50$ audio) paired with cleaner homogeneous teacher input $\tau_{\min} = \min(t, s)$.
  - **Timestep Distributions (`Appendix A.2 & C`):** `timeshift` $s(\alpha, t)$, `uniform`, `shifted_uniform`, `logit_normal`, `plateau_logit_normal`, and video 5% $[0.95, 1.0]$ low-SNR tail sampling.
  - **EMA Self-Distillation Objective (`Section 3.4`, `Eq. 6–7`):** Momentum EMA teacher ($\text{decay} = 0.9999$), projector $h_\theta^{(l)}$ ($l = 0.3D$) aligned via negative cosine similarity to raw EMA teacher features $f_{\theta'}^{(k)}$ ($k = 0.7D$) with $\gamma = 0.8$.
  - **Unified Multi-Modal Backbone (`Appendix D`):** `MultiModalSelfFlow` sharing a single per-token transformer trunk across image, video, and audio token streams with modality loss weights $(w_I, w_V, w_A)$.

## Usage

### 1. Sample with Official BFL Checkpoint (`Hila/Self-Flow`)

```bash
python3 self_flow/sample.py \
  --ckpt checkpoints/selfflow_imagenet256.pt \
  --mode SDE --num-steps 250 --output-dir ./samples
```

### 2. Train Self-Flow from Scratch

```bash
torchrun --nproc_per_node=8 self_flow/train.py \
  --scale SiT-XL --mask-ratio 0.25 --gamma 0.8 \
  --timestep-dist uniform --data-dir /path/to/imagenet
```
