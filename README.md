# 2B-Engram-AttnRes (OLMo-2B Spec v3.1)

A from-scratch deep & narrow ~2.65B decoder ($d_{\text{model}} = 2048$, 41 layers) built on an **Apache 2.0 clean-room foundation** paired with a **4.35B-parameter Engram sidecar table** ($16.98\text{M}$ rows, $2^{22}$ moduli) and trained on top-$k$ distillation shards (text gold + frozen logit anchors).

The donor architecture is AllenAI's `allenai/Olmo-3.1-32B-Think` (Apache 2.0).

---

## Model Release & Checkpoints

- **Hugging Face Repository:** [Ouroboros-Research/Our1-2b](https://huggingface.co/Ouroboros-Research/Our1-2b)
- **Production Milestone (Step 3,200):** Full standalone model weights (`model.safetensors`), Engram host tables (`engram.safetensors`), tokenizer, canonical map, and configs published at repository root.
- **Flight Checkpoints:** All 33 hourly training checkpoints (`ckpt_step10.pt` through `ckpt_step3200.pt`) preserved with bitwise resume safety (model + optimizer + RNG + data cursor).

---

## Architectural Highlights

- **Scale-Normalized SVD Furniture ($\alpha = 0.7$):** Input embeddings (`embed_tokens`) and output head (`lm_head`) are down-projected from donor width $5120 \to 2048$, preserving **63.18% of $H$ variance** and **57.62% of $E$ variance**. Both are **frozen at step 0** (`freeze_embeddings: true`).
- **Cold-Start Interior:** 100% fresh random initialization across all 41 layers (including Layer 40 Gather).
- **Block AttnRes:** Pseudo-query residual routing across 11 connection points (10 block globals + 1 gather layer), providing direct gradient highways to the embedding interface without stream pollution.
- **Dynamic Sliding Window Attention (SWA):** 4k / 4k / 8k windowing across blocks with learned attention sinks ($-10.0$ initialization, 0 KV footprint).
- **Multi-$\theta$ Partial RoPE:** $\rho = 0.25$, $\theta = 1\text{M}$ for Blocks 1–5, $\theta = 5\text{M}$ for Blocks 6–10 and Layer 40 Gather.
- **Engram Sidecar (Spec v3.1, 4x Expansion):**
  - Host RAM resident in bf16 ($16,977,694$ rows $\times 256$ width = **4,346,289,664 parameters**, ~8.10 GiB weights + ~8.10 GiB 8-bit AdamW optimizer state).
  - Injective unigram ($M_1 = 100,279$) and multiplicative-XOR hashed bigram/trigram tables ($M_2 \in [4194301, 4194287]$, $M_3 \in [4194277, 4194271]$).
  - Updated via custom `SparseRowAdamW8bit` optimizer only for rows touched in the microbatch.
- **Context-Aware Readout (`kv_sigmoid`):**
  - Injected at **Layer 3 output** (Block 1 Global, closing Block 1 before Block 2).
  - Residual hidden state queries retrieved row keys with Gemma-2 logit softcapping ($c = 8.0$, $\tau = 1.0$), learned order biases $b_n$, and independent sigmoid gating $\sigma(\text{logit})$.
  - Zero-initialized projection matrix $U$ guarantees exact step-0 identity / no-op (Invariant I1).
- **Lumped-Tail Top-32 Distillation:** Fused/chunked KD loss over frozen teacher top-32 probabilities + explicit lumped tail ($K = 32$, unfolded v2 shards). Full-vocab logits are never materialized in memory.

---

## Production Training Flight (Blackwell RTX PRO 6000)

Trained on the **NVIDIA RTX PRO 6000 Blackwell Server Edition** (96 GB GDDR7, SM120, PyTorch 2.14, Triton 3.8.0):

| Metric | Measurement |
| :--- | :--- |
| **Total Tokens Trained** | **104,857,600** (100 shards) |
| **Optimizer Steps** | **3,200 steps** ($T = 8192, B = 4, \text{accum} = 1 = 32,768\text{ tok/step}$) |
| **Steady-State Throughput** | **1,015 – 1,020 tokens/sec** (~32.4s per step) |
| **Peak VRAM** | **54.43 GiB** (>42 GiB headroom on 96 GB card) |
| **Initial Loss $\to$ Final Loss** | **11.08 $\to$ 2.21** (EMA 2.9556, instantaneous min 1.9193) |
| **Gradient Norm** | **143.76 $\to$ 0.527** |
| **Engram Rounding Loss (`eng_round`)** | **0.020 $\to$ 0.215** (>78% active gradient flow across touched rows) |

---

## Repository Layout

```
docs/                     # authoritative design specs (v3.1), runbooks, engram, pipeline
exports/                  # base checkpoints (e.g. exports/olmo-2b-base-v1)
data_pipeline/            # data production and bulk scoring client
train/                    # core training harness (tracked in git)
  configs/                # YAML run configs (blackwell_6000_max.yaml, rtx4090, etc.)
  src/
    config.py             # dataclass configs and YAML validation
    model/                # decoder, refit (AttnRes, SWA, sinks, p-RoPE)
    data/                 # mmap top-k shard loader & writer (unfolded v2)
    distill/              # fused chunked KD loss with lumped tail
    engram/               # canon, hash addressing, host tables, kv_sigmoid readout
    train/                # trainer loop, 8-bit AdamW, bf16 master weights
    tools/                # SVD extraction, base checkpoint builder, NPZ converter
  tests/                  # comprehensive invariant test suite (115 passed)
  scripts/                # train_phase0, preflight, step0_sanity, etc.
```

---

## Getting Started

### Environment
- Python 3.10+ (tested on 3.10 & 3.13)
- PyTorch $\ge 2.5$ with CUDA (PyTorch 2.14+ on Blackwell SM120)
- Dependencies: `pip install -r train/requirements.txt`

### Running Invariant & Unit Tests
```bash
python -m pytest train/tests -v
```

### Launching Training

- **RTX PRO 6000 Blackwell (96 GB VRAM, Production Config):**
  ```bash
  python -m train.scripts.train_phase0 --config train/configs/blackwell_6000_max.yaml
  ```

- **Local RTX 4090 (24 GB VRAM, Development / Validation):**
  ```bash
  python -m train.scripts.train_phase0 --config train/configs/real_olmo_2b_v1.yaml
  ```
