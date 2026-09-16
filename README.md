# 2B-Engram-AttnRes (OLMo-2B Spec v3.0)

A from-scratch deep & narrow ~2.65B decoder ($d_{\text{model}} = 2048$, 41 layers) built on an **Apache 2.0 clean-room foundation** with an Engram sidecar table, trained on top-$k$ distillation shards (text gold + frozen logit anchors).

The donor architecture is AllenAI's `allenai/Olmo-3.1-32B-Think` (Apache 2.0).

---

## Key Highlights

- **Scale-Normalized SVD Furniture ($\alpha = 0.7$):** Input embeddings (`embed_tokens`) and output head (`lm_head`) are down-projected from donor width $5120 \to 2048$, preserving **63.18% of $H$ variance** and **57.62% of $E$ variance**. Both are **frozen at step 0**.
- **Cold-Start Interior:** 100% fresh random initialization across all 41 layers (including Layer 40 Gather).
- **Block AttnRes:** Pseudo-query residual routing across 11 connection points, providing direct gradient highways to the embedding interface.
- **Dynamic Sliding Window Attention (SWA):** 4k / 4k / 8k windowing across blocks with learned attention sinks ($-10.0$).
- **Multi-$\theta$ Partial RoPE:** $\rho = 0.25$, $\theta = 1\text{M}$ for Blocks 1–5, $\theta = 5\text{M}$ for Blocks 6–10 and Layer 40 Gather.
- **Engram Sidecar (Host RAM):** Injective unigram ($M_1 = 100,279$) and multiplicative-XOR hashed bigram/trigram tables ($M_2 \approx 1\text{M}, M_3 \approx 1\text{M}$) updated with sparse 8-bit AdamW.
- **Lumped-Tail Top-32 Distillation:** Fused/chunked KD loss over frozen teacher top-32 probabilities + lumped tail.

---

## Layout

```
docs/                     # authoritative design specs (v3.0), runbooks, engram, pipeline
OriginalModel/            # OLMo-3.1-32B-Think donor weights & tokenizer (local)
exports/                  # base checkpoints (e.g. exports/olmo-2b-base-v1)
data_pipeline/            # data production and bulk scoring client
train/                    # core training harness
  configs/                # YAML run configs (rtx4090, rtx6000_ada, etc.)
  src/
    config.py             # dataclass configs and YAML validation
    model/                # decoder, refit (AttnRes, SWA, sinks, p-RoPE)
    data/                 # mmap top-k shard loader & writer
    distill/              # fused chunked KD loss
    engram/               # canon, hash addressing, host tables, readout
    train/                # trainer loop, 8-bit AdamW, bf16 master weights
    tools/                # SVD extraction, base checkpoint builder, NPZ converter
  tests/                  # comprehensive invariant test suite (111 tests)
  scripts/                # train_phase0, preflight, step0_sanity, etc.
```

---

## Getting Started

### Environment
- Python 3.10+ (tested on 3.10 & 3.13)
- PyTorch $\ge 2.5$ with CUDA
- Dependencies: `pip install -r train/requirements.txt`

### Running Unit Tests
```bash
python -m pytest train/tests -v
```

### Launching Training
- **RTX 4090 (24 GB VRAM):**
  ```bash
  python -m train.scripts.train_phase0 --config train/configs/rtx4090_train_24m.yaml
  ```
- **RTX 6000 Ada (48 GB VRAM):**
  ```bash
  python -m train.scripts.train_phase0 --config train/configs/rtx6000_ada_train.yaml
  ```
