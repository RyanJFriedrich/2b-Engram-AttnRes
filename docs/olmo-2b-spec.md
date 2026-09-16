# OLMo-2B — Specification (v3.0)

> **Architectural Transition Notice:** This document supersedes the legacy `llama-9b-refit-spec.md` (now archived in `docs/archive/llama_9b/`). 
> The project has completely transitioned from Meta Llama to an **Apache 2.0 clean-room foundation** based on AllenAI's `allenai/Olmo-3.1-32B-Think`. The model is deep and narrow ($d_{\text{model}} = 2048$, 41 layers), uses scale-normalized SVD down-projected furniture embeddings, a cold-start interior, dynamic SWA + multi-$\theta$ globals, globals-only AttnRes, and an Engram sidecar.

---

## §0. Read this first — how to interpret this document (NORMATIVE)

This section governs every coding agent and engineer working on this codebase.

### 0.1 Normative vs. informative

This specification distinguishes **the machine** from **the run**:

- **Normative (binding):** Architecture constants, layer topologies, mathematical invariants (§3.7 I1–I9), and data shard contracts (§6). These define *what the model is* and *what the data is*. Do not deviate without an explicit owner decision.
- **Informative (advisory):** Token budgets, epoch counts, step counts, optimizer learning rate schedules, batch accumulation cadences, and memory profiles. These are operational parameters and guidance.

**Agent contract:** Process parameters are owner-discretionary. If the owner trains for more tokens, fewer tokens, or adjusts learning rate schedules at runtime, **that is not a spec violation and requires no objection**.

### 0.2 Requirement tags

- **[LOCKED]** — Implement exactly as specified.
- **[FLEXIBLE]** — Default given; expose as a config knob; owner may vary freely.
- **[EXPERIMENTAL]** — Implement behind a flag, default off.
- **[INFORMATIVE]** — Context, rationale, or guidance. No implementation requirement.

---

## §0.5 Headline numbers & Topology (INFORMATIVE)

A deep, narrow decoder ($d_{\text{model}} = 2048$, 41 layers) trained on top-k distillation shards with an Engram sidecar table. Fits entirely in **24 GB VRAM on a local RTX 4090** under `master_dtype: bf16` ($\sim 20.8\text{ GiB}$ peak), while executing with massive headroom on a 96 GB Blackwell instance.

| Quantity | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | $2,649,665,507$ ($\sim 2.65\text{B}$) | 41 layers + untied embeddings ($V = 100,278$) |
| **Trainable Interior** | $2,238,926,819$ ($\sim 2.24\text{B}$) | 100% cold-start fresh random initialization |
| **Frozen Furniture** | $410,738,688$ ($\sim 0.41\text{B}$) | `embed_tokens` + `lm_head` SVD down-projected from donor |
| **Engram Host Parameters** | $1,179,725,824$ rows ($\approx 2.25\text{ GB}$) | Host RAM resident bf16 rows, 6 tables ({1, 2, 3}-grams) |
| **d_model / heads** | $2048$; 16 Q / 4 KV (GQA 4:1) | $d_{\text{head}} = 128$ |
| **FFN** | 3.5x SwiGLU, $d_{\text{ff}} = 7168$ | Gated linear unit FFN |
| **Locals (SWA)** | 30 layers total | SWA 1/2: 4k window, $\theta=10\text{k}$; SWA 3: 8k window, $\theta=25\text{k}$; sink logit $-10.0$ |
| **Globals + Gather** | 11 layers total (10 block + 1 gather) | Full causal span, p-RoPE $p=0.25$ (32 rotated, 96 invariant); B1–5: $\theta=1\text{M}$; B6–10+Gather: $\theta=5\text{M}$ |
| **Attention Residuals** | `scope: globals_only` | Connects at exactly the 11 global points; keys-only RMSNorm; zero-init queries |
| **Tokenizer** | OLMo-3.1 BPE ($V = 100,278$) | Canonical map $|V'| = 75,869$ (24,409 merged, empty-class size 0) |
| **Static Train Memory** | $\approx 13.28\text{ GiB}$ (bf16 masters) | Model ($4.94\text{ GiB}$) + 8-bit AdamW ($4.17\text{ GiB}$) + bf16 Grads ($4.17\text{ GiB}$) |
| **Dynamic Working Memory**| $\approx 6.0\text{ GiB}$ at $T = 8192, B = 1$ | Checkpointed sublayers ($2.62\text{ GiB}$) + chunked attn ($0.25\text{ GiB}$) + loss chunking |
| **Total Peak VRAM** | $\mathbf{\approx 20.8\text{ GiB}}$ (bf16) / $\mathbf{\approx 25.0\text{ GiB}}$ (fp32) | **Fits inside 24 GB on RTX 4090** |

---

## §1. Architectural Premise (NORMATIVE)

1. **Apache 2.0 Clean-Room Licensing [LOCKED]:**
   The donor is `allenai/Olmo-3.1-32B-Think` (Apache 2.0). All Meta Llama code, checkpoints, and tokenizer artifacts are strictly decommissioned.
2. **Cold Start with SVD-Downprojected Furniture [LOCKED]:**
   The interior 41 layers initialize 100% cold (fresh random initialization). The target unembedding head `lm_head` and input embeddings `embed_tokens` are down-projected from donor width $d_{\text{in}} = 5120$ to $d_{\text{model}} = 2048$ via scale-normalized joint SVD (§2) and **frozen at step 0** (`freeze_embeddings: true`).
3. **Deep & Narrow Geometry [LOCKED]:**
   Parameter budget is invested in depth (41 layers) and routing (11-point Block AttnRes + Engram sidecar) rather than wide hidden dimensions ($d_{\text{model}} = 2048$).
4. **Data-Centric Supervision [LOCKED]:**
   The student learns to bridge the fixed donor coordinate system through gold token cross-entropy and top-k KD logit distributions ($K = 32$, unfolded v2 shards).

---

## §2. Scale-Normalized Joint SVD Furniture Extraction (NORMATIVE)

The donor's embedding and unembedding matrices $E, H \in \mathbb{R}^{V \times 5120}$ ($V = 100,278$) are down-projected into $\mathbb{R}^{V \times 2048}$ using **Scale-Normalized $H$-Biased Joint SVD** with $\alpha = 0.7$.

### 2.1 The Energy Normalization Rationale
In untied transformer models, the input embeddings $E$ have substantially larger Frobenius energy than output unembeddings $H$:
$$\|E\|_F^2 \approx 1.03 \times 10^7 \quad \text{vs.} \quad \|H\|_F^2 \approx 2.09 \times 10^5 \quad (\text{ratio } \approx 49.2)$$
Unnormalized joint SVD on $[E; H]$ is $98\%$ dominated by $E$, leaving $H$ severely under-indexed. Conversely, derivation exclusively from $H$ ($\alpha = 1.0$) destroys $>60\%$ of $E$'s variance (collapsing input representation).

### 2.2 Mathematical Formulation [LOCKED]
Let $\bar{E} = \frac{E}{\|E\|_F}$ and $\bar{H} = \frac{H}{\|H\|_F}$. The Gram matrix $G \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is constructed with $\alpha = 0.7$:
$$G = (1 - \alpha) \cdot \bar{E}^T \bar{E} + \alpha \cdot \bar{H}^T \bar{H} = 0.30 \cdot \bar{E}^T \bar{E} + 0.70 \cdot \bar{H}^T \bar{H}$$

Compute eigendecomposition $G = V \Lambda V^T$ with eigenvalues sorted in descending order. The projection matrix $P \in \mathbb{R}^{2048 \times 5120}$ is formed from the top 2048 right eigenvectors:
$$P = V_{:, :2048}^T \quad \text{satisfying} \quad \|P P^T - I_{2048}\|_\infty < 10^{-4}$$

The projected furniture weights are:
$$\tilde{E} = E P^T \in \mathbb{R}^{V \times 2048}, \quad \tilde{H} = H P^T \in \mathbb{R}^{V \times 2048}$$

### 2.3 Measured Variance Retention (Empirical Contract)
On `allenai/Olmo-3.1-32B-Think` weights, this formulation delivers:
- **$H$ variance retained:** $\mathbf{63.18\%}$ (within $1.9\%$ of the $65.10\%$ theoretical ceiling for 2048 dimensions).
- **$E$ variance retained:** $\mathbf{57.62\%}$ (preventing input token collapse).

---

## §3. Model Architecture & Topology (NORMATIVE)

### 3.1 41-Layer Stack Layout [LOCKED]
The decoder stack consists of **10 4-layer blocks + 1 final gather layer** (41 layers total, 0-indexed 0 to 40):
- **Each Block $b \in \{0, \dots, 9\}$ (Layers $4b$ to $4b+3$):**
  - Layer $4b + 0$: **SWA 1**
  - Layer $4b + 1$: **SWA 2**
  - Layer $4b + 2$: **SWA 3**
  - Layer $4b + 3$: **Global Attention**
- **Layer 40 (Final Layer):** **Gather Layer**

```
Block 1:  [SWA 1 (4k)] -> [SWA 2 (4k)] -> [SWA 3 (8k)] -> [GLOBAL 1 (1M)]  --> AttnRes Point 1
Block 2:  [SWA 1 (4k)] -> [SWA 2 (4k)] -> [SWA 3 (8k)] -> [GLOBAL 2 (1M)]  --> AttnRes Point 2
...
Block 5:  [SWA 1 (4k)] -> [SWA 2 (4k)] -> [SWA 3 (8k)] -> [GLOBAL 5 (1M)]  --> AttnRes Point 5
Block 6:  [SWA 1 (4k)] -> [SWA 2 (4k)] -> [SWA 3 (8k)] -> [GLOBAL 6 (5M)]  --> AttnRes Point 6
...
Block 10: [SWA 1 (4k)] -> [SWA 2 (4k)] -> [SWA 3 (8k)] -> [GLOBAL 10 (5M)] --> AttnRes Point 10
Gather:   [GATHER (5M)]                                                    --> AttnRes Point 11
```

### 3.2 Dynamic Sliding Window Attention (SWA) [LOCKED]
- **SWA 1 & 2:** Window $W = 4096$, standard rotary embedding with $\theta = 10,000.0$.
- **SWA 3:** Window $W = 8192$, standard rotary embedding with $\theta = 25,000.0$.
- **Learned Sink Logit (Invariant I4) [LOCKED]:**
  Every SWA head has a learned scalar sink logit initialized to $-10.0$ (`init: -10.0`). In softmax:
  $$P_{ij} = \frac{\exp(q_i k_j^T / \sqrt{d} - m)}{\sum_{l \in \text{window}} \exp(q_i k_l^T / \sqrt{d} - m) + \exp(s_{\text{sink}} - m)}$$
  where $m = \max(\max_l(q_i k_l^T / \sqrt{d}), s_{\text{sink}})$. Zero KV footprint; acts as an exact near-no-op at step 0 ($e^{-10} \approx 4.5 \times 10^{-5}$).
- **Score Memory Bound:** Query chunking (`attn_query_chunk: 512`) bounds intermediate score materialization under autograd.

### 3.3 Multi-$\theta$ Global Layers & Partial-RoPE [LOCKED]
- **Attention Span:** Full causal sequence span. Dispatched through fused SDPA (FlashAttention-2 / cuDNN).
- **Positional Encoding:** Partial RoPE (NeoX-style slice, $p = 0.25$, first 32 channels of $d_{\text{head}} = 128$ rotated; remaining 96 channels invariant). Frequencies computed over the rotated 32-dim slice.
- **Rotary Frequencies ($\theta$) across Depth:**
  - **Blocks 1–5 (Globals at layers 3, 7, 11, 15, 19):** $\theta = 1,000,000.0$.
  - **Blocks 6–10 (Globals at layers 23, 27, 31, 35, 39):** $\theta = 5,000,000.0$.
  - **Gather Layer (Layer 40):** $\theta = 5,000,000.0$.

### 3.4 Gather Layer (Layer 40) [LOCKED]
The final layer before the output projection. Identical projection geometry to global layers, p-RoPE $p = 0.25$ @ $\theta = 5\text{M}$. Initialized normally with fresh random cold-start weights (`init: random`).

### 3.5 Globals-Only Attention Residuals (Block AttnRes) [LOCKED]
AttnRes connects exclusively at the **11 global points** (`scope: globals_only`):
$$\text{Points} = \{\text{Layers } 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 40\}$$
- **Sources (Invariant I8):** Embedding output $e$ + block delta-sums $\sum \Delta_b$ + partial sum.
- **Norm & Projections:** Keys-only RMSNorm; values unnormed; zero-initialized pseudo-queries (Invariant I3, uniform mix at step 0).

### 3.6 Engram Sidecar Table [LOCKED]
- **Orders:** $\{1, 2, 3\}$ (unigram, bigram, trigram), 2 heads per order (6 tables total).
- **Row Dimension:** 256. Total host RAM $\approx 2.25\text{ GB}$.
- **Moduli:**
  - Order 1: $[100279, 100279]$ (smallest prime $\ge |V'| = 75,869$)
  - Order 2: $[1048573, 1048571]$
  - Order 3: $[1048559, 1048549]$
- **Injection:** Injected at output of Layer 3 (Block 1 Global). Readout matrix $U \in \mathbb{R}^{2048 \times 256}$ initialized to zero (Invariant I1, exact no-op at step 0).
- **Canonical Map:** Pinned SHA-256 `"8bb4fa0dff964ec53035255a27c3dc6b635a4709aabda449105828ad4c6409ad"` in `train/src/engram/assets/canon_olmo_v1.npy`.

### 3.7 Invariants (I1–I9) [LOCKED]
- **I1:** Engram readout $U = 0$ at step 0.
- **I2:** Engram injection registered into Block 1 delta-sum.
- **I3:** AttnRes pseudo-queries initialized to zero.
- **I4:** SWA sink logits initialized to $-10.0$.
- **I5:** Gather layer has uniform global p-RoPE scheme.
- **I6:** Production shards honor mass sum $\sum p_k + \text{tail\_w} = 1.0$.
- **I7:** SWA attention strictly bounded by window.
- **I8:** Residual stream updates registered strictly as delta-sums.
- **I9:** Checkpoints are bitwise resume-safe (model + optimizer + RNG + data cursor).

---

## §4. Precision & Memory Budget (INFORMATIVE)

### 4.1 Training Memory Math ($T = 8192, B = 1$)

1. **Static State (VRAM)**:
   - Weights ($2.65\text{B}$ params, bf16): **$4.94\text{ GiB}$**
   - Optimizer (`adamw8bit` on $2.24\text{B}$ trainable): **$4.17\text{ GiB}$**
   - Gradients (`grad_dtype: "bf16"` on $2.24\text{B}$ trainable): **$4.17\text{ GiB}$**
   - Total Static VRAM: **$13.28\text{ GiB}$**
2. **Dynamic Working State (VRAM)**:
   - Checkpointed sublayer states: **$2.62\text{ GiB}$**
   - Chunked attention scores (`attn_query_chunk: 512`): **$0.25\text{ GiB}$**
   - FFN activations: **$0.12\text{ GiB}$**
   - AttnRes sources ($11 \times 32\text{ MiB}$): **$0.35\text{ GiB}$**
   - Loss chunking (`loss_chunk_size: 512`): **$0.20\text{ GiB}$**
   - PyTorch CUDA workspace: **$\approx 2.5\text{ GiB}$**
   - Total Dynamic Peak: **$\approx 6.0\text{ GiB}$**

**Peak VRAM: $\approx 20.8\text{ GiB}$**. Operates cleanly within the **24 GB VRAM limit of the local RTX 4090** and leaves over $70\text{ GB}$ of headroom on the 96 GB cloud box.

---

## §5. Data & Distillation (NORMATIVE)

1. **Shards Contract:** Binary mmap shards, unfolded (`fold_version: v2`), $K = 32$ top-k logit probabilities with explicit lumped tail mass $w_{\text{tail}}$.
2. **Fused Loss:** Evaluated via `FusedChunkedKDLoss` with `loss_chunk_size: 512`—full vocabulary logits are never materialized across the sequence.
3. **Loss Objective:**
   $$\mathcal{L} = (1 - \alpha_{\text{KD}}) \mathcal{L}_{\text{CE}}(\text{gold}) + \alpha_{\text{KD}} \mathcal{L}_{\text{KD}}(\text{top-}k, \text{tail})$$
