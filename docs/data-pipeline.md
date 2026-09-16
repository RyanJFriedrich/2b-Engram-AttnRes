# Data Production Pipeline & Shard Contract (v3.0)

Work order for the data-production pipeline producing top-k distillation shards for the **OLMo-2B (Spec v3.0)** architecture.

---

## 1. Pipeline Stages

```
Interrogate / Ingest  →  Raw JSONL  →  Format & Mask  →  Score (Top-K)  →  Binary MMap Shards
```

1. **Ingest / Interrogate:** Curate natural pre-training text, multilingual corpora, and structured reasoning traces.
2. **Raw Transcripts:** Store raw token/prompt text as JSONL files. Raw text serves as the single immutable source of truth.
3. **Format & Loss Mask:** Emit training text formatted according to the OLMo-3.1 chat template. System/user prompts are loss-masked (`loss_mask = 0`), while target responses are active (`loss_mask = 1`).
4. **Scoring:** Extract top-$k$ logit distributions ($K = 32$) and compute explicit lumped tail mass:
   $$w_{\text{tail}} = 1.0 - \sum_{i=1}^K p_i$$
5. **Pack Shards:** Pack arrays into binary memory-mapped shards (`tokens.bin`, `topk_indices.bin`, `topk_probs.bin`, `tail_w.bin`, `loss_mask.bin`).

---

## 2. Shard Production Contract (Normative)

- **Tokenizer [LOCKED]:** `allenai/Olmo-3.1-32B-Think` BPE ($V = 100,278$).
- **Shard Format [LOCKED]:** **`fold_version: v2`** (unfolded, with tail probability stored explicitly in `tail_w.bin`).
- **Data Types:**
  - `tokens.bin`: `uint32` token IDs.
  - `topk_indices.bin`: `uint32` indices shape `[N, K]`.
  - `topk_probs.bin`: `float16` probabilities shape `[N, K]`.
  - `tail_w.bin`: `float32` scalar tail mass per position.
  - `loss_mask.bin`: `uint8` binary mask (0 = ignore, 1 = compute loss).
- **Position Alignment Rule [LOCKED]:**
  Row $t$ in the shard contains the predictive distribution for position $t$. The gold token at position $t$ is stored at slot 0 of `topk_indices` with its teacher probability.
- **Mass Conservation Invariant (I6) [LOCKED]:**
  On every unmasked row:
  $$\left| \sum_{k=1}^K p_k + w_{\text{tail}} - 1.0 \right| \le 4 \times 10^{-4}$$

---

## 3. Data Classes & Distillation Mix

| Class | Content Description | Loss Formulation | Weight ($\alpha_{\text{KD}}$) |
| :--- | :--- | :--- | :--- |
| **(a) Bulk Pretraining Text** | OpenWeb, Wikipedia, FineWeb-Edu | Top-K Distillation + Gold CE | $\alpha = 0.9$ |
| **(b) High-Quality SFT** | Curated instruction pairs, dialogues | Top-K Distillation + Gold CE | $\alpha \le 0.2$ |
| **(c) Thinking / Reasoning** | Multi-step reasoning traces with verified uncertainty | Pure Cross-Entropy | $\alpha = 0.0$ |
| **(d) Multilingual Tasks** | Parallel translation, nuance explanations | Gold CE + Distillation | $\alpha = 0.5$ |
| **(e) Knowledge Slices** | Recent factual updates, domain knowledge | Gold CE + Distillation | $\alpha = 0.5$ |

### Mixing Constraints
- Every training batch must maintain **$\ge 70\%$ bulk rehearsal** to prevent catastrophic forgetting.
- Shards must include sidecar metadata (`metadata.json`) declaring `text_source`, `tokenizer_version`, `fold_version`, and `data_class`.
