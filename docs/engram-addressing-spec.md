# Engram Sidecar Table Addressing Specification (v3.0)

**Status:** Normative specification for the Engram module ([`train/src/engram/`](file:///j:/Kimi/projects/Llama-3-Rebuild/train/src/engram/)).
**Purpose:** Defines canonical token vocabulary compression, injective unigram addressing, $n$-gram multiplicative-XOR hashing, host-resident table layout, and sparse row optimization.

---

## 1. The Addressing Design at a Glance

| Component | Specification |
| :--- | :--- |
| **Orders ($n$) & Heads ($k$)** | $n \in \{1, 2, 3\}$ (unigram, bigram, trigram), 2 heads per order (6 tables total). |
| **Row Dimensions** | $d_{\text{row}} = 256$, stored host-resident in **bf16** ($\approx 2.25\text{ GB}$ RAM total). |
| **Unigram Addressing ($n=1$)** | Injective affine map: $\text{idx} = (A \cdot c + B) \bmod M_1$ where $M_1$ is prime $\ge \max(\text{canonical\_id}) + 1$. **Zero collisions guaranteed.** |
| **$N$-Gram Addressing ($n \ge 2$)** | 64-bit multiplicative-XOR state mix over canonical token IDs: $\text{idx} = \text{state} \bmod M_{n,k}$. Distinct per-head primes decorrelate collision sets. |
| **Salt Stream** | Pinned SHA-256 stream: derived deterministically from `engram.v1:n={n}:k={k}:{field}`. No magic numbers. |
| **Row Initialization** | $\text{row} \sim \text{Uniform}(-0.01, +0.01)$, initialized in fp32 with dedicated per-table RNG generators, stored in bf16. |
| **Injection Point** | Injected strictly at the output of **Layer 3** (Block 1 Global). Readout matrix $U \in \mathbb{R}^{2048 \times 256}$ is zero-initialized (Invariant I1, exact no-op at step 0). |
| **Optimizer** | Sparse 8-bit AdamW operating strictly over rows touched in the forward batch (LR $\times 5$, weight decay 0). |

---

## 2. Stage 0 — Canonical Vocabulary Compression

To prevent surface variants ("apple", "Apple", "APPLE") from fragmenting the lexical table into separate rows, token IDs are mapped to **canonical IDs**:

```
canon_string(token) = casefold(NFKC(text(token)))
canonical_id(token) = min { token_id | token_id in class(token) }
```

### OLMo-3.1 Canonical Mapping Artifact
- **Raw Vocabulary ($|V|$):** 100,278 (OLMo-3.1 BPE).
- **Canonical Vocabulary ($|V'|$):** 75,869 (24,409 surface variants merged, empty class size 0).
- **Artifact Path:** [`train/src/engram/assets/canon_olmo_v1.npy`](file:///j:/Kimi/projects/Llama-3-Rebuild/train/src/engram/assets/canon_olmo_v1.npy)
- **Pinned SHA-256 Checksum:**
  `"8bb4fa0dff964ec53035255a27c3dc6b635a4709aabda449105828ad4c6409ad"`

---

## 3. Stage 1 — Salt Constants Derivation

All hashing multipliers, XOR masks, and initial states are derived via SHA-256 digests from a standard format:

```python
import hashlib

def const64(n: int, k: int, field: str) -> int:
    label = f"engram.v1:n={n}:k={k}:{field}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "little")
```

Constants per (order $n$, head $k$):
- `init`: 64-bit starting state.
- `A`: 64-bit multiplier (forced odd via `A | 1`).
- `B`: 64-bit additive constant.
- `C`: 64-bit XOR mask.

---

## 4. Stage 2 — Addressing Equations

### 4.1 Injective Unigram Addressing ($n=1$)
Because the unigram table size is sized to prime $M_1 = 100,279 > \max(c) = 100,277$, unigram hashing is an **injective affine permutation**:
$$\text{idx} = (A_{1,k} \cdot c_t + B_{1,k}) \bmod M_1$$
Since $\gcd(A, M_1) = 1$, every distinct canonical token $c_t$ maps to a unique row. **Zero collisions exist.**

### 4.2 Multiplicative-XOR N-Gram Mix ($n \ge 2$)
For tokens at sequence position $t \ge n - 1$:
```python
state = const64(n, k, "init")
for offset in range(n - 1, -1, -1):
    c = canonical_ids[t - offset]
    state = (state * const64(n, k, f"mult_{offset}") + c) & 0xFFFFFFFFFFFFFFFF
    state = state ^ const64(n, k, f"xor_{offset}")
idx = state % M[n][k]
```
- **Boundary Condition:** For positions $t < n - 1$, the n-gram is undefined; its contribution is masked to zero.
- **Table Sizes (Primes):**
  - Order 2 (bigrams): $M_{2,0} = 1,048,573$, $M_{2,1} = 1,048,571$.
  - Order 3 (trigrams): $M_{3,0} = 1,048,559$, $M_{3,1} = 1,048,549$.

---

## 5. Host Table Layout & Memory Footprint

The tables reside in pinned host RAM; rows are gathered on the CPU and transferred to GPU in a non-blocking stream:

| Table (Order, Head) | Prime Modulus ($M$) | Row Dimension | Element Type | Host Memory |
| :--- | :--- | :--- | :--- | :--- |
| **Order 1, Head 0** | 100,279 | 256 | `torch.bfloat16` | $\approx 51.3\text{ MB}$ |
| **Order 1, Head 1** | 100,279 | 256 | `torch.bfloat16` | $\approx 51.3\text{ MB}$ |
| **Order 2, Head 0** | 1,048,573 | 256 | `torch.bfloat16` | $\approx 536.9\text{ MB}$ |
| **Order 2, Head 1** | 1,048,571 | 256 | `torch.bfloat16` | $\approx 536.9\text{ MB}$ |
| **Order 3, Head 0** | 1,048,559 | 256 | `torch.bfloat16` | $\approx 536.9\text{ MB}$ |
| **Order 3, Head 1** | 1,048,549 | 256 | `torch.bfloat16` | $\approx 536.9\text{ MB}$ |
| **Total Host Memory**| | | | **$\mathbf{\approx 2.25\text{ GB}}$ ($\mathbf{2.09\text{ GiB}}$)** |

---

## 6. Readout & Residual Stream Injection

1. **Gating & Readout:**
   Given gathered head rows $r_{n,k} \in \mathbb{R}^{B \times T \times 256}$:
   $$r_{\text{cat}} = \text{concat}([r_{1,0}, r_{1,1}, r_{2,0}, r_{2,1}, r_{3,0}, r_{3,1}], \dim=-1) \in \mathbb{R}^{B \times T \times 1536}$$
   $$\text{readout} = \text{gate} \cdot U(\text{RMSNorm}(r_{\text{cat}})) \in \mathbb{R}^{B \times T \times 2048}$$
2. **Zero-Initialization (Invariant I1):**
   $U \in \mathbb{R}^{2048 \times 1536}$ is initialized to **all zeros**. At step 0, readout is identically zero.
3. **Delta-Sum Bookkeeping (Invariant I2 / I8):**
   The readout is added to the residual stream immediately after Layer 3 (Block 1 Global) and registered into the Block 1 partial delta-sum.
