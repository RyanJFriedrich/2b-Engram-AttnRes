# AGENTS.md — OLMo-2B (spec v3.0)

## What this project is

A from-scratch deep & narrow ~2.65B decoder ($d_{\text{model}} = 2048$, 41 layers) built on an **Apache 2.0 clean-room foundation** with an Engram sidecar table, trained on top-k distillation shards (text gold + frozen logit anchors).
**The Meta Llama variant is completely retired due to licensing restrictions.** 
The donor is AllenAI's `allenai/Olmo-3.1-32B-Think` (Apache 2.0).

**Spec v3.0 premise: cold start with SVD down-projected donor furniture.**
- **Furniture:** Input embeddings (`embed_tokens`) and output head (`lm_head`) are down-projected from donor width $5120 \to 2048$ via scale-normalized $H$-biased SVD ($\alpha = 0.7$), preserving **$63.18\%$ of $H$ variance** and **$57.62\%$ of $E$ variance**. Both are **frozen at step 0** (`freeze_embeddings: true`).
- **Interior:** 100% cold-start fresh random initialization across all 41 layers (including Layer 40 Gather).
- **Teacher:** Signal comes from gold text tokens + frozen top-k logit distributions from the teacher anchor ($K = 32$, unfolded v2 shards).
- **Memory footprint:** Full training fits inside **24 GB VRAM on a local RTX 4090** ($\approx 20.8\text{ GiB}$ peak VRAM under `master_dtype: bf16` at $T = 8192$), and runs with vast headroom on the 96 GB Blackwell deploy box.

The authoritative design docs are in `docs/`:

- `docs/olmo-2b-spec.md` (**v3.0**) — the authoritative *what/why*. [LOCKED]/[FLEXIBLE]/[EXPERIMENTAL] tags and the normative/informative split (§0.1) are binding.
- `docs/olmo-2b-runbook.md` (v3.0) — operational runbook covering local 4090 execution (Windows / WSL2) and deploy box bring-up.
- `docs/engram-addressing-spec.md` (v3.0) — Engram sidecar addressing, canonical map ($|V| = 100,278 \to |V'| = 75,869$), injective unigram and multiplicative-XOR mix equations.
- `docs/data-pipeline.md` (v3.0) — data-production work order and shard contracts.
- `docs/NPZFormat.md` — the production binary mmap shard contract ($K=32$, unfolded v2, explicit lumped tail).
- Legacy Llama-9B documents are permanently archived in `docs/archive/llama_9b/`.

Where code/docs and the spec disagree, **the spec wins — flag the conflict, don't improvise.**

## Layout

```
docs/                     # authoritative design docs (spec v3.0, runbooks, engram, pipeline)
  archive/llama_9b/       # retired legacy Llama-9B specs and runbooks
OriginalModel/            # OLMo-3.1-32B-Think donor weights & tokenizer (Apache 2.0)
exports/                  # base checkpoints (e.g. exports/olmo-2b-base-v1)
data_pipeline/            # data production (bulk NPZ scoring; separate agent)
train/                    # everything the agent builds lives here
  configs/                # YAML: model (olmo_2b_v1.yaml), run configs (real_olmo_2b_v1.yaml)
  src/
    config.py             # config dataclasses + YAML loader (supports SWA patterns & block thetas)
    model/                # decoder, refit (dynamic SWA/sink/p-RoPE/AttnRes/gather), partial rotary
    data/                 # TopK shard writer/loader (mmap, spec §6.1)
    distill/              # fused/chunked lumped-tail KD loss (§6.2)
    engram/               # Engram sidecar: canon (OLMo-3.1 NFKC+casefold map, pinned SHA-256),
                          # addressing (injective unigram + multiplicative-XOR), host bf16 tables,
                          # readout (zero-init U, I1), sparse host 8-bit AdamW
    train/                # trainer loop, 8-bit AdamW, fp8, checkpointing, bf16 grad accum
    eval/                 # perplexity, NIAH, attention probes
    tools/                # svd_donor (scale-normalized alpha=0.7 SVD down-projection),
                          # base_ckpt (base checkpoint export/import), npz_converter
  tests/                  # invariant + unit test suite (I1–I9, SVD math, layer dispatch)
  scripts/                # entry points (train_phase0.py, preflight.py, build_engram_canon.py)
  runs/                   # outputs, checkpoints, logs (gitignored)
  utils/log.py            # owner-provided logging helper (see Conventions)
```

Run everything from the repo root so `train` imports as a package
(`python -m train.scripts.<entry>`, `python -m pytest train/tests`).

**Git tracks ONLY `train/`** (plus `.gitignore`). `docs/`, `OriginalModel/`, `exports/`, etc. stay local to each machine.

## Commands

- Tests: `python -m pytest train/tests -v`
- Env: Python 3.13, torch 2.10+cu130, transformers 5.7.

## Hardware Profiles

- **Local dev box: RTX 4090, 24 GB VRAM; 68.5 GB RAM.**
  - **Full Training Supported**: At $T = 8192, B = 1$ with `master_dtype: bf16`, static VRAM is $13.28\text{ GiB}$ and dynamic peak is $\approx 6.0\text{ GiB}$ ($\mathbf{\approx 20.8\text{ GiB}}$ total peak).
  - **WSL2 Recommended**: Reclaims $\sim 1.5\text{ GiB}$ Windows desktop VRAM, enables native FlashAttention-2 in SDPA, and unlocks Triton GPU kernel compilation for `torch.compile`.
- **Deploy box: RTX 6000 Pro Blackwell, 96 GB**, Linux, NGC PyTorch container.
  - Generous headroom ($>70\text{ GiB}$ free VRAM), allowing micro-batch scaling ($B=2$ or $4$) and extended sequence lengths (16k/32k).

## Conventions

- **Logging — always use `log()`, never bare `print`.** `log()` from
  `train/utils/log.py` is a drop-in `print` replacement; every call appends a
  timestamped line to a log file (default `common.log`), console output is
  opt-in via `print_console=True`. Check what any run is doing with
  `tail common.log` or PowerShell `Get-Content common.log -Tail 50 -Wait`.
- **Config-driven everything.** Ablations and runs are config diffs, never code
  branches. Unknown config keys are rejected on load — extend
  `train/src/config.py` deliberately.
- Match surrounding code style; keep changes minimal and scoped.

## Standing rules (binding)

1. Invariants I1–I9 (spec §3.7) are enforced by tests, not by care. The locked
   near-no-op inits: AttnRes zero-init pseudo-queries (I3), sink logits at −10
   (I4), Engram zero-init U (I1), cold-start gather layer. Training starts at
   FINAL topology.
2. No YaRN/NTK/position-interpolation code anywhere. No NoPE layers. No QK-norm.
3. Never materialize full-vocab logits in the loss path (fused/chunked KD only via `loss_chunk_size: 512`).
4. Config-driven; no architecture constants hardcoded outside the config.
5. Never edit a run's config mid-flight; `max_steps` exists for clean
   interrupts. Schedules are functions of the run config.
6. Every run logs its full config + seed + code revision + data manifest;
   checkpoints are resume-safe bitwise (I9: model + optimizer + RNG + cursor).
7. One experimental variable at a time (optimizer, precision, architecture).
8. AttnRes sources are delta-sums, never stream snapshots; nothing may be added
   to the residual stream without delta-sum registration (I8 — Engram injection depends on this).
9. Deliberate exclusions: contextual Engram gates, NoPE, full-RoPE globals as default, FP8 weight storage, fp32 AdamW on large runs.
10. vLLM WSL2 teacher scoring on RTX 4090 requires a HARD MINIMUM `--gpu-memory-utilization 0.91` to initialize dequantized bfloat16 weights. Never lower or suggest lowering below 0.91.

## Status (2026-09-13, v3.0 OLMo-2B Transition Landed)

- **OLMo-2B Architecture & Dispatch: DONE.** 41 layers (10 blocks $\times$ 4 + 1 gather layer), $d_{\text{model}} = 2048$, $d_{\text{ff}} = 7168$, 16 Q heads, 4 KV heads ($d_{\text{head}} = 128$, 4:1 GQA). Dynamic SWA (4k/4k/8k), learned sink logits ($-10.0$), multi-$\theta$ globals (Blocks 1–5: 1M; Blocks 6–10+Gather: 5M), p-RoPE @ 0.25, globals-only AttnRes (11 connection points).
- **Engram Module for OLMo: DONE.** Canonical map built from OLMo-3.1 BPE ($|V| = 100,278 \to |V'| = 75,869$, 24,409 merged), pinned SHA-256 `"8bb4fa0dff964ec53035255a27c3dc6b635a4709aabda449105828ad4c6409ad"`. Sized unigram modulus to $M_1 = 100,279$ prime (zero collisions guaranteed).
- **Scale-Normalized SVD Extraction ($\alpha = 0.7$): DONE.** Balances $49\times$ energy disparity between $E$ and $H$. Preserves **$63.18\%$ of $H$ variance** and **$57.62\%$ of $E$ variance**. Orthonormal projection matrix $P \in \mathbb{R}^{2048 \times 5120}$ verified.
- **Base Checkpoint Exported: DONE.** Saved to `exports/olmo-2b-base-v1/` ($4.28\text{ GB} + 1.02\text{ GB}$ model safetensors + $2.25\text{ GB}$ Engram tables). Verified with end-to-end 41-layer forward pass producing finite logits `[1, 128, 100278]`.
- **Training Configuration: DONE.** `train/configs/real_olmo_2b_v1.yaml` configured for 12,735 steps, 8k sequence length, grad accum 4, `adamw8bit`, `freeze_embeddings: true`, `init: prebuilt`.
- **Tests: 111/111 PASSED.** Full test suite green.
