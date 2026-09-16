# OLMo-2B Data Production Pipeline — RUNBOOK

This is the runbook for the data-production pipeline producing top-$k$ ($K=32$)
distillation shards for **OLMo-2B (Spec v3.0)**. Run every command from the repo
root so `data_pipeline` and `train` import correctly.

## What you need before starting

- **Teacher Model (Local GGUF)**:
  `QuantizedModel/Elbaz-Olmo-3-7B-Instruct-abliterated/Elbaz-OLMo-3-7B-Instruct-abliterated-Q8_0.gguf` (7.76 GB)
- **Tokenizer**:
  `QuantizedModel/Elbaz-Olmo-3-7B-Instruct-abliterated/` ($V = 100,278$, matching donor and student)
- **Inference Server**:
  `LlamaCPPBinaries/llama-server.exe` (runs on RTX 4090 with `--flash-attn on --no-jinja` at ~712 tok/s prefill)

## Stage 4 — Bulk Corpus Scoring (Wikipedia + FineWeb-Edu)

Stream open web corpora in $T = 8192$ chunks, score top-$32$ predictive distributions with the local 7B Q8_0 GGUF, and flush lossless `.npz` shards:

```powershell
python -m data_pipeline.run_bulk_score --chunks-per-npz 128 --out-dir data_pipeline/bulk_out
```

Convert `.npz` shards to binary mmap shards (`tokens.bin`, `topk_indices.bin`, `topk_probs.bin`, `tail_w.bin`, `loss_mask.bin`):

```powershell
python -m train.src.tools.npz_converter data_pipeline/bulk_out/bulk_00000.npz train/data/shards/bulk_00000
```

## Configure

Copy the template and fill in the secrets:

```powershell
cp data_pipeline/config.yaml data_pipeline/config.local.yaml
```

Edit `data_pipeline/config.local.yaml`:

```yaml
interrogate:
  teacher:
    provider: openai
    base_url: "https://api.example.com/v1"
    api_key: "${TEACHER_API_KEY}"   # or paste directly; exported env var is safer
    model: "llama-3.3-70b-instruct"

  student:
    # Use `native_chat` for LM Studio's `/api/v1/chat` endpoint when the
    # OpenAI-style endpoint returns empty content for thinking models.
    provider: native_chat
    base_url: "http://localhost:1234"
    api_path: "api/v1/chat"
    api_key: "lm-studio"
    model: "gemma-4-31b-it-abliterated@iq3_s"

  seeds: "data_pipeline/seeds.jsonl"
```

All subsequent commands use `--config data_pipeline/config.local.yaml`.

## Stage 1 — Interrogation loop

Gemma (student) asks follow-ups; the 70B API answers. Raw transcripts are
appended immediately to `data_pipeline/raw/conversations.jsonl`.

```powershell
python -m data_pipeline.run_interrogate --config data_pipeline/config.local.yaml
```

Resume: re-run the same command. The manifest skips already-completed seeds.

## Stage 3 — Format & mask

Render transcripts with the exact Llama-3.1 chat template, tokenize, and build
a per-position loss mask (1 = assistant content, 0 = user/system/scaffolding).

```powershell
python -m data_pipeline.run_format --config data_pipeline/config.local.yaml
```

Output: `data_pipeline/formatted/formatted.jsonl`.

This stage asserts token-id identity: re-tokenizing the rendered text reproduces
the stored ids exactly. Failures are logged and skipped.

## Stage 4 — Anchor scoring

The default config uses the PyTorch donor backend (`backend: pytorch`) because
the current `LlamaCPPBinaries/llama-server.exe` build returns empty
`prompt_probabilities` and cannot score existing prompt tokens. The PyTorch
backend loads `OriginalModel` (Llama-3.1-8B-Instruct bf16) and scores the full
formatted token stream in one forward pass.

```powershell
python -m data_pipeline.run_score --config data_pipeline/config.local.yaml
```

Output: `data_pipeline/shards/class_b_v1/` containing
`tokens.npy`, `topk_idx.npy`, `topk_w.npy`, `tail_w.npy`, `doc_id.npy`,
`loss_mask.npy`, and `sidecar.json`.

The script validates the shard with the training loader's mass invariant
(Σ topk_w + tail_w ≈ 1) before exiting.

### Backend notes

- `pytorch` (default, tested): uses the HF donor checkpoint. Fits on the 4090
  for inference (~16 GB) and is slower than llama.cpp but produces correct shards.
- `llama_server`: requires a `llama-server.exe` build that returns
  `prompt_probabilities` for prompt tokens via `/completion`. The current
  `LlamaCPPBinaries` build does not; do not use this backend until a compatible
  binary is available.

## Validate a shard

At any point:

```powershell
python -m data_pipeline.validate_shards data_pipeline/shards/class_b_v1
```

This opens the shard with the existing `TopKShard` loader and checks dtypes,
shapes, and the mass invariant.

## Typical dry-run workflow

A minimal 3-seed dry run is preconfigured in `data_pipeline/config.dryrun.local.yaml`
(production secrets are inherited from `config.local.yaml`; keep both gitignored).

```powershell
python -m data_pipeline.run_interrogate --config data_pipeline/config.dryrun.local.yaml
python -m data_pipeline.run_format    --config data_pipeline/config.dryrun.local.yaml
python -m data_pipeline.run_score     --config data_pipeline/config.dryrun.local.yaml
python -m data_pipeline.validate_shards data_pipeline/shards.dryrun/class_b_v1
```

For a custom dry run:

1. Create 10–20 seed questions in `data_pipeline/seeds.jsonl`.
2. Run Stage 1 with `max_conversations: 10`.
3. Run Stage 3.
4. Run Stage 4 with `max_conversations: 0` (all) or a small limit.
5. Inspect `data_pipeline/common.log` and validate the shard.
6. Eyeball-review `data_pipeline/raw/conversations.jsonl` before mass production.

## Operational notes

- **Resume point:** `data_pipeline/raw/conversations.jsonl` is the source of
  truth. Stages 3 and 4 are pure functions of it. On resume, Stage 1 rewrites
  the JSONL to match the manifest, dropping any orphan or duplicate lines left
  by a crash between append and manifest save.
- **API cost:** log API tokens/hour and running cost per batch in the dataset
  card. The local 8B scorer is the bottleneck in the PyTorch fallback; budget
  GPU time accordingly.
- **Windows:** the `llama_server` backend spawns `llama-server.exe`. If a
  previous run left an orphan server, set `score.kill_existing_server: true`.
- **Backend:** the default `pytorch` backend was verified end-to-end. The
  `llama_server` backend is implemented but the current `LlamaCPPBinaries`
  build returns empty prompt logprobs, so it is not usable without a compatible
  binary.
- **Troubleshooting:** server stderr is written to `%TEMP%/llama_server_<port>.log`.

## Next: mass production

After the dry run passes:

1. Replace `seeds.jsonl` with the full subject list (D-p2).
2. Set `interrogate.max_conversations` to the desired volume
   (suggested first batch: enough conversations for 5–10M assistant tokens).
3. Run stages 1 → 3 → 4 in order.
4. Record preflight numbers (top-1 agreement, top-10 overlap, KL) in the
  dataset card per spec §6.1.
