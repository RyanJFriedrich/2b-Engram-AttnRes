# OLMo-2B — Operational Runbook (v3.0)

Operational guide for bringing up, validating, and training the **OLMo-2B** architecture across local development (RTX 4090) and deploy boxes (RTX 6000 Pro Blackwell, 96 GB).

---

## 1. Environment & Prerequisites

- **Python Version:** 3.13
- **PyTorch:** $\ge 2.5$ (built with CUDA 12.4+)
- **Transformers:** $\ge 4.45$
- **Harness Root:** Run all commands from repo root (`J:\Kimi\projects\Llama-3-Rebuild`).

### Platform Guidelines: Native Windows vs. WSL2
- **Local Dev Box (RTX 4090, 24 GB VRAM):**
  - **Native Windows:** Runs out of the box with standard PyTorch. VRAM is constrained by $\sim 1.5\text{ GiB}$ OS display overhead, leaving $22.5\text{ GiB}$ for CUDA. Peak VRAM under `master_dtype: bf16` is $\approx 20.8\text{ GiB}$ at $T = 8192$.
  - **WSL2 (Ubuntu):** Highly recommended for full-speed local training. Reclaims the $1.5\text{ GiB}$ VRAM overhead, enables native FlashAttention-2 in SDPA, and unlocks OpenAI Triton kernel compilation under `torch.compile: true`.
- **Deploy Box (RTX 6000 Pro Blackwell, 96 GB VRAM):**
  - Standard NGC PyTorch Linux container. $96\text{ GB}$ VRAM provides massive headroom ($>70\text{ GiB}$ free).

---

## 2. Model Bring-Up & Preflight

### Step 1: Preflight Verification
Verify environment health, GPU visibility, VRAM, and the Engram canonical SHA-256:
```bash
python -m train.scripts.preflight \
  --weights OriginalModel \
  --model-config train/configs/model/olmo_2b_v1.yaml \
  --min-vram-gb 20.0
```

### Step 2: Extract Donor Furniture via Normalized SVD ($\alpha = 0.7$)
If regenerating the base checkpoint from the donor weights (`allenai/Olmo-3.1-32B-Think`):
```bash
python -m train.src.tools.svd_donor \
  --donor OriginalModel \
  --model-config train/configs/model/olmo_2b_v1.yaml \
  --out exports/olmo-2b-base-v1 \
  --alpha 0.7
```
- Output: `exports/olmo-2b-base-v1/` containing model safetensors shards, `engram.safetensors`, and `config.json`.
- Measured variance retention: **$63.18\%$ on $H$**, **$57.62\%$ on $E$**.

### Step 3: Verify Step-0 Model Sanity
Confirm that the base checkpoint loads and executes a clean forward pass:
```bash
python -c "
from train.src.config import load_config
from train.src.model.refit import RefitModel
from train.src.tools.base_ckpt import load_base_checkpoint
import torch

cfg = load_config('train/configs/model/olmo_2b_v1.yaml')
model = RefitModel(cfg)
load_base_checkpoint(model, 'exports/olmo-2b-base-v1')
model.eval()
x = torch.randint(0, cfg.vocab_size, (1, 128))
out = model(x)
print('Base checkpoint verified! Logits shape:', out.shape, 'Finite:', torch.isfinite(out).all().item())
"
```

---

## 3. Training Execution

### Standard Training Run (`real_olmo_2b_v1.yaml`)
Launch the primary training run:
```bash
python -m train.scripts.train_phase0 \
  --config train/configs/real_olmo_2b_v1.yaml
```

### Run Configuration Settings
- `init: prebuilt` + `prebuilt_path: exports/olmo-2b-base-v1`: Loads step-0 furniture from base checkpoint.
- `freeze_embeddings: true`: Keeps `embed_tokens` and `lm_head` frozen.
- `optimizer: adamw8bit`: Quantized 8-bit AdamW optimizer ($4.17\text{ GiB}$ VRAM).
- `master_dtype: bf16`: Keeps master weights in bf16, saving $4.2\text{ GiB}$ of VRAM.
- `grad_dtype: "bf16"`: Gradients accumulate directly in bf16 ($4.17\text{ GiB}$ buffer).
- `grad_accum: 4`: Effective batch size $= 32,768$ tokens/step ($1 \times 8192 \times 4$).

### Resuming an Interrupted Run
Checkpoints are resume-safe bitwise (Invariant I9: weights, 8-bit optimizer states, RNG states, data stream cursors):
```bash
python -m train.scripts.train_phase0 \
  --config train/configs/real_olmo_2b_v1.yaml \
  --resume train/runs/real_olmo_2b_v1/ckpt_step_001000.pt
```

---

## 4. Troubleshooting & Operational Advice

1. **OOM during Training on 24 GB GPU**:
   - Ensure `master_dtype: bf16` and `freeze_embeddings: true` are enabled.
   - If background desktop processes take $>2.5\text{ GiB}$ of VRAM on Windows, switch execution to **WSL2** or reduce `seq_len` to 4096 during local prototyping.
2. **Log File Monitoring**:
   - All runs append timestamped logs to `common.log` and per-run logs in `train/runs/<run_name>/run.log`.
   - Monitor live progress: `Get-Content common.log -Tail 50 -Wait` (PowerShell) or `tail -f common.log` (Linux/WSL).
3. **Running the Unit Test Suite**:
   ```bash
   python -m pytest train/tests -v
   ```

---

## 5. Hugging Face Hub Operations (Manual)

All heavy models and dataset shards are tracked off-git via Hugging Face.

### Authentication
```bash
huggingface-cli login
```

### Model Checkpoint (`Ouroboros-Research/Our1-2b`)

- **Upload Base Checkpoint to Hub:**
  ```bash
  huggingface-cli upload Ouroboros-Research/Our1-2b exports/olmo-2b-base-v1 . --repo-type model
  ```
  *Alternative (Python one-liner):*
  ```bash
  python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='Ouroboros-Research/Our1-2b', folder_path='exports/olmo-2b-base-v1', repo_type='model')"
  ```

- **Download Base Checkpoint (on remote server / GPU node):**
  ```bash
  huggingface-cli download Ouroboros-Research/Our1-2b --local-dir exports/olmo-2b-base-v1 --repo-type model
  ```
  *Alternative (Python one-liner):*
  ```bash
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Ouroboros-Research/Our1-2b', local_dir='exports/olmo-2b-base-v1', repo_type='model')"
  ```

### Dataset Shards (`Ouroboros-Research/Our1-2b-Dataset`)

- **Upload Shards to Hub:**
  ```bash
  huggingface-cli upload Ouroboros-Research/Our1-2b-Dataset data_pipeline/shards . --repo-type dataset
  ```
  *Alternative (Python one-liner):*
  ```bash
  python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='Ouroboros-Research/Our1-2b-Dataset', folder_path='data_pipeline/shards', repo_type='dataset')"
  ```

- **Download Shards (on remote server / GPU node):**
  ```bash
  huggingface-cli download Ouroboros-Research/Our1-2b-Dataset --local-dir data_pipeline/shards --repo-type dataset
  ```
  *Alternative (Python one-liner):*
  ```bash
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Ouroboros-Research/Our1-2b-Dataset', local_dir='data_pipeline/shards', repo_type='dataset')"
  ```
