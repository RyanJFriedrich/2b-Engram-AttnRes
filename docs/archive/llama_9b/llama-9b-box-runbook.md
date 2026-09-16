# Llama-9B — Box Runbook (human edition, v2.2, 2026-09-04)

This walks you from "I just rented the box" to "training is running", with
every command spelled out. Two machines are involved:

- **YOUR PC** = your Windows machine at home (this one). Part 1 happens here.
- **THE BOX** = the rented Linux GPU machine. You do Parts 2–5 there, over SSH.

The concrete addresses (no placeholders — this is what's actually set up):

- **Code (private GitHub):** `https://github.com/RyanJFriedrich/Llama-9b-Trainer.git`
- **Base checkpoint + ckpt_step100.pt (PUBLIC HF):** `Ouroboros-Research/llama-9b-base-v1`
- **Training data (bulk NPZs) (PUBLIC HF):** `Ouroboros-Research/llama-9b-bulk-npz`
- Your HF login is user **Ophidian**, member of the **Ouroboros-Research** org.

Both Hugging Face repositories are **PUBLIC** — anyone can download them without
authentication, enabling full community transparency and hassle-free box downloads.
(Pushing/uploading still requires your authenticated Ophidian write token). The GitHub
repository remains private, so cloning code still requires GitHub credentials.

**How paste-and-run works here:** each grey block is one complete thing to
paste into the terminal and hit Enter. Multi-line blocks are meant to be
pasted all at once.

---

## ⚡ FAST-TRACK: Resuming Step 100 on a Fresh Blackwell Box

If you are spinning up a fresh box and resuming directly from **Checkpoint 100**
with all optimizations active (Stages 1 & 2: FlashAttention SDPA, Block Checkpointing,
FP8 weight caching, and BF16 master weights), paste these blocks in order:

### 1. Setup Python 3.13 and Pinned Stack
```bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv python install 3.13
uv venv --python 3.13 ~/venv
source ~/venv/bin/activate
echo 'source ~/venv/bin/activate' >> ~/.bashrc
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers==5.7.0 numpy==2.5.1 safetensors==0.7.0 PyYAML==6.0.3 pytest==8.3.0 hf-transfer huggingface-hub
```

### 2. Clone Code & HF Auth
```bash
git clone https://github.com/RyanJFriedrich/Llama-9b-Trainer.git llama-9b
cd llama-9b
hf auth login
```
*(Enter your HF token when prompted. Note: GitHub username is your GH username, and password is your GitHub Personal Access Token).*

### 3. Download Checkpoint & Training Shards
```bash
# Downloads base checkpoint and ckpt_step100.pt (~34 GiB):
hf download Ouroboros-Research/llama-9b-base-v1 --repo-type model --local-dir exports/llama-9b-base-v1

# Downloads 81 bulk NPZs (~15 GiB):
hf download Ouroboros-Research/llama-9b-bulk-npz --repo-type dataset --local-dir data_pipeline/bulk_out
```
*(Note: You do NOT need `OriginalModel` when resuming from `ckpt_step100.pt`).*

### 4. Convert Data Shards & Fill Config
```bash
python -m train.scripts.convert_bulk
python -m train.scripts.fill_smoke_shards --configs train/configs/real_base_v1.yaml
```

### 5. Launch Training inside tmux
```bash
tmux new -s train
source ~/venv/bin/activate
cd ~/llama-9b
python -m train.scripts.train_phase0 \
  --config train/configs/real_base_v1.yaml \
  --resume exports/llama-9b-base-v1/checkpoint-1/ckpt_step100.pt
```

*(Detach from tmux anytime with `Ctrl+B`, then `D`. Monitor with `tail -f train/runs/real_base_v1/real_base_v1.log` or `nvidia-smi`).*

---

## Part 0 — one-time accounts (already mostly done)

1. ~~Create the GitHub repo~~ — done: `RyanJFriedrich/Llama-9b-Trainer` (private).
2. Hugging Face token: you already have one on your PC (logged in as Ophidian).
   You'll paste the same token into the box's login prompt in Part 2. If you
   ever need a new one: hf.co → Settings → Access Tokens → Create (type "Write").
3. GitHub credential for THE BOX: cloning a private repo over HTTPS asks for a
   username + password — the "password" must be a **GitHub personal access
   token**, not your real password. Make one at github.com → Settings →
   Developer settings → Personal access tokens (repo scope is enough). Keep it
   handy for Part 2.
4. Make sure your HF account has the Llama 3.1 license accepted:
   hf.co/meta-llama/Meta-Llama-3.1-8B-Instruct → accept if prompted.

---

## Part 1 — publish from YOUR PC (DONE — reference for re-runs only)

The initial publish is already complete: the code was pushed to GitHub and
both HF repos were created and uploaded from this machine. You only need this
part again to **top up data** or push code changes.

**1a. Push code changes** (from `J:\Kimi\projects\Llama-3-Rebuild`). Only the
`train/` folder is tracked — data, docs, and weights never go to GitHub.

```bash
git push
```

**1b. Top up the training data** after the pipeline produces more NPZs
(re-running uploads only the new/changed files):

```bash
hf upload Ouroboros-Research/llama-9b-bulk-npz data_pipeline/bulk_out --repo-type dataset
```

**1c. Re-upload the base checkpoint** (only if it's ever rebuilt — normally
this is a once-ever artifact):

```bash
hf upload Ouroboros-Research/llama-9b-base-v1 exports/llama-9b-base-v1
```

---

## Part 2 — set up THE BOX (~15 min + downloads)

Box this runbook was validated on (2026-09-03): **Massed Compute dedicated,
RTX PRO 6000 Blackwell 96 GB, $1.6425/hr**, plain Ubuntu 22.04 image (Python
3.10, no PyTorch — hence the uv/venv flow below), 713 GB disk, 141 GiB RAM.
A future **Verda** box will differ slightly in environment — expect to re-debug
2a there; everything else should carry over.

SSH into the box. Every command below runs in that SSH session.

**2a. PATH fix + Python environment.** The box is plain Ubuntu (Python 3.10,
no PyTorch) — too old for the pinned stack, so we build an isolated Python
3.13 environment with `uv` (no sudo needed). Paste the whole block:

```bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv python install 3.13
uv venv --python 3.13 ~/venv
source ~/venv/bin/activate
echo 'source ~/venv/bin/activate' >> ~/.bashrc
```

From now on your prompt starts with `(venv)` — that's how you know the right
Python is active. Every runbook command assumes it. Then the pinned stack
(torch comes from PyTorch's CUDA-13 wheel index, ~3 GB):

```bash
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers==5.7.0 numpy==2.5.1 safetensors==0.7.0 PyYAML==6.0.3 pytest==8.3.0
```

Verify — must print `2.10.0+cu130 True NVIDIA RTX PRO 6000 ...`:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**2b. Get the code and log into HF:**

```bash
git clone https://github.com/RyanJFriedrich/Llama-9b-Trainer.git llama-9b
cd llama-9b
hf auth login
```

The clone asks for a username and password — username is your GitHub
username, the "password" is the **GitHub token** from Part 0, step 3 (not
your real password). Then paste your Hugging Face token into the `hf auth
login` prompt — the repos are private, so no token = no downloads in 2c.
From here on, **every command is run from inside the `llama-9b` folder** — if
you ever open a fresh SSH session, `cd llama-9b` first.

**2c. Download the three big things** (runs a while; that's normal):

```bash
hf download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir OriginalModel
hf download Ouroboros-Research/llama-9b-base-v1 --repo-type model --local-dir exports/llama-9b-base-v1
hf download Ouroboros-Research/llama-9b-bulk-npz --repo-type dataset --local-dir data_pipeline/bulk_out
```

What they are, in order: the donor 8B model (used to initialize and to
double-check sanity), your untrained 9B base checkpoint, and the training
data. If one gets interrupted, run the exact same line again — it resumes.

You should see: three folders exist — `OriginalModel/`, `exports/llama-9b-base-v1/`,
`data_pipeline/bulk_out/` (full of `bulk_*.npz` files). Quick check:

```bash
ls OriginalModel
ls data_pipeline/bulk_out | head
```

**2d. Convert the data into the format training reads.** One command; it
converts every NPZ and skips any already done, so re-running is always safe.
Expect roughly the same size on disk again (~15 GB) and a few minutes per GB.

```bash
python -m train.scripts.convert_bulk
```

**2e. Point the two smoke-test configs at the converted data:**

```bash
python -m train.scripts.fill_smoke_shards
```

You should see: two lines saying how many shard folders were written into
each config. If it says "no converted shard dirs", step 2d didn't finish.

**Disk budget on a ~768 GB box** (what everything costs, so nothing surprises
you later): donor ~16 GB + base checkpoint ~18 GB + NPZ ~15 GB + converted
shards ~15–30 GB + each training checkpoint **~50 GB** (fp32 masters + 8-bit
optimizer state). The two smoke runs keep 2 checkpoints each — delete
`train/runs/bringup_*_smoke/` after the overlay comparison; they're meant to
be discarded. Keep only the latest 1–2 checkpoints of any real run.

---

## Part 3 — health checks (THE BOX, ~15 min)

Run these in order. If one fails, stop — don't skip ahead.

**3a. Preflight** — checks weights present, GPU, disk, RAM, torch, and the
Engram canon checksum. Must end in **PASS**. (The `--min-disk-gb 500`
matters: preflight's default demands 1 TB free, which this box's 713 GB disk
can never satisfy — 500 still leaves headroom for checkpoints, which are
~50 GB each. As of bring-up the box had 587 GiB free.)

```bash
python -m train.scripts.preflight --model-config train/configs/model/llama_9b_engram_v1.yaml --min-disk-gb 500
```

**3b. GPU self-check** — verifies the FP8 math path and the compiler on this
specific GPU model (it was only proven on a different one at home). Takes a
few minutes; must end in **passed**:

```bash
python -m pytest train/tests/test_fp8.py train/tests/test_torch_compile.py -q
```

You should see: `8 passed` (some may say skipped — that's OK, passed is what matters).

**3c. Step-0 sanity** — loads the donor, runs a fixed probe, checks the
untrained model's starting loss is in the expected band:

```bash
python -m train.scripts.step0_sanity --model-config train/configs/model/llama_9b_engram_v1.yaml
```

---

## Part 4 — the two smoke runs (THE BOX, hours)

First, make sure the box has the latest code (fixes land between sessions):

```bash
cd ~/llama-9b && git pull
```

One fragmentation-insurance setting, then the runs:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

These are the real test: does it train, and is FP8 as good as bf16 but faster.
Each is 200 steps (~6.6M tokens). **Run them inside tmux so a dropped SSH
connection doesn't kill them.** If you've never used tmux:

```bash
tmux new -s train
```

Now you're inside a protected session. Later: press `Ctrl+B`, then `D` to
leave it running and detach; `tmux attach -t train` to come back.

**4a. bf16 reference run** (the baseline; its checkpoint gets discarded):

```bash
python -m train.scripts.train_phase0 --config train/configs/bringup_bf16_smoke.yaml
```

Watch it any time with:

```bash
tail -f train/runs/bringup_bf16_smoke/train.log
```

(`Ctrl+C` stops *watching*, not the run.) You should see a `loss` number that
starts around 10–11 and trends down, plus a tokens/sec line. Note the tok/s.

**4b. FP8 run** (same data, same seed, FP8 math path):

```bash
python -m train.scripts.train_phase0 --config train/configs/bringup_fp8_smoke.yaml
```

**4c. Compare.** Paste this to see the two loss curves side by side:

```bash
grep "loss " train/runs/bringup_bf16_smoke/train.log | tail -10
grep "loss " train/runs/bringup_fp8_smoke/train.log | tail -10
```

**Acceptance:** the FP8 curve should sit on top of the bf16 curve (same
downward trend, no big gap or blow-up) **and** its tok/s should be clearly
higher. If yes — FP8 becomes the standard path, exactly as planned. If the
curve diverges or tok/s doesn't improve, stop here and report it.

---

## Part 5 — the real run (base v1)

**First, report the overlay result** (the two loss curves + both tok/s
numbers). If the FP8 overlay is accepted, the real run is already written —
`train/configs/real_base_v1.yaml`: 5 epochs over the current 83.5M-token
corpus (12,735 steps), FP8 + torch.compile, initializing from the prebuilt
base checkpoint.

One-time setup, then launch (inside tmux):

```bash
python -m train.scripts.fill_smoke_shards --configs train/configs/real_base_v1.yaml
python -m train.scripts.train_phase0 --config train/configs/real_base_v1.yaml
```

Notes:

- The first step compiles for a few minutes before anything moves — normal.
- Checkpoints drop every 2,500 steps (~50 GB each). Pull them down for
  inspection whenever you like (`hf download`-style transfer or `scp`), then
  **delete old ones off the box** — the trainer never prunes.
- **When more data lands:** run 2c (3rd line) + 2d to convert it, then copy
  the config to `real_base_v2.yaml`, fill its shard list
  (`fill_smoke_shards --configs ...v2.yaml`), and launch with
  `--resume train/runs/real_base_v1/ckpt_step<N>.pt`. That's the continual-
  training loop: new config per data drop, resume carries the model state.

---

## Cheat sheet (THE BOX)

| I want to… | Command |
|---|---|
| Get the latest code on the box | `cd ~/llama-9b && git pull` |
|---|---|
| Watch the current run | `tail -f train/runs/<run_name>/train.log` (Ctrl+C to stop watching) |
| Leave a run going over SSH disconnect | run it inside `tmux new -s train`; reattach with `tmux attach -t train` |
| Check GPU usage | `nvidia-smi` |
| Resume an interrupted run | add `--resume train/runs/<run_name>/ckpt_step<N>.pt` to the train_phase0 command |
| Top up data after more NPZs are produced | on your PC: `hf upload Ouroboros-Research/llama-9b-bulk-npz data_pipeline/bulk_out --repo-type dataset`; on the box: re-run the 3rd download line (2c), then 2d and 2e |

---

## 📌 TODO: Production Hardening (For Next Session)

1. **Make `Ouroboros-Research/llama-9b-base-v1` Public**:
   - In HF repo settings: change visibility from **Private** to **Public**.
   - Removes private LFS storage/bandwidth limits so multiple 34 GiB checkpoints can be safely stored.
   - Simplifies box bring-up by allowing unauthenticated `hf download` of weights.
2. **Async Background Checkpoint Uploader**:
   - Hook into `Trainer.save_checkpoint()` using a background daemon thread (`concurrent.futures.ThreadPoolExecutor(1)` or `threading.Thread`) via `huggingface_hub.HfApi().upload_file()`.
   - Automatically uploads new checkpoints (`ckpt_step*.pt`) to `Ouroboros-Research/llama-9b-base-v1` without blocking training.
   - **Failsafe**: If the rented box disconnects, preempts, or encounters a hardware interruption while unattended, the latest checkpoint is already preserved on Hugging Face.

