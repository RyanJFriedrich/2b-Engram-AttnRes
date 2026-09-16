# Second Training Run: Step 101–105 Resumption (Optimized)

Resumed on **NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, SM120)** from `ckpt_step100.pt`.

Optimizations active:
* **Stage 1**: Native FlashAttention SDPA for Global/Gather layers, Block Checkpointing (33 layer blocks instead of 66 sublayer boundaries), and FP8 Weight Caching across micro-batches.
* **Stage 2**: BF16 Master Weights (`master_dtype: bf16`), reducing checkpoint size from 49 GiB to 34 GiB and peak resident state to 77.20 GiB.

Full technical documentation, bug fixes, and memory profiling: see [`docs/BlackwellOptimizationResults.md`](../BlackwellOptimizationResults.md).

```
[2026-09-04 12:05:14] [INFO] prebuilt base checkpoint loaded: exports/llama-9b-base-v1
[2026-09-04 12:05:15] [INFO] warm start: 168/168 donor tensors, 96 new params
[2026-09-04 12:05:32] [INFO] engram loaded: tables=256128x512/1048608x512
[2026-09-04 12:05:40] [INFO] checkpoint loaded: ckpt_step100.pt (resuming from step 100, loss 7.0554, ema 6.9442)
[2026-09-04 12:05:40] [INFO] resumed training from step 100
[2026-09-04 12:07:08] [INFO] step 101/105 loss 6.8967 lr 1.25e-04 gnorm 12.396 window 4096 theta 1.000 tok/s 462 eng_round 0.054
[2026-09-04 12:08:19] [INFO] step 102/105 loss 6.7812 lr 1.24e-04 gnorm 11.954 window 4096 theta 1.000 tok/s 463 eng_round 0.054
[2026-09-04 12:09:30] [INFO] step 103/105 loss 6.9421 lr 1.22e-04 gnorm 13.102 window 4096 theta 1.000 tok/s 463 eng_round 0.053
[2026-09-04 12:10:41] [INFO] step 104/105 loss 6.7405 lr 1.21e-04 gnorm 12.085 window 4096 theta 1.000 tok/s 462 eng_round 0.053
[2026-09-04 12:11:52] [INFO] step 105/105 loss 6.9151 lr 1.19e-04 gnorm 12.550 window 4096 theta 1.000 tok/s 462 eng_round 0.053
[2026-09-04 12:12:04] [INFO] checkpoint saved: train/runs/real_base_v1/ckpt_step105.pt (step 105)
```

### Key Metrics
* **Loss**: `7.0554` (step 100) $\rightarrow$ `6.9151` (step 105), $ema: 6.8584$
* **Peak VRAM**: 77.20 GiB (Allocated: 44.12 GiB, Reserved: 77.80 GiB)
* **Effective Throughput**: ~1,850 tokens/sec across 4 micro-batches
* **Checkpoint Saved**: `train/runs/real_base_v1/ckpt_step105.pt` (34 GiB)

---

## Batch Size Tuning Probe: Steps 106–108 (`batch_size: 2, grad_accum: 2`)

Resumed from `ckpt_step105.pt` on the Blackwell box with doubled micro-batch size:

```
[2026-09-04 12:28:16] [INFO] checkpoint loaded: train/runs/real_base_v1/ckpt_step105.pt (resuming at step 105)
[2026-09-04 12:29:34] [INFO] step 106/108 (98.1%) loss 6.9066 (ema 6.8632) lr 1.18e-04 gnorm 6.960 tok/s 422 (ema 421) mem_alloc 44.34GiB reserved 92.39GiB peak 90.02GiB
[2026-09-04 12:30:42] [INFO] step 107/108 (99.1%) loss 6.8057 (ema 6.8574) lr 1.16e-04 gnorm 9.108 tok/s 481 (ema 427) mem_alloc 44.34GiB reserved 92.39GiB peak 90.02GiB
[2026-09-04 12:31:50] [INFO] step 108/108 (100.0%) loss 6.9221 (ema 6.8639) lr 1.15e-04 gnorm 8.451 tok/s 481 (ema 433) mem_alloc 44.34GiB reserved 92.39GiB peak 90.02GiB
[2026-09-04 12:32:12] [INFO] checkpoint saved: train/runs/probe_bs2/ckpt_step108.pt (step 108)
```

* **Step Time**: 68.0 seconds / step (vs 71.0 seconds at `batch_size: 1`) $\rightarrow$ ~4.2% faster.
* **Effective Throughput**: ~1,925 tokens/sec.
* **Peak VRAM**: 90.02 GiB in PyTorch (95.28 GiB / 93.0 GiB in `nvidia-smi`), leaving only ~2.6 GiB headroom.
* **Takeaway**: Confirmed that `batch_size: 1, grad_accum: 4` is the safer, optimal configuration for long-running stability.

