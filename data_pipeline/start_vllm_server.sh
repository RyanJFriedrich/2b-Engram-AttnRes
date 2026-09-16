#!/usr/bin/env bash
# ==============================================================================
# vLLM Distillation Scorer Server Launch Script (WSL2 / RTX 4090 24GB)
# Model: Llama-3.3-8B-Instruct-Abliterated-Q8_0.gguf
# ==============================================================================

set -euo pipefail

# 1. Environment & Path configuration
source /home/ophidian/.vllm_env/bin/activate
export PATH="/usr/local/cuda/bin:/home/ophidian/.vllm_env/bin:/home/ophidian/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# 2. Performance & Hardware acceleration flags
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
# Dequantize GGUF weights to bfloat16 on load -> unlocks cuBLAS tensor cores (~12x speedup)
export VLLM_GGUF_DEQUANT_ON_LOAD=1
# Fast direct binary disk dump directory (bypasses HTTP/JSON serialization overhead)
export VLLM_RAW_DUMP_DIR="/mnt/j/Kimi/projects/Llama-3-Rebuild/data_pipeline/bulk_out/raw_chunks"

mkdir -p "$VLLM_RAW_DUMP_DIR"

MODEL_PATH="/home/ophidian/win_models/Elbaz-OLMo-3-7B-Instruct-abliterated-Q8_0.gguf"
TOKENIZER_PATH="/mnt/j/Kimi/projects/Llama-3-Rebuild/QuantizedModel/Elbaz-Olmo-3-7B-Instruct-abliterated"
PORT=8390

echo "======================================================================"
echo "Starting vLLM server on port ${PORT}..."
echo "Model:     ${MODEL_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Raw Dump:  ${VLLM_RAW_DUMP_DIR}"
echo "======================================================================"

exec vllm serve "${MODEL_PATH}" \
    --load-format gguf \
    --tokenizer "${TOKENIZER_PATH}" \
    --hf-config-path "${TOKENIZER_PATH}" \
    --max-model-len 8200 \
    --max-num-batched-tokens 8200 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --max-logprobs 32 \
    --port "${PORT}"
