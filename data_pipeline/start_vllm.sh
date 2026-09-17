#!/usr/bin/env bash
# ==============================================================================
# vLLM Server Runner Loop (WSL2)
# ==============================================================================

source /home/ophidian/.vllm_env/bin/activate
export PATH="/usr/local/cuda/bin:/home/ophidian/.vllm_env/bin:/home/ophidian/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_RAW_DUMP_DIR=/home/ophidian/bulk_out/raw_chunks
export VLLM_GGUF_DEQUANT_ON_LOAD=1

# Clean shutdown if user hits Ctrl+C in this terminal
trap "echo ''; echo '[start_vllm] SIGINT/SIGTERM received. Exiting server runner loop.'; exit 0" INT TERM

COOL_DOWN_SEC=15

while true; do
    echo "======================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start_vllm] Launching vLLM server..."
    echo "======================================================================"

    # HARD REQUIREMENT: Minimum 0.91 VRAM utilization required for vLLM to initialize dequantized weights on 24GB. NEVER lower below 0.91.
    vllm serve /home/ophidian/win_models/Elbaz-OLMo-3-7B-Instruct-abliterated-Q8_0.gguf \
        --host 0.0.0.0 \
        --load-format gguf \
        --tokenizer /mnt/j/Kimi/projects/Llama-3-Rebuild/QuantizedModel/Elbaz-Olmo-3-7B-Instruct-abliterated \
        --max-model-len 8200 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 8200 \
        --gpu-memory-utilization 0.91 \
        --kv-cache-dtype fp8 \
        --hf-config-path /mnt/j/Kimi/projects/Llama-3-Rebuild/QuantizedModel/Elbaz-Olmo-3-7B-Instruct-abliterated \
        --enforce-eager \
        --max-logprobs 32 \
        --port 8000

    EXIT_CODE=$?
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start_vllm] vLLM server process stopped (exit code: $EXIT_CODE)."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start_vllm] Cooling down ${COOL_DOWN_SEC}s for Windows WDDM VRAM to settle..."
    sleep "$COOL_DOWN_SEC"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [start_vllm] Restarting vLLM..."
done
