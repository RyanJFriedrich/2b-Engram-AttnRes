#!/usr/bin/env bash
source /home/ophidian/.vllm_env/bin/activate
export PATH="/usr/local/cuda/bin:/home/ophidian/.vllm_env/bin:/home/ophidian/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_RAW_DUMP_DIR=/home/ophidian/bulk_out/raw_chunks
export VLLM_GGUF_DEQUANT_ON_LOAD=1

exec vllm serve /home/ophidian/win_models/Elbaz-OLMo-3-7B-Instruct-abliterated-Q8_0.gguf \
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
