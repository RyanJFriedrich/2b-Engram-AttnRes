"""High-Speed Bulk Corpus Distillation Scorer via vLLM.

Accelerates KD logprob generation using dequantized bf16 cuBLAS tensor cores
and native PyTorch top-k. Reaches ~5,100 tokens/sec on RTX 4090.

Supports:
  1. Offline RedPajama JSONL data with exact length compliance:
     - Documents >= 8k: trimmed to exactly 8k (remainder discarded)
     - Documents < 8k: right-padded to 8k with <|finetune_right_pad_id|>
     - loss_mask: 0 on pos 0, 1 on doc tokens, 0 on padded tokens
  2. Live Wikipedia + FineWeb-Edu stream (legacy mode)
  3. Execution modes:
     - In-process: direct vllm.LLM engine (WSL)
     - Server: connects to vLLM server via HTTP with direct binary disk dump hook

NPZ contract per shard file (docs/NPZFormat.md):
  tokens        u32 [N]         raw token ids (8192 per chunk)
  teacher_ids   i32 [N,K]       row t predicts tokens[t]. Slot 0 = GT with TRUE prob;
                                slots 1..K-1 = teacher top-K minus GT, shuffled intact pairs.
                                Masked rows carry -1 in slot 0.
  teacher_probs f32 [N,K]       true softmax mass, NOT renormalized
  loss_mask     u8  [N]         1 on active doc tokens (except pos 0), 0 on pos 0 and padding
  chunk_start   i64 [C], chunk_length i64 [C]
"""
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# Ensure cuda and venv bin are on PATH
os.environ["PATH"] = "/usr/local/cuda/bin:/home/ophidian/.vllm_env/bin:" + os.environ.get("PATH", "")
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
os.environ["VLLM_GGUF_DEQUANT_ON_LOAD"] = "1"

DEFAULT_RAW_DIR = Path("data_pipeline/bulk_out/raw_chunks")
os.environ["VLLM_RAW_DUMP_DIR"] = str(DEFAULT_RAW_DIR.resolve())

DOC_SEP = 128001          # <|end_of_text|>
DEFAULT_PAD_ID = 128004   # <|finetune_right_pad_id|>
STOP = False


def sigint_handler(sig, frame):
    global STOP
    print("\n[SIGINT received] Gracefully finishing current chunk and saving...", flush=True)
    STOP = True


signal.signal(signal.SIGINT, sigint_handler)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["inprocess", "server"], default="inprocess",
                   help="In-process LLM engine (WSL) or connect to running vLLM server")
    p.add_argument("--server-url", default="http://127.0.0.1:8390/v1/completions",
                   help="vLLM server completions URL when --mode=server")
    p.add_argument("--data-source", choices=["redpajama", "stream"], default="redpajama",
                   help="Input data: local RedPajama JSONL shards or live HF stream")
    p.add_argument("--data-dir", default="data_pipeline/DownloadedData/RedPajama",
                   help="Directory containing RedPajama shards (Part1, Part2)")
    p.add_argument("--model", default="/home/ophidian/win_models/Llama-3.3-8B-Instruct-Abliterated-Q8_0.gguf",
                   help="Path to GGUF model")
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--hf-config-path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--seq-len", type=int, default=8192, help="Tokens per chunk")
    p.add_argument("--pad-token-id", type=int, default=DEFAULT_PAD_ID,
                   help="Token ID for right-padding chunks < seq_len (default: 128004)")
    p.add_argument("--k", type=int, default=32, help="Top-K logprob columns")
    p.add_argument("--num-chunks", type=int, default=None, help="Stop after N chunks (default: all)")
    p.add_argument("--chunks-per-npz", type=int, default=128, help="Chunks packed per bulk_XXXXX.npz shard")
    p.add_argument("--out-dir", default="data_pipeline/bulk_out")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--fresh", action="store_true", help="Ignore existing manifest and start fresh")
    # Stream options (if --data-source=stream)
    p.add_argument("--wiki-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min-doc-chars", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------- Data streams
def redpajama_stream(args, tokenizer):
    """Walk local RedPajama JSONL files.
    - If doc tokens > seq_len: trim to seq_len
    - If doc tokens < seq_len: pad to seq_len with pad-token-id
    Yields (chunk_ids, valid_len, doc_index, file_name).
    """
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("**/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {data_dir.resolve()}")
    print(f"[Data] Found {len(files)} RedPajama JSONL files across {data_dir.name}.")

    doc_idx = 0
    pad_id = args.pad_token_id
    seq_len = args.seq_len

    for fpath in files:
        if STOP:
            break
        rel_path = fpath.relative_to(data_dir)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if STOP:
                    break
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                text = d.get("text", "")
                if not text:
                    continue

                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if not ids:
                    continue

                orig_len = len(ids)
                if orig_len >= seq_len:
                    chunk = ids[:seq_len]
                    valid_len = seq_len
                else:
                    valid_len = orig_len
                    chunk = ids + [pad_id] * (seq_len - orig_len)

                yield chunk, valid_len, doc_idx, str(rel_path)
                doc_idx += 1


def hf_stream(args, tokenizer):
    """Legacy: seeded interleave of Wikipedia + FineWeb-Edu."""
    from datasets import load_dataset
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    it_wiki, it_fw = iter(wiki), iter(fw)
    rng = random.Random(args.seed)
    buf = []
    chunk_idx = 0

    while not STOP:
        it = it_wiki if rng.random() < args.wiki_frac else it_fw
        try:
            doc = next(it)
        except StopIteration:
            continue
        text = doc.get("text", "")
        if len(text) < args.min_doc_chars:
            continue

        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buf.append(DOC_SEP)
        while len(buf) >= args.seq_len:
            yield buf[:args.seq_len], args.seq_len, chunk_idx, "hf_stream"
            buf = buf[args.seq_len:]
            chunk_idx += 1


def get_chunk_stream(args, tokenizer):
    if args.data_source == "redpajama":
        return redpajama_stream(args, tokenizer)
    else:
        return hf_stream(args, tokenizer)


# ---------------------------------------------------------------- Main Loop
def main():
    global STOP
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_chunks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_RAW_DUMP_DIR"] = str(raw_dir.resolve())

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.fresh:
        manifest = json.loads(manifest_path.read_text())
        skip_chunks = manifest.get("stream_cursor", manifest.get("total_chunks", 0))
        shard_idx = len(manifest["shards"])
        print(f"Resuming from manifest: cursor at chunk {skip_chunks}, "
              f"{manifest.get('total_chunks', 0)} chunks scored, {shard_idx} shards written.")
    else:
        manifest = {
            "config": vars(args),
            "shards": [],
            "total_tokens": 0,
            "total_chunks": 0,
            "stream_cursor": 0,
        }
        skip_chunks, shard_idx = 0, 0

    print("Loading tokenizer:", args.tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    llm = None
    sp = None
    if args.mode == "inprocess":
        print(f"Initializing in-process vLLM engine ({args.model})...")
        from vllm import LLM, SamplingParams
        t0_load = time.time()
        llm = LLM(
            model=args.model,
            load_format="gguf",
            tokenizer=args.tokenizer,
            hf_config_path=args.hf_config_path,
            max_model_len=args.seq_len + 8,
            max_num_batched_tokens=args.seq_len + 8,
            max_num_seqs=4,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_dtype="fp8",
            enforce_eager=True,
            max_logprobs=args.k,
        )
        print(f"vLLM engine initialized in {time.time() - t0_load:.2f}s!")
        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=args.k,
        )
    else:
        print(f"Server mode selected: sending requests to {args.server_url}")

    chunks = get_chunk_stream(args, tokenizer)
    if skip_chunks:
        print(f"Fast-forwarding stream past {skip_chunks} chunks...")
        t0_ff = time.time()
        for i in range(skip_chunks):
            next(chunks)
            if (i + 1) % 1000 == 0:
                print(f"  Fast-forwarded {i + 1}/{skip_chunks} chunks...")
        print(f"Fast-forwarded {skip_chunks} chunks in {time.time() - t0_ff:.1f}s.")

    # Shard accumulators
    tok_acc, ids_acc, probs_acc, mask_acc = [], [], [], []
    c_start, c_len = [], []
    cursor = skip_chunks

    def flush_shard():
        nonlocal shard_idx
        manifest["stream_cursor"] = cursor
        if not tok_acc:
            manifest_path.write_text(json.dumps(manifest, indent=2))
            return
        shard_path = out_dir / f"bulk_{shard_idx:05d}.npz"
        print(f"\n[Saving Shard] Compressing {len(c_len)} chunks to {shard_path.name}...")
        t0_save = time.time()
        np.savez_compressed(
            shard_path,
            tokens=np.concatenate(tok_acc).astype(np.uint32),
            teacher_ids=np.concatenate(ids_acc),
            teacher_probs=np.concatenate(probs_acc),
            loss_mask=np.concatenate(mask_acc).astype(np.uint8),
            chunk_start=np.array(c_start, dtype=np.int64),
            chunk_length=np.array(c_len, dtype=np.int64),
        )
        ntok = int(sum(c_len))
        manifest["shards"].append({"file": shard_path.name, "chunks": len(c_len), "tokens": ntok})
        manifest["total_tokens"] += ntok
        manifest["total_chunks"] += len(c_len)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved {shard_path.name} ({shard_path.stat().st_size / 1e6:.1f} MB, {ntok:,} tokens) in {time.time() - t0_save:.1f}s.")
        shard_idx += 1
        tok_acc.clear(); ids_acc.clear(); probs_acc.clear(); mask_acc.clear()
        c_start.clear(); c_len.clear()

    n_done = 0
    t_start = time.time()

    try:
        for chunk, valid_len, doc_idx, src_info in chunks:
            if STOP:
                break
            if args.num_chunks is not None and n_done >= args.num_chunks:
                break

            t0_chunk = time.time()
            before_files = set(raw_dir.glob("*.npz"))

            if args.mode == "inprocess":
                llm.generate([chunk], sp, use_tqdm=False)
            else:
                body = json.dumps({
                    "prompt": chunk,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": args.k,
                }).encode("utf-8")
                req = urllib.request.Request(args.server_url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    resp.read()

            dt = time.time() - t0_chunk
            tok_s = (args.seq_len - 1) / dt

            after_files = set(raw_dir.glob("*.npz"))
            new_files = list(after_files - before_files)
            if not new_files:
                new_files = sorted(raw_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)[:1]

            raw_file = new_files[0]
            d = np.load(raw_file)

            tokens = d["tokens"]
            teacher_ids = d["teacher_ids"].copy()
            teacher_probs = d["teacher_probs"].copy()
            loss_mask = d["loss_mask"].copy()

            # Apply padding & masking contract:
            # - Pos 0: loss_mask = 0, teacher_ids[0, 0] = -1
            # - Real tokens (1..valid_len-1): loss_mask = 1, teacher_ids[t, 0] = GT
            # - Padded tokens (valid_len..seq_len-1): loss_mask = 0, teacher_ids[t, 0] = -1
            loss_mask[0] = 0
            teacher_ids[0, 0] = -1
            if valid_len < args.seq_len:
                loss_mask[valid_len:] = 0
                teacher_ids[valid_len:, 0] = -1
                teacher_ids[valid_len:, 1:] = 0
                teacher_probs[valid_len:, :] = 0.0

            unmasked = (loss_mask == 1)
            gt_match = (teacher_ids[unmasked, 0] == tokens[unmasked].astype(np.int32)).all()
            if not gt_match:
                raise RuntimeError(f"Ground Truth contract violated in chunk {cursor} (doc {doc_idx})!")

            current_offset = sum(c_len)
            c_start.append(current_offset)
            c_len.append(len(tokens))
            tok_acc.append(tokens)
            ids_acc.append(teacher_ids)
            probs_acc.append(teacher_probs)
            mask_acc.append(loss_mask)

            cursor += 1
            n_done += 1

            if n_done % 10 == 0 or n_done == 1:
                elapsed = time.time() - t_start
                rate = (n_done * (args.seq_len - 1)) / elapsed
                daily = (rate * 86400) / 1e6
                pad_str = f" [pad: {args.seq_len - valid_len}]" if valid_len < args.seq_len else ""
                print(f"Chunk {n_done} (cursor {cursor}){pad_str} | {dt:.3f}s ({tok_s:.1f} tok/s) | Avg: {rate:.1f} tok/s (~{daily:.1f}M/day)", flush=True)

            if len(c_len) >= args.chunks_per_npz:
                flush_shard()

    finally:
        if c_len:
            flush_shard()

    total_time = time.time() - t_start
    if n_done > 0:
        overall_rate = (n_done * (args.seq_len - 1)) / total_time
        print("\n=======================================================")
        print(f"Scored {n_done} chunks ({n_done * args.seq_len:,} tokens) in {total_time:.1f}s")
        print(f"Overall speed: {overall_rate:.1f} tokens/sec (~{overall_rate * 86400 / 1e6:.1f}M tokens/day)")
        print(f"Total shards written: {shard_idx}")
        print("=======================================================")


if __name__ == "__main__":
    main()
