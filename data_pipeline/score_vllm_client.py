"""High-Speed Bulk Distillation Client for vLLM Server.

Connects to a running vLLM server (WSL or remote) and scores chunks at ~5,100 tok/s.
Supports multiple corpora with extensible sequence lengths (8k, 12k, 16k, etc.)
and features a live TQDM status bar.

Available Corpora:
  - redpajama: Local RedPajama JSONL shards (pads < seq_len, trims > seq_len)
  - wikidict:  Local WikiDictionary headwords formatted into continuous text
  - wiki:      Streaming English Wikipedia (continuous EOS-separated stream)
  - fineweb:   Streaming FineWeb-Edu 10BT (continuous EOS-separated stream)
  - stream:    50/50 seeded interleave of Wikipedia + FineWeb-Edu

Usage:
  # Score RedPajama (default 8k chunks, 128 chunks per bulk shard)
  python data_pipeline/score_vllm_client.py --corpus redpajama

  # Score 500 chunks of WikiDictionary
  python data_pipeline/score_vllm_client.py --corpus wikidict --num-chunks 500

  # Custom sequence length or port
  python data_pipeline/score_vllm_client.py --seq-len 8192 --port 8390
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
from tqdm import tqdm

DOC_SEP = 100257          # <|endoftext|>
DEFAULT_PAD_ID = 100277   # <|pad|>
STOP = False


def sigint_handler(sig, frame):
    global STOP
    tqdm.write("\n[SIGINT received] Gracefully finishing current chunk and saving shard...")
    STOP = True


signal.signal(signal.SIGINT, sigint_handler)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", choices=["redpajama", "wikidict", "wiki", "fineweb", "stream"], default="redpajama",
                   help="Corpus to draw text from (default: redpajama)")
    p.add_argument("--host", default="auto",
                   help="vLLM server host (default: 'auto', tries localhost then WSL IP)")
    p.add_argument("--server-url", default=None,
                   help="vLLM server completions endpoint (default: http://<host>:{port}/v1/completions)")
    p.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000)")
    p.add_argument("--tokenizer", default="QuantizedModel/Elbaz-Olmo-3-7B-Instruct-abliterated",
                   help="Tokenizer path or HF identifier (default: local Elbaz OLMo)")
    p.add_argument("--seq-len", type=int, default=8192,
                   help="Target sequence length in tokens per chunk (default: 8192)")
    p.add_argument("--pad-token-id", type=int, default=DEFAULT_PAD_ID,
                   help="Padding token ID for short documents (default: 100277)")
    p.add_argument("--k", type=int, default=32, help="Top-K logprob columns (default: 32)")
    p.add_argument("--num-chunks", type=int, default=None, help="Stop after N chunks (default: run until exhausted)")
    p.add_argument("--chunks-per-npz", type=int, default=128, help="Chunks packed per bulk_XXXXX.npz (default: 128)")
    p.add_argument("--out-dir", default="data_pipeline/bulk_out", help="Output directory for bulk shards")
    p.add_argument("--raw-dir", default=None, help="Directory where vLLM dumps raw chunks (default: out-dir/raw_chunks)")
    p.add_argument("--redpajama-dir", default="data_pipeline/DownloadedData/RedPajama")
    p.add_argument("--wikidict-file", default="data_pipeline/DownloadedData/WikiDictionary/raw-wiktextract-data.jsonl")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fresh", action="store_true", help="Ignore existing manifest and start from shard 0")
    return p.parse_args()


# ---------------------------------------------------------------- Corpora Streams
def redpajama_stream(args, tokenizer, skip_chunks=0):
    """Local RedPajama JSONL documents:
       - If doc tokens >= seq_len: trim to seq_len
       - If doc tokens < seq_len: pad to seq_len with pad-token-id
    """
    rp_dir = Path(args.redpajama_dir)
    files = sorted(rp_dir.glob("**/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No RedPajama .jsonl files found under {rp_dir.resolve()}")

    seq_len = args.seq_len
    pad_id = args.pad_token_id
    doc_idx = 0
    skipped = 0

    t0_ff = time.time()
    if skip_chunks > 0:
        print(f"Fast-forwarding stream past {skip_chunks} chunks via raw file seek...")

    for fpath in files:
        if STOP:
            break
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if STOP:
                    break
                line = line.strip()
                if not line:
                    continue

                if skipped < skip_chunks:
                    skipped += 1
                    doc_idx += 1
                    if skipped == skip_chunks:
                        print(f"Fast-forwarded {skip_chunks} chunks in {time.time() - t0_ff:.2f}s.")
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

                yield chunk, valid_len, doc_idx, fpath.name
                doc_idx += 1


def wikidict_stream(args, tokenizer):
    """Local WikiDictionary: formats entries and packs continuously into seq_len chunks."""
    dict_file = Path(args.wikidict_file)
    if not dict_file.exists():
        raise FileNotFoundError(f"WikiDictionary file not found at {dict_file.resolve()}")

    buf = []
    chunk_idx = 0
    seq_len = args.seq_len

    with open(dict_file, "r", encoding="utf-8") as f:
        for line in f:
            if STOP:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            word = d.get("word", "")
            pos = d.get("pos", "")
            lang = d.get("lang", "")
            senses = d.get("senses", [])
            etym = d.get("etymology_text", "")

            parts = [f"Word: {word} ({pos}, {lang})"]
            if etym:
                parts.append(f"Etymology: {etym.strip()}")
            glosses = [s["glosses"][0] for s in senses if s.get("glosses")]
            if glosses:
                parts.append("Definitions:")
                for idx, g in enumerate(glosses[:5], 1):
                    parts.append(f"{idx}. {g}")

            entry_text = "\n".join(parts) + "\n\n"
            buf.extend(tokenizer(entry_text, add_special_tokens=False)["input_ids"])

            while len(buf) >= seq_len:
                yield buf[:seq_len], seq_len, chunk_idx, "wikidict"
                buf = buf[seq_len:]
                chunk_idx += 1


def streaming_hf_stream(args, tokenizer, corpus_name):
    """Streaming Wikipedia / FineWeb continuous document streams."""
    from datasets import load_dataset
    rng = random.Random(args.seed)
    seq_len = args.seq_len
    buf = []
    chunk_idx = 0

    if corpus_name == "wiki":
        stream = iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True))
    elif corpus_name == "fineweb":
        stream = iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True))
    else:  # 'stream': 50/50 interleave
        it_w = iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True))
        it_f = iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True))

    while not STOP:
        if corpus_name == "stream":
            it = it_w if rng.random() < 0.5 else it_f
        else:
            it = stream
        try:
            doc = next(it)
        except StopIteration:
            continue
        text = doc.get("text", "")
        if len(text) < 200:
            continue

        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buf.append(DOC_SEP)

        while len(buf) >= seq_len:
            yield buf[:seq_len], seq_len, chunk_idx, corpus_name
            buf = buf[seq_len:]
            chunk_idx += 1


def get_corpus_stream(args, tokenizer, skip_chunks=0):
    if args.corpus == "redpajama":
        return redpajama_stream(args, tokenizer, skip_chunks=skip_chunks)
    elif args.corpus == "wikidict":
        return wikidict_stream(args, tokenizer)
    else:
        return streaming_hf_stream(args, tokenizer, args.corpus)


def resolve_server_host(host: str, port: int) -> str:
    if host != "auto":
        return host
    # Try localhost first
    for candidate in ("127.0.0.1", "localhost"):
        try:
            url = f"http://{candidate}:{port}/health"
            with urllib.request.urlopen(url, timeout=1) as resp:
                data = resp.read()
                if resp.status == 200 and b"TRANSPORT_AUTH_REQUIRED" not in data:
                    return candidate
        except Exception:
            pass
    # Fallback to WSL IP if running under Windows
    try:
        import subprocess
        wsl_ip = subprocess.check_output(["wsl", "hostname", "-I"], text=True, timeout=2).strip().split()[0]
        url = f"http://{wsl_ip}:{port}/health"
        with urllib.request.urlopen(url, timeout=1) as resp:
            if resp.status == 200:
                return wsl_ip
    except Exception:
        pass
    return "127.0.0.1"


# ---------------------------------------------------------------- Main Execution
def main():
    global STOP
    args = parse_args()
    if args.server_url:
        server_url = args.server_url
        health_url = server_url.replace("/v1/completions", "/health")
    else:
        host = resolve_server_host(args.host, args.port)
        server_url = f"http://{host}:{args.port}/v1/completions"
        health_url = f"http://{host}:{args.port}/health"

    # Test server health
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            if resp.status != 200:
                sys.exit(f"Error: Server at {health_url} returned status {resp.status}")
    except Exception as e:
        sys.exit(f"Error: Could not connect to vLLM server at {health_url}. Is the server running?\nDetails: {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out_dir / "raw_chunks"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.fresh:
        manifest = json.loads(manifest_path.read_text())
        if "corpus_cursors" not in manifest:
            manifest["corpus_cursors"] = {}
            # Shards 0..687 were produced from the redpajama stream
            old_cursor = manifest.get("stream_cursor", manifest.get("total_chunks", 0))
            manifest["corpus_cursors"]["redpajama"] = old_cursor

        skip_chunks = manifest["corpus_cursors"].get(args.corpus, 0)
        shard_idx = len(manifest["shards"])
        print(f"Resuming from manifest: [{args.corpus}] cursor at chunk {skip_chunks}, "
              f"{manifest.get('total_chunks', 0)} total chunks scored, {shard_idx} shards written.")
    else:
        manifest = {
            "config": vars(args),
            "corpus": args.corpus,
            "seq_len": args.seq_len,
            "shards": [],
            "total_tokens": 0,
            "total_chunks": 0,
            "stream_cursor": 0,
            "corpus_cursors": {args.corpus: 0},
        }
        skip_chunks, shard_idx = 0, 0

    print(f"Loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    chunks = get_corpus_stream(args, tokenizer, skip_chunks=skip_chunks)
    if skip_chunks and args.corpus != "redpajama":
        print(f"Fast-forwarding [{args.corpus}] stream past {skip_chunks} chunks...")
        t0_ff = time.time()
        for i in range(skip_chunks):
            next(chunks)
            if (i + 1) % 1000 == 0 or (i + 1) == skip_chunks:
                print(f"  Fast-forwarded {i + 1}/{skip_chunks} chunks...")
        print(f"Fast-forwarded {skip_chunks} chunks in {time.time() - t0_ff:.1f}s.")

    # Shard accumulators
    tok_acc, ids_acc, probs_acc, mask_acc = [], [], [], []
    c_start, c_len = [], []
    cursor = skip_chunks

    def flush_shard():
        nonlocal shard_idx
        if "corpus_cursors" not in manifest:
            manifest["corpus_cursors"] = {}
        manifest["corpus_cursors"][args.corpus] = cursor
        manifest["stream_cursor"] = cursor
        manifest["last_corpus"] = args.corpus
        if not tok_acc:
            manifest_path.write_text(json.dumps(manifest, indent=2))
            return
        shard_path = out_dir / f"bulk_{shard_idx:05d}.npz"
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
        tqdm.write(
            f"\n[Saved Shard] {shard_path.name} | {len(c_len)} chunks ({ntok:,} tokens) | "
            f"Size: {shard_path.stat().st_size / 1e6:.1f} MB | Total: {manifest['total_tokens']:,} tokens "
            f"({time.time() - t0_save:.1f}s)"
        )
        shard_idx += 1
        tok_acc.clear(); ids_acc.clear(); probs_acc.clear(); mask_acc.clear()
        c_start.clear(); c_len.clear()

    n_done = 0
    t_start = time.time()

    bar = tqdm(
        total=args.num_chunks,
        desc=f"Scoring ({args.corpus})",
        unit="chunk",
        dynamic_ncols=True,
    )

    try:
        for chunk, valid_len, doc_idx, src_info in chunks:
            if STOP:
                break
            if args.num_chunks is not None and n_done >= args.num_chunks:
                break

            t0_chunk = time.time()

            body = json.dumps({
                "prompt": chunk,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": args.k,
            }).encode("utf-8")
            req = urllib.request.Request(server_url, data=body, headers={"Content-Type": "application/json"})
            resp_data = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    tqdm.write(f"\n[Warning] Request attempt {attempt + 1} failed: {e}. Retrying in 2s...")
                    time.sleep(2)

            dt = time.time() - t0_chunk
            tok_s = (args.seq_len - 1) / dt

            # Locate the dumped .npz chunk by request ID or latest file
            req_id = resp_data.get("id", "").replace("cmpl-", "")
            target_file = raw_dir / f"{req_id}.npz"
            if not target_file.exists():
                target_file = raw_dir / f"cmpl-{req_id}.npz"
            if not target_file.exists():
                candidates = sorted(raw_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not candidates:
                    raise FileNotFoundError(f"No raw chunk .npz found in {raw_dir}")
                target_file = candidates[0]

            with np.load(target_file) as d:
                tokens = d["tokens"].copy()
                teacher_ids = d["teacher_ids"].copy()
                teacher_probs = d["teacher_probs"].copy()
                loss_mask = d["loss_mask"].copy()

            # Clean up raw chunk to save disk space
            for _ in range(5):
                try:
                    target_file.unlink()
                    break
                except PermissionError:
                    time.sleep(0.05)
                except Exception:
                    break

            # Apply exact padding & loss mask contract
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

            elapsed = time.time() - t_start
            rate = (n_done * (args.seq_len - 1)) / elapsed
            daily = (rate * 86400) / 1e6

            bar.update(1)
            bar.set_postfix({
                "tok/s": f"{tok_s:.0f}",
                "avg": f"{rate:.0f}",
                "daily": f"~{daily:.1f}M",
                "valid": f"{valid_len}/{args.seq_len}",
                "shard": shard_idx,
            }, refresh=True)

            if len(c_len) >= args.chunks_per_npz:
                flush_shard()

    finally:
        bar.close()
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
