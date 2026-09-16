"""Converter: data_pipeline bulk NPZ -> spec §6.1 shard (fold_version v2).

Ingests the lossless `.npz` shards produced by `data_pipeline/run_bulk_score.py`
(documented in docs/NPZFormat.md) and rewrites them as the flat memmap shard
format this repo's trainer consumes (`train/src/data/topk_writer.py`).

The load-bearing detail — the ONE-ROW SHIFT between the two formats:

- NPZ row t holds the teacher distribution PREDICTING tokens[t] (i.e. the
  distribution at causal position t-1). Slot 0 is the GT token with its true
  teacher prob; slots 1..K-1 are the teacher's top-K excluding GT, shuffled
  as intact (id, prob) pairs; row mass <= 1 with the tail exact as
  1 - row_sum. loss_mask==0 rows (chunk position 0, no distribution exists)
  carry -1/zeros.
- Spec-shard row t holds the distribution AT position t (predicting
  tokens[t+1]) — the trainer's shifted-label convention (see topk_loader).

So spec_row[t] = npz_row[t+1] within each chunk; each chunk's final row is
zero-filled and loss-masked (its target lies outside the chunk). The NPZ's
masked chunk-start rows carry no distribution and are dropped by the shift.
After conversion, topk_idx[t, 0] == tokens[t+1] (gold at slot 0 with true
teacher mass — the anchor's genuine distribution, asserted on every row).

Chunks map to shard documents 1:1 (doc_id boundaries = chunk boundaries), so
the loader's doc-boundary masking aligns with the NPZ's no-cross-chunk
scoring. Probs are NOT renormalized; tail_w is recomputed in fp32 by the
writer at storage time (spec §6.1 fp32-tail rule). The output is unfolded —
fold_version "v2".

Usage:
    python -m train.src.tools.npz_converter data_pipeline/bulk_out/bulk_00000.npz out_shard/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np

from train.src.data.topk_writer import ShardWriter
from train.utils.log import log

FOLD_VERSION = "v2"  # unfolded: tail mass separate, GT at slot 0 with true prob


def _validate_npz(d: np.lib.npyio.NpzFile, path: Path) -> tuple[int, int]:
    """Shape/dtype/coverage checks on the NPZ contract (docs/NPZFormat.md)."""
    required = {
        "tokens": np.dtype(np.uint32),
        "teacher_ids": np.dtype(np.int32),
        "teacher_probs": np.dtype(np.float32),
        "loss_mask": np.dtype(np.uint8),
        "chunk_start": np.dtype(np.int64),
        "chunk_length": np.dtype(np.int64),
    }
    for name, dt in required.items():
        if name not in d:
            raise ValueError(f"{path}: missing array {name!r}")
        if d[name].dtype != dt:
            raise ValueError(f"{path}: {name} dtype {d[name].dtype} != expected {dt}")
    n = d["tokens"].shape[0]
    k = d["teacher_ids"].shape[1] if d["teacher_ids"].ndim == 2 else 0
    if d["teacher_ids"].shape != (n, k) or d["teacher_probs"].shape != (n, k) or k < 1:
        raise ValueError(f"{path}: teacher arrays must be [N, K] with K >= 1")
    if d["loss_mask"].shape != (n,):
        raise ValueError(f"{path}: loss_mask shape {d['loss_mask'].shape} != ({n},)")
    starts, lengths = d["chunk_start"], d["chunk_length"]
    if starts.ndim != 1 or starts.shape != lengths.shape:
        raise ValueError(f"{path}: chunk_start/chunk_length must be equal-length 1-D")
    # Chunks must tile [0, N) exactly.
    pos = 0
    for c in range(len(starts)):
        if starts[c] != pos or lengths[c] < 1:
            raise ValueError(
                f"{path}: chunks do not tile the stream (chunk {c}: start "
                f"{starts[c]} != {pos} or length {lengths[c]} < 1)"
            )
        pos += int(lengths[c])
    if pos != n:
        raise ValueError(f"{path}: chunks cover {pos} tokens but stream has {n}")
    return n, k


def convert_npz(
    npz_path: Union[str, Path],
    shard_dir: Union[str, Path],
    teacher_id: str = "olmo-3-7b-instruct-abliterated",
    quantization: str = "Q8_0 GGUF, fork vllm raw dump",
    text_source: str = "wikipedia + wikidict",
    data_class: str = "a",
    alpha_override: Optional[float] = None,
    vocab_size: int = 100278,
    max_chunks: Optional[int] = None,
    log_filename: str = "common.log",
) -> dict:
    """Convert one bulk NPZ to a spec shard directory. Returns the sidecar."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as d:
        n, k = _validate_npz(d, npz_path)
        tokens = np.asarray(d["tokens"])
        teacher_ids = np.asarray(d["teacher_ids"])
        teacher_probs = np.asarray(d["teacher_probs"])
        loss_mask = np.asarray(d["loss_mask"])
        starts = np.asarray(d["chunk_start"])
        lengths = np.asarray(d["chunk_length"])

        # NPZ contract: unmasked rows carry the GT token at slot 0.
        unmasked = loss_mask == 1
        if not (teacher_ids[unmasked, 0] == tokens[unmasked].astype(np.int32)).all():
            raise ValueError(
                f"{npz_path}: GT-slot contract violated (teacher_ids[t,0] != tokens[t] "
                "on an unmasked row) — is this a docs/NPZFormat.md shard?"
            )

        writer = ShardWriter(
            shard_dir, k=k, teacher_id=teacher_id, quantization=quantization,
            fold_version=FOLD_VERSION, vocab_size=vocab_size,
            log_filename=log_filename, text_source=text_source,
            logit_source=f"{teacher_id} ({quantization})",
            data_class=data_class, alpha_override=alpha_override,
        )
        for c, (s, length) in enumerate(zip(starts, lengths)):
            if max_chunks is not None and c >= max_chunks:
                break
            s, L = int(s), int(length)
            doc_tokens = tokens[s:s + L]
            # spec_row[t] = npz_row[s + t + 1]; the chunk's final row has no
            # in-chunk target -> zeros + masked.
            src = np.arange(s + 1, s + L)  # source rows for spec rows 0..L-2
            ids = np.zeros((L, k), dtype=np.int64)
            probs = np.zeros((L, k), dtype=np.float32)
            mask = np.zeros(L, dtype=np.uint8)
            if L > 1:
                src_mask = loss_mask[src] == 1
                ids[:-1][src_mask] = teacher_ids[src][src_mask]
                p_chunk = teacher_probs[src][src_mask]
                # Guard against bf16 logsumexp rounding overshoot (> 1.0)
                row_sums = p_chunk.sum(axis=1, keepdims=True)
                over = (row_sums > 1.0).squeeze(-1)
                if over.any():
                    p_chunk[over] = p_chunk[over] / row_sums[over]
                probs[:-1][src_mask] = p_chunk
                mask[:-1] = src_mask.astype(np.uint8)
            writer.add_document(doc_tokens, ids, probs, loss_mask=mask)

    sidecar = writer.finalize()
    log(
        f"npz_converter: {npz_path} -> {shard_dir}: {sidecar['num_documents']} chunks "
        f"as docs, {sidecar['total_tokens']} tokens, k={k}, fold_version={FOLD_VERSION}",
        print_console=True,
    )
    return sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", nargs="?", default=None, help="Input .npz file (or omit if using --all)")
    p.add_argument("shard_dir", nargs="?", default=None, help="Output shard directory")
    p.add_argument("--all", action="store_true",
                   help="Batch convert all bulk_*.npz in bulk-dir to shards-dir")
    p.add_argument("--bulk-dir", default="data_pipeline/bulk_out")
    p.add_argument("--shards-dir", default="data_pipeline/shards")
    p.add_argument("--teacher-id", default="allenai/Olmo-3.1-32B-Think")
    p.add_argument("--quantization",
                   default="Q8_0 GGUF, fork vLLM prompt_logprobs prefill")
    p.add_argument("--text-source",
                   default="wikipedia-20231101.en")
    p.add_argument("--data-class", default="a")
    p.add_argument("--alpha-override", type=float, default=None)
    p.add_argument("--max-chunks", type=int, default=None)
    args = p.parse_args()

    if args.all:
        bulk_dir = Path(args.bulk_dir)
        shards_dir = Path(args.shards_dir)
        shards_dir.mkdir(parents=True, exist_ok=True)
        npz_files = sorted(bulk_dir.glob("bulk_*.npz"))
        if not npz_files:
            log(f"No bulk_*.npz files found under {bulk_dir}", print_console=True)
            return
        converted_count = 0
        for npz_file in npz_files:
            shard_name = npz_file.stem.replace("bulk_", "shard_")
            out_target = shards_dir / shard_name
            if (out_target / "sidecar.json").exists():
                continue
            log(f"Batch converting: {npz_file.name} -> {out_target.name}...", print_console=True)
            convert_npz(
                npz_file, out_target, teacher_id=args.teacher_id,
                quantization=args.quantization, text_source=args.text_source,
                data_class=args.data_class, alpha_override=args.alpha_override,
                max_chunks=args.max_chunks,
            )
            converted_count += 1
        log(f"Batch conversion complete: {converted_count} new shards converted.", print_console=True)
        return

    if not args.npz or not args.shard_dir:
        p.error("the following arguments are required: npz, shard_dir (or pass --all)")

    convert_npz(
        args.npz, args.shard_dir, teacher_id=args.teacher_id,
        quantization=args.quantization, text_source=args.text_source,
        data_class=args.data_class, alpha_override=args.alpha_override,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
