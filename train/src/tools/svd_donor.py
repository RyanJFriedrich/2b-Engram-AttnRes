"""SVD donor projection tool: extract and compress embed_tokens and lm_head.

Down-projects the donor's untied embeddings and output head from d_model=4096
to d_model=2048 using joint SVD on the concatenated [E; H] in R^(2V x 4096).
The interior remains 100% cold start / fresh random initialization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Union

import torch
from safetensors.torch import load_file

from train.src.config import ModelConfig, load_config
from train.src.model.refit import RefitModel
from train.src.tools.base_ckpt import save_base_checkpoint
from train.utils.log import log


def load_donor_embed_and_head(donor_dir: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Read embed_tokens.weight and lm_head.weight from an HF checkpoint directory."""
    donor_dir = Path(donor_dir)
    config_path = donor_dir / "config.json"
    cfg = {}
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

    embed_weight = None
    head_weight = None

    index_path = donor_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        embed_shard = weight_map.get("model.embed_tokens.weight") or weight_map.get("embed_tokens.weight")
        head_shard = weight_map.get("lm_head.weight")

        if embed_shard and (donor_dir / embed_shard).exists():
            state = load_file(str(donor_dir / embed_shard))
            if "model.embed_tokens.weight" in state:
                embed_weight = state["model.embed_tokens.weight"]
            elif "embed_tokens.weight" in state:
                embed_weight = state["embed_tokens.weight"]

        if head_shard and (donor_dir / head_shard).exists():
            state = load_file(str(donor_dir / head_shard))
            if "lm_head.weight" in state:
                head_weight = state["lm_head.weight"]

    if embed_weight is None or head_weight is None:
        shard_files = sorted(p.name for p in donor_dir.glob("model*.safetensors"))
        for shard in shard_files:
            state = load_file(str(donor_dir / shard))
            if embed_weight is None:
                if "model.embed_tokens.weight" in state:
                    embed_weight = state["model.embed_tokens.weight"]
                elif "embed_tokens.weight" in state:
                    embed_weight = state["embed_tokens.weight"]
            if head_weight is None:
                if "lm_head.weight" in state:
                    head_weight = state["lm_head.weight"]
            if embed_weight is not None and head_weight is not None:
                break

    if embed_weight is None:
        raise KeyError(f"model.embed_tokens.weight not found in {donor_dir}")
    if head_weight is None:
        # Tied embeddings case
        log("lm_head.weight not found; assuming tied word embeddings", print_console=True)
        head_weight = embed_weight.clone()

    return embed_weight, head_weight, cfg


def compute_joint_svd_projection(
    embed_tokens: torch.Tensor,
    lm_head: torch.Tensor,
    target_dim: int = 2048,
    alpha: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute scale-normalized, target-biased joint SVD projection.

    Balances representation quality between input embeddings E and target
    unembedding head H. Because raw ||E||_F^2 is ~49x larger than ||H||_F^2,
    raw concatenation is ~98% dominated by E. Normalizing both by their
    Frobenius norm and weighting by alpha (default 0.7 for H-priority) preserves
    ~63.2% of H's variance (near its 65.1% theoretical ceiling) while retaining
    ~57.6% of E's variance (preventing input collapse).

    Args:
        embed_tokens: [V, d_in]
        lm_head: [V, d_in]
        target_dim: target d_model (e.g. 2048)
        alpha: weight on H (unembeddings), in [0, 1]. Default 0.7.

    Returns:
        (E_proj [V, target_dim], H_proj [V, target_dim], P [target_dim, d_in], var_H)
    """
    v_e, d_in = embed_tokens.shape
    v_h, _ = lm_head.shape
    assert d_in > target_dim, f"donor d_model ({d_in}) must be > target_dim ({target_dim})"
    assert 0.0 <= alpha <= 1.0, f"alpha ({alpha}) must be in [0, 1]"

    # Cast to float32 for high-precision SVD
    E = embed_tokens.to(torch.float32)
    H = lm_head.to(torch.float32)

    norm_E = torch.norm(E)
    norm_H = torch.norm(H)

    E_norm = E / norm_E
    H_norm = H / norm_H

    w_E = 1.0 - alpha
    w_H = alpha

    log(
        f"computing scale-normalized joint Gram matrix G = {w_E:.2f}*(E_norm^T E_norm) + {w_H:.2f}*(H_norm^T H_norm) "
        f"({d_in}x{d_in}) for SVD (alpha={alpha})...",
        print_console=True,
    )
    # Gram matrix G in R^(d_in x d_in)
    G = w_E * torch.matmul(E_norm.transpose(0, 1), E_norm) + w_H * torch.matmul(H_norm.transpose(0, 1), H_norm)

    eigenvalues, eigenvectors = torch.linalg.eigh(G)
    # eigh returns eigenvalues in ascending order; reverse to descending
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    V = eigenvectors[:, sorted_indices]

    # Projection matrix P in R^(target_dim x d_in)
    P = V[:, :target_dim].transpose(0, 1)  # [target_dim, d_in]

    # Verify orthonormality: P @ P^T == I
    diff_eye = (torch.matmul(P, P.transpose(0, 1)) - torch.eye(target_dim)).abs().max().item()
    assert diff_eye < 1e-4, f"orthonormality assertion failed: max diff {diff_eye:.2e}"

    # Down-project embeddings and head
    E_proj = torch.matmul(E, P.transpose(0, 1))  # [V, target_dim]
    H_proj = torch.matmul(H, P.transpose(0, 1))  # [V, target_dim]

    # Measure exact variance retained for both matrices
    E_rec = torch.matmul(E_proj, P)
    H_rec = torch.matmul(H_proj, P)
    var_E = (1.0 - (torch.norm(E - E_rec)**2 / norm_E**2).item())
    var_H = (1.0 - (torch.norm(H - H_rec)**2 / norm_H**2).item())

    log(
        f"normalized joint SVD complete: {d_in} -> {target_dim} (alpha={alpha}) | "
        f"H variance retained: {var_H * 100:.2f}%, E variance retained: {var_E * 100:.2f}%",
        print_console=True,
    )
    return E_proj, H_proj, P, var_H


def build_base_checkpoint_from_svd(
    model_cfg: ModelConfig,
    embed_proj: torch.Tensor,
    head_proj: torch.Tensor,
    out_dir: Union[str, Path],
    note: str = "",
) -> Path:
    """Build and save base checkpoint with projected embeddings and fresh interior."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"constructing cold-start RefitModel with config {model_cfg.hidden_size} hidden, {model_cfg.num_hidden_layers} layers...", print_console=True)
    model = RefitModel(model_cfg)

    # Inject down-projected furniture embeddings
    assert embed_proj.shape == (model_cfg.vocab_size, model_cfg.hidden_size), (
        f"embed_proj shape {embed_proj.shape} mismatch with vocab {model_cfg.vocab_size}, hidden {model_cfg.hidden_size}"
    )
    assert head_proj.shape == (model_cfg.vocab_size, model_cfg.hidden_size), (
        f"head_proj shape {head_proj.shape} mismatch with vocab {model_cfg.vocab_size}, hidden {model_cfg.hidden_size}"
    )

    model.model.embed_tokens.weight.data.copy_(embed_proj.to(torch.bfloat16))
    model.lm_head.weight.data.copy_(head_proj.to(torch.bfloat16))

    note_full = (
        f"Base checkpoint generated via joint SVD down-projection to d_model={model_cfg.hidden_size}.\n"
        f"Embeddings: SVD projected from donor furniture (frozen at step 0).\n"
        f"Interior: 100% cold-start fresh random initialization.\n"
        f"{note}"
    )
    return save_base_checkpoint(model, out_dir, dtype=torch.bfloat16, note=note_full)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and down-project donor embed/head via joint SVD.")
    parser.add_argument("--donor", required=True, help="Path to donor HF checkpoint directory")
    parser.add_argument("--model-config", required=True, help="Path to target model YAML config")
    parser.add_argument("--out", required=True, help="Path to export base checkpoint")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight on H (unembeddings) in normalized joint SVD [0, 1]. Default 0.7.")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    log(f"loading donor embeddings from {args.donor}...", print_console=True)
    E, H, _ = load_donor_embed_and_head(args.donor)

    log(f"donor embed shape: {E.shape}, head shape: {H.shape}", print_console=True)
    E_proj, H_proj, P, var_ratio = compute_joint_svd_projection(
        E, H, target_dim=model_cfg.hidden_size, alpha=args.alpha
    )

    out_path = build_base_checkpoint_from_svd(
        model_cfg,
        E_proj,
        H_proj,
        args.out,
        note=f"Donor: {args.donor}, Normalized SVD (alpha={args.alpha}), H variance explained: {var_ratio * 100:.2f}%",
    )
    log(f"SVD base checkpoint successfully created at: {out_path}", print_console=True)


if __name__ == "__main__":
    main()
