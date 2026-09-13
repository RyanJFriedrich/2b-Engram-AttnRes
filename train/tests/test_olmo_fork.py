"""Unit and invariant tests for the OLMo-2B architectural fork (Spec v3.0).

Tests:
1. Joint SVD down-projection mathematics (orthonormality, variance explained).
2. Dynamic SWA windows [4096, 4096, 8192] and thetas [10k, 10k, 25k].
3. Split-frequency Globals (Blocks 1-5: 1M; Blocks 6-10 + Gather: 5M).
4. globals_only AttnRes scope (exactly 11 connection points across 41 layers).
5. Cold-start Gather layer initialization (not identity-zeroed).
"""
import pytest
import torch
import yaml

from train.src.config import ModelConfig, load_config
from train.src.model.refit import RefitModel
from train.src.tools.svd_donor import compute_joint_svd_projection

OLMO_2B_CONFIG = "train/configs/model/olmo_2b_v1.yaml"


def test_joint_svd_math():
    """Verify joint SVD on [E; H] extracts orthonormal P and projects accurately."""
    torch.manual_seed(42)
    v, d_in, d_out = 500, 128, 64
    E = torch.randn(v, d_in)
    H = torch.randn(v, d_in)

    E_proj, H_proj, P, var_ratio = compute_joint_svd_projection(E, H, target_dim=d_out)

    assert E_proj.shape == (v, d_out)
    assert H_proj.shape == (v, d_out)
    assert P.shape == (d_out, d_in)
    assert 0.0 < var_ratio <= 1.0

    # Orthonormality: P @ P^T == I
    eye = torch.eye(d_out)
    diff = (torch.matmul(P, P.transpose(0, 1)) - eye).abs().max().item()
    assert diff < 1e-4, f"P was not orthonormal: max diff {diff:.2e}"


def test_olmo_2b_config_validation():
    """Verify olmo_2b_v1.yaml config loads, validates, and roundtrips."""
    cfg = load_config(OLMO_2B_CONFIG)
    assert cfg.vocab_size == 100278
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 41
    assert cfg.attn_res.scope == "globals_only"
    assert cfg.gather.position == 40
    assert cfg.gather.init == "random"

    # Round-trip check
    cfg_d = cfg.to_dict()
    reloaded = ModelConfig.from_dict(cfg_d)
    assert reloaded.to_dict() == cfg_d


def test_olmo_2b_layer_dispatch_and_thetas():
    """Verify layer dispatch produces exact windows, thetas, and AttnRes points."""
    cfg = load_config(OLMO_2B_CONFIG)
    # Use canonical_compression=False for quick test without canon npy
    cfg.engram.canonical_compression = False
    # Use smaller dummy tables to keep test instantaneous in memory
    cfg.engram.rows_per_head = {2: [1009, 1013], 3: [1019, 1021]}
    cfg.vocab_size = 1024

    model = RefitModel(cfg)

    assert len(model.model.layers) == 41
    # AttnRes globals_only: 10 block globals + 1 gather = 11 points
    assert len(model.model.attn_res) == 11
    assert len(model.attn_res_map) == 11
    for i in range(41):
        if cfg.layer_types[i] in ("global", "gather"):
            assert (i, "pre_attn") in model.attn_res_map
            assert (i, "pre_mlp") not in model.attn_res_map
        else:
            assert (i, "pre_attn") not in model.attn_res_map
            assert (i, "pre_mlp") not in model.attn_res_map

    # Check SWA windows and thetas across all blocks
    for block in range(10):
        base_idx = block * 4
        # SWA 1
        l0 = model.model.layers[base_idx].self_attn
        assert l0.layer_type == "swa"
        assert l0.window == 4096
        # SWA 2
        l1 = model.model.layers[base_idx + 1].self_attn
        assert l1.layer_type == "swa"
        assert l1.window == 4096
        # SWA 3
        l2 = model.model.layers[base_idx + 2].self_attn
        assert l2.layer_type == "swa"
        assert l2.window == 8192

        # SWA rotary theta check
        for l, expected_th in ((l0, 10000), (l1, 10000), (l2, 25000)):
            inv = l.rotary.inv_freq
            dim = len(inv) * 2
            th = round(float((1.0 / inv[-1]) ** (dim / (dim - 2))))
            assert th == expected_th

        # Global layer theta check
        lg = model.model.layers[base_idx + 3].self_attn
        assert lg.layer_type == "global"
        expected_global_th = 1000000 if block < 5 else 5000000
        inv_g = lg.rotary.inv_freq
        dim_g = len(inv_g) * 2
        th_g = round(float((1.0 / inv_g[-1]) ** (dim_g / (dim_g - 2))))
        assert th_g == expected_global_th

    # Gather layer (Layer 40)
    gather_attn = model.model.layers[40].self_attn
    assert gather_attn.layer_type == "gather"
    inv_gather = gather_attn.rotary.inv_freq
    dim_gather = len(inv_gather) * 2
    th_gather = round(float((1.0 / inv_gather[-1]) ** (dim_gather / (dim_gather - 2))))
    assert th_gather == 5000000

    # Verify gather layer is NOT zeroed out (cold-start / random init)
    assert not torch.all(model.model.layers[40].self_attn.o_proj.weight == 0)
    assert not torch.all(model.model.layers[40].mlp.down_proj.weight == 0)


def test_olmo_2b_forward_pass():
    """Verify forward pass computes finite outputs at d_model=2048."""
    cfg = load_config(OLMO_2B_CONFIG)
    cfg.engram.canonical_compression = False
    cfg.engram.rows_per_head = {2: [1009, 1013], 3: [1019, 1021]}
    cfg.vocab_size = 512

    model = RefitModel(cfg).eval()
    input_ids = torch.randint(0, 512, (2, 32))

    with torch.no_grad():
        logits = model(input_ids)
        hiddens = model(input_ids, return_hidden=True)

    assert logits.shape == (2, 32, 512)
    assert hiddens.shape == (2, 32, 2048)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(hiddens).all()
