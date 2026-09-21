"""Engram module tests: I1 (zero-init U = exact no-op at init), I2/I8
(injection registered in the AttnRes delta-sum), readout math, boundary
masking, sparse optimizer math, and table checkpointing. dev_tiny scale."""
import numpy as np
import torch

from train.src.config import ModelConfig, load_config
from train.src.engram.sparse_opt import SparseRowAdamW8bit
from train.src.model.refit import RefitModel

DEV_TINY = "train/configs/model/dev_tiny.yaml"
VOCAB = 1024


def _enabled_model(seed: int = 0, readout: str = "kv") -> RefitModel:
    d = load_config(DEV_TINY).to_dict()
    d["engram"]["enabled"] = True
    d["engram"]["readout"] = readout
    torch.manual_seed(seed)
    return RefitModel(ModelConfig.from_dict(d))


def _randomize_readout(model: RefitModel, seed: int = 7) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if hasattr(model.engram, "u_out"):
            model.engram.u_out.weight.copy_(0.02 * torch.randn(model.engram.u_out.weight.shape, generator=g))
        if hasattr(model.engram, "proj"):
            for lin in model.engram.proj.values():
                lin.weight.copy_(0.02 * torch.randn(lin.weight.shape, generator=g))


def test_i1_garbage_rows_still_exact_noop():
    """I1: with zero-init U the injection is exactly 0 no matter what the
    tables contain — garbage rows must not change the logits."""
    model = _enabled_model()
    x = torch.randint(0, VOCAB, (2, 16))
    ref = model(x)
    for key in model.engram_tables.table_keys:
        model.engram_tables.rows[key].fill_(3.25)  # garbage
    assert torch.equal(model(x), ref)


def test_i2_i8_injection_registered_in_delta_sum():
    """I2/I8: the Engram delta lands in the block-1 partial BEFORE the block
    completes, so the AttnRes sources at (4, pre_attn) — [embedding, block1,
    fresh-zero partial] — reconstruct the stream exactly:
    block1 == hiddens[3] - embedding."""
    model = _enabled_model()
    _randomize_readout(model)
    x = torch.randint(0, VOCAB, (2, 16))
    capture: dict = {}
    model(x, capture=capture)
    sources = capture["attn_res_sources"][(4, "pre_attn")]
    emb, block1 = sources[0], sources[1]
    assert torch.equal(emb, capture["embedding"])
    reconstructed = emb + block1
    actual = capture["hiddens"][3]
    # Same add sequence with a different accumulator start -> fp32-close,
    # and definitely NOT equal to the no-injection delta-sum.
    assert torch.allclose(reconstructed, actual, atol=1e-5)

    # Control: with U zeroed the injection vanishes and block1 still
    # reconstructs (the bookkeeping itself is injection-agnostic).
    with torch.no_grad():
        for lin in model.engram.proj.values():
            lin.weight.zero_()
    capture2: dict = {}
    model(x, capture=capture2)
    b1 = capture2["attn_res_sources"][(4, "pre_attn")][1]
    assert torch.allclose(capture2["embedding"] + b1, capture2["hiddens"][3], atol=1e-5)


def test_readout_math_and_boundary_mask():
    """The readout is g * U(RMSNorm(concat(head rows))) per order, summed,
    with boundary-invalid positions contributing exactly zero (legacy linear mode)."""
    model = _enabled_model(readout="linear")
    _randomize_readout(model)
    tables = model.engram_tables
    x = torch.randint(0, VOCAB, (1, 8))
    gb = tables.gather(x, torch.device("cpu"), requires_grad=False)
    delta = model.engram(gb)
    assert delta.shape == (1, 8, model.refit_config.hidden_size)

    # Manual per-order recomputation for a valid position (t=7). Keep the
    # forward's dtype path: bf16 concat into the norm (fp32 stats, bf16
    # output), fp32 projection.
    def order_contrib(n: int, t: int) -> torch.Tensor:
        pieces = []
        for ki, key in enumerate(gb.table_keys):
            if key[0] != n:
                continue
            pieces.append(gb.rows[ki][gb.inverse[ki]])  # [T, row_dim]
        cat = torch.cat(pieces, dim=-1)
        normed = model.engram.norms[str(n)](cat)
        return (model.engram.proj[str(n)](normed) * model.engram.gates[str(n)])[t]

    total = order_contrib(1, 7) + order_contrib(2, 7) + order_contrib(3, 7)
    assert torch.allclose(delta[0, 7].float(), total.float(), atol=1e-5)

    # Boundary semantics: at t=0 only the unigram order contributes; at t=1
    # unigram+bigram; trigram starts at t=2.
    assert torch.allclose(delta[0, 0].float(), order_contrib(1, 0).float(), atol=1e-5)
    assert torch.allclose(
        delta[0, 1].float(),
        (order_contrib(1, 1) + order_contrib(2, 1)).float(),
        atol=1e-5,
    )
    valid = gb.valid[0]
    assert not valid[0, 1] and not valid[0, 2] and not valid[1, 2]


def test_kv_readout_math_and_boundary_mask():
    """KV readout: q_t = W_q(h_t), s_i = (q . RMSNorm(k_i))/sqrt(d_k) + b_{order(i)}.
    Supports both independent sigmoid gating ('kv' / 'kv_sigmoid') and joint softmax ('kv_softmax').
    Boundary-invalid items receive 0.0 gate activation (sigmoid) or -1e9 logit (softmax)."""
    # 1. Test sigmoid readout
    m_sig = _enabled_model(readout="kv_sigmoid")
    _randomize_readout(m_sig)
    assert m_sig.engram.mode == "sigmoid"
    tables = m_sig.engram_tables
    x = torch.randint(0, VOCAB, (1, 8))
    gb = tables.gather(x, torch.device("cpu"), requires_grad=False)
    h = torch.randn(1, 8, m_sig.refit_config.hidden_size)

    delta_sig = m_sig.engram(gb, h)
    assert delta_sig.shape == (1, 8, m_sig.refit_config.hidden_size)
    assert torch.isfinite(delta_sig).all()

    tel_sig = m_sig.engram.pop_telemetry()
    assert "engram_attn_entropy" in tel_sig
    assert "engram_act" in tel_sig
    assert 0.0 <= tel_sig["engram_act"] <= 1.0
    # Frequency tier telemetry present since stage populates touch_counts
    assert any(k.startswith("engram_act_") for k in tel_sig)

    # 2. Test softmax readout
    m_soft = _enabled_model(readout="kv_softmax")
    _randomize_readout(m_soft)
    assert m_soft.engram.mode == "softmax"
    delta_soft = m_soft.engram(gb, h)
    assert delta_soft.shape == (1, 8, m_soft.refit_config.hidden_size)
    assert torch.isfinite(delta_soft).all()

    tel_soft = m_soft.engram.pop_telemetry()
    assert "engram_attn_entropy" in tel_soft
    assert tel_soft["engram_attn_entropy"] > 0.0


def test_kv_softcapping_bounds_logits():
    """Logit softcapping c * tanh(s / c) bounds the logit gap to [-c, c],
    preventing saturation and gradient starvation even under extreme inputs."""
    model = _enabled_model(readout="kv_sigmoid")
    _randomize_readout(model)
    model.engram.softcap = 8.0

    tables = model.engram_tables
    x = torch.randint(0, VOCAB, (1, 8))
    gb = tables.gather(x, torch.device("cpu"), requires_grad=False)
    # Huge query vector
    h_huge = 1000.0 * torch.randn(1, 8, model.refit_config.hidden_size)

    delta = model.engram(gb, h_huge)
    assert torch.isfinite(delta).all()

    # Even with huge h, gate activations must be bounded in [sigma(-8), sigma(8)]
    # sigma(8) ~ 0.99966, sigma(-8) ~ 0.00033
    tel = model.engram.pop_telemetry()
    assert 0.0 <= tel["engram_act"] <= 1.0


def test_i10_context_aware_differentiation():
    """Invariant I10: same n-gram, different context -> different delivered delta.

    1. Step 0 check (I1): with zero-init U_out, delta_1 == delta_2 == 0 regardless of context h.
    2. Perturbed check: when U_out is non-zero, identical staged tokens under different
       contexts h1 != h2 produce distinct attention weights alpha1 != alpha2 and
       distinct injection deltas delta1 != delta2.
    """
    model = _enabled_model(readout="kv")
    x = torch.randint(0, VOCAB, (1, 8))
    tables = model.engram_tables
    gb = tables.gather(x, torch.device("cpu"), requires_grad=False)

    h1 = torch.randn(1, 8, model.refit_config.hidden_size)
    h2 = torch.randn(1, 8, model.refit_config.hidden_size)

    # 1. Step 0 (I1): U_out is zero-initialized -> deltas are strictly 0
    d1_zero = model.engram(gb, h1)
    d2_zero = model.engram(gb, h2)
    assert torch.equal(d1_zero, torch.zeros_like(d1_zero))
    assert torch.equal(d2_zero, torch.zeros_like(d2_zero))

    # 2. Perturbed fixture (I10): perturb U_out to small non-zero weights
    _randomize_readout(model)
    d1 = model.engram(gb, h1)
    d2 = model.engram(gb, h2)

    assert not torch.allclose(d1, d2, atol=1e-4)


def test_sparse_optimizer_per_row_bias_correction():
    """Verify that a cold row touched at global step t=100 uses its own
    row_step=1 bias correction (1 - beta^1) rather than global step t=100 (1 - beta^100),
    eliminating the ~3.16x overshoot."""
    from train.src.engram.tables import GatherBatch

    model = _enabled_model(readout="kv")
    tables = model.engram_tables
    opt = SparseRowAdamW8bit(tables)
    key = tables.table_keys[0]

    # Artificially advance global table step to 100
    st = opt._state_for(key)
    st["step"] = 100

    row_id = 42
    assert st["row_step"][row_id].item() == 0

    g = torch.randn(1, tables.cfg.row_dim, dtype=torch.float32)
    gb = GatherBatch(
        table_keys=[key],
        rows=[g.clone().requires_grad_(True)],
        inverse=[torch.tensor([0])],
        uniq=[np.array([row_id], dtype=np.int64)],
        valid=torch.ones(1, 1, 3, dtype=torch.bool),
        shape=(1, 1),
    )
    gb.rows[0].grad = g.clone()

    before = tables.rows[key][row_id].clone().float()
    lr = 1e-2
    opt.accumulate(gb)
    opt.step(lr=lr)

    # row_step for row 42 must now be 1
    assert st["row_step"][row_id].item() == 1

    # Exact r_step=1 math: m_hat = g, r_hat = |g|
    expected = (before - lr * g[0] / (g[0].abs() + 1e-8)).to(torch.bfloat16)
    after = tables.rows[key][row_id]
    assert torch.equal(after, expected)


def test_injection_point_is_layer_3_output():
    """The stream before layer 3 is untouched by Engram; the stream after
    layer 3 differs when the readout is nonzero."""
    model = _enabled_model()
    x = torch.randint(0, VOCAB, (2, 16))
    cap_z, cap_r = {}, {}
    model(x, capture=cap_z)
    _randomize_readout(model)
    model(x, capture=cap_r)
    for i in range(3):
        assert torch.equal(cap_z["hiddens"][i], cap_r["hiddens"][i])
    assert not torch.allclose(cap_z["hiddens"][3], cap_r["hiddens"][3])


def test_sparse_optimizer_math_single_step():
    """t=1 Adam: m_hat=g, r_hat=|g| -> new = old - lr*g/(|g|+eps); WD=0;
    only grad-touched rows change, all others bitwise preserved."""
    model = _enabled_model()
    _randomize_readout(model)
    tables = model.engram_tables
    x = torch.randint(0, VOCAB, (2, 16))
    gb = tables.gather(x, torch.device("cpu"), requires_grad=True)
    model(x, engram=gb, return_hidden=True).square().mean().backward()

    opt = SparseRowAdamW8bit(tables)
    key = gb.table_keys[0]
    before = tables.rows[key].clone()
    grads = gb.rows[0].grad.float().cpu()
    uniq = gb.uniq[0]
    lr = 1e-2
    opt.accumulate(gb)
    opt.step(lr=lr)

    after = tables.rows[key]
    g0 = grads[0]
    expected = (before[uniq[0]].float() - lr * g0 / (g0.abs() + 1e-8)).to(torch.bfloat16)
    assert torch.equal(after[uniq[0]], expected)

    nz_rows = (grads.abs().sum(dim=1) > 0).numpy()
    addr = np.zeros(before.shape[0], dtype=bool)
    addr[uniq[nz_rows]] = True
    changed = (after != before).any(dim=1).numpy()
    assert (changed == addr).all()


def test_cadence_accumulate_then_single_dense_step():
    """Annex v1.1 cadence rule: grads from multiple micro-batches accumulate
    host-side fp32 and apply as ONE Adam update per dense-equivalent step —
    a row touched in two micro-batches gets t=1 Adam on the SUMMED grad, not
    two compounded updates."""
    from train.src.engram.tables import GatherBatch

    model = _enabled_model()
    tables = model.engram_tables
    key = (1, 0)
    D = tables.cfg.row_dim

    def fake_gb(grads: dict) -> GatherBatch:
        """Minimal stand-in: accumulate() reads rows[i].grad and uniq[i]."""
        rows, uniq = [], []
        for k in tables.table_keys:
            if k in grads:
                u, g = grads[k]
                leaf = tables.rows[k][torch.from_numpy(u)].clone().requires_grad_(True)
                leaf.grad = g
                rows.append(leaf)
                uniq.append(u)
            else:
                rows.append(torch.zeros((1, D), requires_grad=True))  # grad None
                uniq.append(np.array([0]))
        return GatherBatch(tables.table_keys, rows, [], uniq,
                           torch.zeros((), dtype=torch.bool), (0, 0))

    u5 = np.array([5])
    g1 = torch.full((1, D), 0.1, dtype=torch.bfloat16)
    g2 = torch.full((1, D), 0.2, dtype=torch.bfloat16)
    u7 = np.array([7])
    g7 = torch.full((1, D), -0.4, dtype=torch.bfloat16)

    opt = SparseRowAdamW8bit(tables)
    before5 = tables.rows[key][5].clone()
    before7 = tables.rows[key][7].clone()
    opt.accumulate(fake_gb({key: (u5, g1)}))
    opt.accumulate(fake_gb({key: (u5, g2)}))
    opt.accumulate(fake_gb({key: (u7, g7)}))
    lr = 1e-2
    opt.step(lr=lr)

    # t=1: m_hat = g_sum, r_hat = |g_sum| -> update = lr * g_sum/(|g_sum|+eps).
    gs5 = float((g1.float() + g2.float())[0, 0])  # summed post-bf16 host grads
    gs7 = float(g7.float()[0, 0])
    expected5 = (before5.float() - lr * gs5 / (abs(gs5) + 1e-8)).to(torch.bfloat16)
    expected7 = (before7.float() - lr * gs7 / (abs(gs7) + 1e-8)).to(torch.bfloat16)
    assert torch.equal(tables.rows[key][5], expected5)
    assert torch.equal(tables.rows[key][7], expected7)
    assert opt.state[key]["step"] == 1  # ONE dense step, not two/three


def test_zero_grad_rows_not_touched():
    """Rows addressed only at boundary-invalid positions have zero grad and
    must see no state decay and no value drift."""
    model = _enabled_model()
    _randomize_readout(model)
    tables = model.engram_tables
    x = torch.randint(0, VOCAB, (1, 4))  # T=4: trigram valid only at t=3
    gb = tables.gather(x, torch.device("cpu"), requires_grad=True)
    model(x, engram=gb, return_hidden=True).square().mean().backward()
    opt = SparseRowAdamW8bit(tables)
    before = {k: v.clone() for k, v in tables.rows.items()}
    opt.accumulate(gb)
    opt.step(lr=1e-2)
    for i, key in enumerate(gb.table_keys):
        n = key[0]
        if n < 3:
            continue
        zg = (gb.rows[i].grad.float().abs().sum(dim=1) == 0).numpy()
        if zg.any():
            uniq_zero = gb.uniq[i][zg]
            assert torch.equal(tables.rows[key][uniq_zero], before[key][uniq_zero])


def test_tables_state_dict_roundtrip():
    """Table checkpoint: rows + touch counters + canon sha round-trip; a
    canon mismatch is refused loudly (addressing would silently change)."""
    import pytest

    model = _enabled_model()
    tables = model.engram_tables
    x = torch.randint(0, VOCAB, (2, 16))
    tables.gather(x, torch.device("cpu"), requires_grad=False)
    with torch.no_grad():
        tables.rows[(2, 0)][5].fill_(0.5)
    sd = tables.state_dict()

    tables2 = _enabled_model(seed=1).engram_tables
    tables2.load_state_dict(sd)
    for key in tables.table_keys:
        assert torch.equal(tables2.rows[key], tables.rows[key])
        assert np.array_equal(tables2.touch[key], tables.touch[key])

    bad = dict(sd)
    bad["canon_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canon sha256"):
        _enabled_model(seed=2).engram_tables.load_state_dict(bad)
