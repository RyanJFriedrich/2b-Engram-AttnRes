"""Engram readout (spec §3.6, annex A1.5, proposal v3.1) — device-side projection.

Supports two readout modes:
1. "kv" (default): Context-aware cross-attention readout.
   - Rows are split into key and value halves: [k; v] where k, v in R^{row_dim/2}.
   - Query q_t = W_q(h_t) formed from the hidden state at Layer 3 output.
   - Attention scores over the candidate set (2 heads x orders):
     s_i = (q_t . RMSNorm(k_i)) / sqrt(d_k) + b_{order(i)}
   - Boundary-invalid items masked with -1e9 before softmax.
   - Attention weights alpha = softmax(s).
   - Values gated per-order: v_{gated, i} = RMSNorm(v_i) * g_{order(i)}.
   - Delta delta_t = U_out( sum_i alpha_i * v_{gated, i} ).
   - U_out is ZERO-INIT (Invariant I1: exact no-op at step 0).
   - Tracks attention entropy H(alpha) for premature collapse diagnostics.

2. "linear" (legacy/ablation): Static n-gram linear projection:
   delta_n = g_n * U_n( RMSNorm( concat_k rows[n,k] ) ).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn

from train.src.config import EngramConfig
from train.src.engram.tables import GatherBatch
from train.src.model.decoder import LlamaRMSNorm


class EngramLinearReadout(nn.Module):
    """Legacy static n-gram readout (spec §3.6)."""

    def __init__(self, cfg: EngramConfig, d_model: int, rms_eps: float) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        in_dim = cfg.heads_per_order * cfg.row_dim
        self.orders = list(cfg.orders)
        self.norms = nn.ModuleDict(
            {str(n): LlamaRMSNorm(in_dim, rms_eps) for n in self.orders}
        )
        self.proj = nn.ModuleDict(
            {str(n): nn.Linear(in_dim, d_model, bias=False) for n in self.orders}
        )
        self.gates = nn.ParameterDict(
            {str(n): nn.Parameter(torch.ones(())) for n in self.orders}
        )
        for n in self.orders:
            nn.init.zeros_(self.proj[str(n)].weight)  # I1: zero-init U

    @torch.compiler.disable
    def forward(self, gb: GatherBatch, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """GatherBatch -> injection delta [B, T, d_model]. Accepts optional h for interface parity."""
        B, T = gb.shape
        out: torch.Tensor | None = None
        for oi, n in enumerate(self.orders):
            pieces = []
            for ki, key in enumerate(gb.table_keys):
                if key[0] != n:
                    continue
                pieces.append(gb.rows[ki][gb.inverse[ki]])
            x = torch.cat(pieces, dim=-1)  # [B*T, heads*row_dim]
            x = self.norms[str(n)](x)
            x = self.proj[str(n)](x) * self.gates[str(n)]
            x = x.view(B, T, self.d_model)
            x = x * gb.valid[:, :, oi].unsqueeze(-1).to(x.dtype)
            out = x if out is None else out + x
        assert out is not None
        return out

    def pop_telemetry(self) -> dict[str, float]:
        return {}


class _ProjView:
    """Non-module view of u_out for backward compatibility with tests/tools
    expecting model.engram.proj. Kept outside nn.Module inheritance so PyTorch
    does not register it as a submodule, avoiding duplicate parameter
    registration in state_dict which breaks safetensors save_file.
    """
    def __init__(self, u_out: nn.Linear):
        self._u_out = u_out

    def parameters(self):
        return self._u_out.parameters()

    def values(self):
        return [self._u_out]

    def items(self):
        return [("u_out", self._u_out)]

    def __getitem__(self, key: str) -> nn.Linear:
        return self._u_out


class EngramKVReadout(nn.Module):
    """Context-aware Key-Value readout for Engram sidecar table.

    Option A (128 | 128 split):
    - Keys [B, T, N_items, key_dim] and Values [B, T, N_items, val_dim]
    - Query q_t = W_q h_t from Layer 3 output hidden state
    - Gating mode:
        * "sigmoid" (readout: "kv" / "kv_sigmoid"): Independent sigmoid gates per candidate.
          No winner-take-all competition; preserves hierarchical additive n-gram backoff.
        * "softmax" (readout: "kv_softmax"): Joint softmax routing across 6 candidates.
    - Gemma-2 style logit softcapping: s <- c * tanh(s / c) with c=8.0 (bounds logit gap).
    - Temperature scaling: s <- s / tau.
    - Boundary masking:
        * Sigmoid: padding masked to gate 0.0 (exact zero injection).
        * Softmax: padding masked to logit -1e9 before softmax.
    - Retained per-order scalar gates g_n on values.
    - Zero-initialized output projection U_out (Invariant I1: exact no-op at step 0).
    - Frequency-bucketed telemetry: tracks gate activations and entropy for low (<=5),
      mid (5-50), and high (>50) touch tiers using host touch counts.
    """

    def __init__(self, cfg: EngramConfig, d_model: int, rms_eps: float = 1e-5) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.key_dim = cfg.key_dim or (cfg.row_dim // 2)
        self.val_dim = cfg.val_dim or (cfg.row_dim - self.key_dim)
        self.orders = list(cfg.orders)
        self.heads_per_order = cfg.heads_per_order

        # Gating mode: independent sigmoid (default for "kv" / "kv_sigmoid") or softmax
        if cfg.readout in ("kv", "kv_sigmoid"):
            self.mode = "sigmoid"
        elif cfg.readout == "kv_softmax":
            self.mode = "softmax"
        else:
            self.mode = "sigmoid"

        self.softcap = cfg.softcap
        self.temperature = cfg.temperature

        # Query projection from hidden state at Layer 3 output: R^{d_model} -> R^{key_dim}
        self.w_q = nn.Linear(d_model, self.key_dim, bias=False)
        nn.init.normal_(self.w_q.weight, std=0.02)

        # Independent key and value RMSNorms per (order, head)
        self.k_norms = nn.ModuleDict({
            f"{n}_{k}": LlamaRMSNorm(self.key_dim, rms_eps)
            for n in self.orders for k in range(self.heads_per_order)
        })
        self.v_norms = nn.ModuleDict({
            f"{n}_{k}": LlamaRMSNorm(self.val_dim, rms_eps)
            for n in self.orders for k in range(self.heads_per_order)
        })

        # Learned scalar order biases in attention logits
        self.order_biases = nn.ParameterDict({
            str(n): nn.Parameter(torch.zeros(())) for n in self.orders
        })

        # Per-order scalar gates on values (retained magnitude control, WD=0)
        self.gates = nn.ParameterDict({
            str(n): nn.Parameter(torch.ones(())) for n in self.orders
        })

        # Output projection back to d_model: ZERO-INITIALIZED (Invariant I1)
        self.u_out = nn.Linear(self.val_dim, d_model, bias=False)
        nn.init.zeros_(self.u_out.weight)

        # Non-module view of u_out for tests/tools expecting model.engram.proj
        self.proj = _ProjView(self.u_out)

        # Telemetry accumulators
        self._reset_telemetry()

    def _reset_telemetry(self) -> None:
        self._tel_entropy: float = 0.0
        self._tel_act: float = 0.0
        self._tel_count: int = 0
        self._tel_buckets: dict[str, list[float]] = {
            "low": [],
            "mid": [],
            "high": [],
        }

    @torch.compiler.disable
    def forward(self, gb: GatherBatch, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Context-conditioned cross-attention injection delta [B, T, d_model]."""
        B, T = gb.shape
        if h is None:
            raise ValueError("EngramKVReadout requires context hidden states h from Layer 3 output")

        # 1. Project query: [B, T, key_dim]
        q = self.w_q(h)
        scale = 1.0 / math.sqrt(self.key_dim)

        # 2. Gather, split, and normalize candidate keys and values
        keys_list = []
        vals_list = []
        order_list = []

        for ki, (n, k) in enumerate(gb.table_keys):
            staged = gb.rows[ki][gb.inverse[ki]]  # [B*T, row_dim]
            k_raw = staged[:, :self.key_dim]
            v_raw = staged[:, self.key_dim:]

            k_norm = self.k_norms[f"{n}_{k}"](k_raw).view(B, T, self.key_dim)
            v_norm = self.v_norms[f"{n}_{k}"](v_raw).view(B, T, self.val_dim)

            keys_list.append(k_norm)
            # Retain per-order scalar gate on candidate value vectors
            vals_list.append(v_norm * self.gates[str(n)])
            order_list.append(n)

        # Stack candidate items: [B, T, N_items, dim]
        keys = torch.stack(keys_list, dim=2)  # [B, T, 6, key_dim]
        vals = torch.stack(vals_list, dim=2)  # [B, T, 6, val_dim]

        # 3. Dot product + learned order biases
        scores = (q.unsqueeze(2) * keys).sum(dim=-1) * scale

        bias_vec = torch.stack([self.order_biases[str(n)] for n in order_list], dim=0)  # [6]
        scores = scores + bias_vec.view(1, 1, -1)

        # Softcapping: c * tanh(scores / c)
        if self.softcap is not None and self.softcap > 0:
            scores = self.softcap * torch.tanh(scores / self.softcap)

        # Temperature scaling
        if self.temperature != 1.0 and self.temperature > 0:
            scores = scores / self.temperature

        # 4. Boundary masking
        order_to_idx = {n: i for i, n in enumerate(self.orders)}
        valid_masks = []
        for n in order_list:
            valid_masks.append(gb.valid[:, :, order_to_idx[n]])
        valid_stacked = torch.stack(valid_masks, dim=2)  # [B, T, 6] bool

        # 5. Gating / Routing
        if self.mode == "sigmoid":
            alpha = torch.sigmoid(scores)  # [B, T, 6]
            alpha = alpha.masked_fill(~valid_stacked, 0.0)
        else:
            scores = scores.masked_fill(~valid_stacked, -1e9)
            alpha = torch.softmax(scores, dim=-1)  # [B, T, 6]

        # 6. Telemetry: track activation, entropy, and frequency tiers
        with torch.no_grad():
            self._update_telemetry(alpha, valid_stacked, gb)

        # 7. Weighted sum over values: [B, T, val_dim]
        v_mix = (alpha.unsqueeze(-1) * vals).sum(dim=2)

        # 8. Output projection through zero-init U_out: [B, T, d_model]
        delta = self.u_out(v_mix)
        return delta

    def _update_telemetry(
        self,
        alpha: torch.Tensor,
        valid: torch.Tensor,
        gb: GatherBatch,
    ) -> None:
        if torch.compiler.is_compiling():
            return
        B, T, K = alpha.shape
        v_float = valid.float()
        n_valid = v_float.sum(dim=-1).clamp_min(1.0)  # [B, T]

        if self.mode == "sigmoid":
            # Mean gate activation across valid candidates
            mean_act = (alpha * v_float).sum(dim=-1) / n_valid  # [B, T]
            self._tel_act += float(mean_act.mean().detach())

            # Mean binary entropy per valid candidate
            a_clamped = alpha.clamp(1e-7, 1.0 - 1e-7)
            b_ent = -(a_clamped * torch.log(a_clamped) + (1.0 - a_clamped) * torch.log(1.0 - a_clamped))
            pos_metric = (b_ent * v_float).sum(dim=-1) / n_valid  # [B, T]
            self._tel_entropy += float(pos_metric.mean().detach())
        else:
            # Categorical Shannon entropy over candidates
            pos_metric = -(alpha * torch.log(alpha.clamp_min(1e-12))).sum(dim=-1)  # [B, T]
            self._tel_entropy += float(pos_metric.mean().detach())

        self._tel_count += 1

        # Frequency bucketing if gb.touch_counts available
        if gb.touch_counts is not None and len(gb.touch_counts) == K:
            touches = torch.stack([
                gb.touch_counts[i][gb.inverse[i]].view(B, T).float()
                for i in range(K)
            ], dim=-1)  # [B, T, 6]
            avg_touch = (touches * v_float).sum(dim=-1) / n_valid  # [B, T]

            mask_low = avg_touch <= 5.0
            mask_mid = (avg_touch > 5.0) & (avg_touch <= 50.0)
            mask_high = avg_touch > 50.0

            metric_to_bucket = pos_metric if self.mode == "softmax" else mean_act
            if mask_low.any():
                self._tel_buckets["low"].append(float(metric_to_bucket[mask_low].mean().detach()))
            if mask_mid.any():
                self._tel_buckets["mid"].append(float(metric_to_bucket[mask_mid].mean().detach()))
            if mask_high.any():
                self._tel_buckets["high"].append(float(metric_to_bucket[mask_high].mean().detach()))

    def pop_telemetry(self) -> dict[str, float]:
        """Pop mean attention entropy and frequency tier telemetry."""
        if self._tel_count == 0:
            return {}
        res = {
            "engram_attn_entropy": self._tel_entropy / self._tel_count,
        }
        if self.mode == "sigmoid":
            res["engram_act"] = self._tel_act / self._tel_count

        prefix = "engram_H" if self.mode == "softmax" else "engram_act"
        for tier, vals in self._tel_buckets.items():
            if vals:
                res[f"{prefix}_{tier}"] = sum(vals) / len(vals)

        self._reset_telemetry()
        return res


def build_engram_readout(cfg: EngramConfig, d_model: int, rms_eps: float) -> nn.Module:
    """Factory creating the configured Engram readout module."""
    if cfg.readout in ("kv", "kv_sigmoid", "kv_softmax"):
        return EngramKVReadout(cfg, d_model, rms_eps)
    elif cfg.readout == "linear":
        return EngramLinearReadout(cfg, d_model, rms_eps)
    else:
        raise ValueError(
            f"unknown engram.readout: {cfg.readout!r} (expected 'linear', 'kv', 'kv_sigmoid', or 'kv_softmax')"
        )


# Default alias
EngramReadout = EngramKVReadout
