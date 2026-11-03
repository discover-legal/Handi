"""Authoring primitives that emit weights for the stock Hugging Face Qwen3.5 text model.

Everything here mirrors `transformers.models.qwen3_5.modeling_qwen3_5` exactly:
  * RMSNorm is zero-centered: y = x / rms(x) * (1 + w).
  * q_proj emits, per head, [query (head_dim) | gate (head_dim)]; attn output *= sigmoid(gate).
  * q and k are RMS-normalized per head (shared gain vector across heads in a layer) BEFORE RoPE.
  * RoPE acts on the first `head_dim * partial_rotary_factor` dims with the rotate_half layout:
    pair j = (dims j, j + rot/2) rotates at inv_freq[j] radians per position.
  * logits = (q . k) * head_dim ** -0.5 ; MLP is SwiGLU: down(silu(gate(x)) * up(x)).
No weight is typed as a literal matrix: every entry comes from a slot reference and a design
constant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from .space import Slot, Space


# --------------------------------------------------------------------------------------------
# Rotary helpers (exactly what the model computes)
# --------------------------------------------------------------------------------------------

def rotary_dim(config: Qwen3_5TextConfig) -> int:
    return int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))


def inv_freqs(config: Qwen3_5TextConfig) -> torch.Tensor:
    inv, _ = Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters(config)
    return inv


def theta_for_pair_frequency(config_head_dim: int, partial: float, pair: int, freq: float) -> float:
    """rope_theta such that rotary pair `pair` rotates at `freq` radians per position.

    inv_freq[j] = theta ** (-2 j / rot_dim). Pair 0 is always 1 rad/position regardless of theta.
    """
    rot = int(config_head_dim * partial)
    if pair == 0:
        raise ValueError("pair 0 always has frequency 1; choose pair >= 1")
    return freq ** (-rot / (2 * pair))


# --------------------------------------------------------------------------------------------
# One attention head as a program
# --------------------------------------------------------------------------------------------

@dataclass
class Head:
    """Weights of one head, written in residual-space terms.

    q, k, v: [head_dim, d] read matrices; o: [d, head_dim] write matrix; gate: [head_dim, d].
    """
    space: Space
    head_dim: int
    q: torch.Tensor = field(init=False)
    k: torch.Tensor = field(init=False)
    v: torch.Tensor = field(init=False)
    o: torch.Tensor = field(init=False)
    gate: torch.Tensor = field(init=False)

    def __post_init__(self):
        d = self.space.d
        self.q = torch.zeros(self.head_dim, d)
        self.k = torch.zeros(self.head_dim, d)
        self.v = torch.zeros(self.head_dim, d)
        self.o = torch.zeros(d, self.head_dim)
        self.gate = torch.zeros(self.head_dim, d)

    # -- reads: head dim `i` <- residual (slot, idx) * value
    def read(self, mat: torch.Tensor, i: int, slot: Slot, idx: int = 0, value: float = 1.0) -> None:
        mat[i, slot.index(idx)] += value

    def q_read(self, i, slot, idx=0, value=1.0): self.read(self.q, i, slot, idx, value)
    def k_read(self, i, slot, idx=0, value=1.0): self.read(self.k, i, slot, idx, value)
    def v_read(self, i, slot, idx=0, value=1.0): self.read(self.v, i, slot, idx, value)

    def write(self, i: int, slot: Slot, idx: int = 0, value: float = 1.0) -> None:
        """o: residual (slot, idx) <- head dim i * value."""
        self.o[slot.index(idx), i] += value

    def copy(self, src: Slot, dst: Slot, first_dim: int, scale: float = 1.0) -> None:
        """OV copy of a whole slot from the attended position into `dst`, through head dims."""
        if src.dim != dst.dim:
            raise ValueError("copy needs equal slot dims")
        if first_dim + src.dim > self.head_dim:
            raise ValueError("not enough head dims for copy")
        for j in range(src.dim):
            self.v_read(first_dim + j, src, j, 1.0)
            self.write(first_dim + j, dst, j, scale)

    def open_gate(self, const: Slot, logit: float = 10.0) -> None:
        """Make sigmoid(gate) ~ 1 on every head dim by reading the constant slot."""
        for i in range(self.head_dim):
            self.gate[i, const.index()] = logit

    def relative_position_query(self, const: Slot, pairs: dict[int, tuple[float, float]], delta: int,
                                rot_dim: int) -> None:
        """Make q.k peak at key position = query position - delta, using rotary pairs.

        pairs: {pair_index: (inv_freq, amplitude)}. The matching key must put (1, 0) in the same pairs
        (see `relative_position_key`). With rotate_half RoPE, q_m . k_n = amp * cos(f * (m - n) + phi),
        so phi = -f * delta puts the peak at m - n = delta.
        """
        half = rot_dim // 2
        for pair, (f, amp) in pairs.items():
            phi = -f * delta
            self.q_read(pair, const, 0, amp * math.cos(phi))
            self.q_read(pair + half, const, 0, amp * math.sin(phi))

    def relative_position_key(self, const: Slot, pairs: dict[int, tuple[float, float]], rot_dim: int,
                              minus: Slot | None = None) -> None:
        """Key side of relative-position matching: (amp, 0) in each pair, read from the constant slot.

        `minus` optionally subtracts a slot so that positions carrying it (e.g. BOS) get no rotary term.
        """
        for pair, (_, amp) in pairs.items():
            self.k_read(pair, const, 0, amp)
            if minus is not None:
                self.k_read(pair, minus, 0, -amp)


def relative_score_table(pairs: dict[int, tuple[float, float]], delta: int, max_delta: int) -> torch.Tensor:
    """Unnormalized q.k for Delta = 0..max_delta given the pair design (for margin assertions)."""
    out = torch.zeros(max_delta + 1)
    for d in range(max_delta + 1):
        out[d] = sum(amp * math.cos(f * (d - delta)) for f, amp in pairs.values())
    return out


# --------------------------------------------------------------------------------------------
# SwiGLU lookup tables
# --------------------------------------------------------------------------------------------

@dataclass
class Lookup:
    """Truth table implemented with one SwiGLU neuron per entry.

    Inputs are one-hot slots (post-norm values ~1). For an entry selecting index i_s in each input
    slot s, the neuron computes
        pre  = sum_s x[s, i_s] - (n_inputs - 1) * x[const]      (1 if all match, <= 0 otherwise)
        act  = silu(K * pre) * pre
    which is K for a full match, 0 for one mismatch, and ~K e^-K for more. down_proj writes
    out_vec / K so a matched entry adds exactly out_vec to the residual.
    An input may be given as (slot, "neg") to use 1 - x[slot] (slot must be a 1-dim indicator).
    """
    space: Space
    const: Slot
    inputs: list[Slot]
    K: float = 10.0
    rows: list[tuple[tuple[int, ...], torch.Tensor]] = field(default_factory=list)

    def entry(self, idxs: tuple[int, ...], out: torch.Tensor) -> None:
        if len(idxs) != len(self.inputs):
            raise ValueError("one index per input slot")
        self.rows.append((tuple(idxs), out.clone()))

    def pack(self, intermediate: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (gate_proj [I, d], up_proj [I, d], down_proj [d, I])."""
        d = self.space.d
        if len(self.rows) > intermediate:
            raise ValueError(f"{len(self.rows)} entries do not fit intermediate_size={intermediate}")
        up = torch.zeros(intermediate, d)
        down = torch.zeros(d, intermediate)
        n = len(self.inputs)
        for r, (idxs, out) in enumerate(self.rows):
            for slot, i in zip(self.inputs, idxs):
                up[r, slot.index(i)] += 1.0
            up[r, self.const.index()] += -(n - 1)
            down[:, r] = out / self.K
        gate = up * self.K
        return gate, up, down


# --------------------------------------------------------------------------------------------
# Norm absorption
# --------------------------------------------------------------------------------------------

def rms_gain_weight(d: int, n_active: float) -> torch.Tensor:
    """Zero-centered RMSNorm weight so that a vector with `n_active` unit slots (of d) comes out
    with those slots exactly 1: (1 + w) = sqrt(n_active / d)."""
    return torch.full((d,), math.sqrt(n_active / d) - 1.0)


def qk_norm_weight(head_dim: int, gain: float) -> torch.Tensor:
    """Per-head q/k RMSNorm weight giving ||q_normed|| = gain * sqrt(head_dim). Attention logits
    between unit-direction q and k are then gain_q * gain_k * sqrt(head_dim) * cos(angle)."""
    return torch.full((head_dim,), gain - 1.0)


# --------------------------------------------------------------------------------------------
# Layer and model assembly
# --------------------------------------------------------------------------------------------

@dataclass
class AttentionLayer:
    heads: list[Head]
    q_gain: float
    k_gain: float

    def pack(self, config: Qwen3_5TextConfig, prefix: str) -> dict[str, torch.Tensor]:
        H, hd, d = config.num_attention_heads, config.head_dim, config.hidden_size
        if config.num_key_value_heads != H:
            raise NotImplementedError("GQA packing not needed for the adder; heads must equal kv heads")
        if len(self.heads) != H:
            raise ValueError(f"layer needs exactly {H} heads (zero heads are fine)")
        q = torch.zeros(H * hd * 2, d)
        k = torch.zeros(H * hd, d)
        v = torch.zeros(H * hd, d)
        o = torch.zeros(d, H * hd)
        for h, head in enumerate(self.heads):
            q[h * 2 * hd: h * 2 * hd + hd] = head.q
            q[h * 2 * hd + hd: (h + 1) * 2 * hd] = head.gate
            k[h * hd: (h + 1) * hd] = head.k
            v[h * hd: (h + 1) * hd] = head.v
            o[:, h * hd: (h + 1) * hd] = head.o
        return {
            f"{prefix}.q_proj.weight": q,
            f"{prefix}.k_proj.weight": k,
            f"{prefix}.v_proj.weight": v,
            f"{prefix}.o_proj.weight": o,
            f"{prefix}.q_norm.weight": qk_norm_weight(hd, self.q_gain),
            f"{prefix}.k_norm.weight": qk_norm_weight(hd, self.k_gain),
        }


def pack_mlp(lookup: Lookup, config: Qwen3_5TextConfig, prefix: str) -> dict[str, torch.Tensor]:
    gate, up, down = lookup.pack(config.intermediate_size)
    return {f"{prefix}.gate_proj.weight": gate, f"{prefix}.up_proj.weight": up, f"{prefix}.down_proj.weight": down}


def zero_head(space: Space, head_dim: int) -> Head:
    return Head(space, head_dim)
