"""Rung 1: a 10-digit adder hand-authored into the stock Qwen3.5 text architecture.

Sequence (34 positions, MSB first):   BOS a0..a9 + b0..b9 = s0..s10
Column k (k=0 units .. k=10 overflow): a_k at position 10-k, b_k at 21-k, s_k predicted at 32-k.
Every match the circuit needs is at a FIXED relative offset, which RoPE encodes natively:
    prediction -> a digit of its column : Delta = 22
    prediction -> b digit of its column : Delta = 11
    b digit    -> a digit of its column : Delta = 11
and the only "absolute" information needed, the column index, is recovered as a one-hot distance
to a landmark token: a prediction position is at distance 10-k from '=', a b digit at distance
10-k from '+'. Eleven heads each copy the token KIND found at offset d (d = 0..10) into kind_at[d],
so kind_at[d] == EQ marks "I am d after '='" and kind_at[d] == PLUS marks "I am d after '+'".

Layer 0 (13 of 16 heads):
    A        copies tok.digit at Delta=22 into work_a
    B        copies tok.digit at Delta=11 into work_b
    K[d]     copies tok.kind  at Delta=d  into kind_at[d]      (d = 0..10)
  MLP 0:     flag = Generate (tok.digit + work_b >= 10) | Propagate (== 9) | Kill
             (meaningful at b positions; computed everywhere, never read elsewhere)
Layer 1 (1 of 16 heads):
    D        carry-lookahead. Query at a prediction position with distance dp to '=' scores a
             b-digit key with distance db to '+' by the table T[dp, db] = C - (db - dp - 1) when
             db > dp (a lower column; nearest lower column scores highest) and 0 otherwise.
             Propagate keys get a large penalty, BOS gets a constant sink score between 0 and
             the lowest valid score, every other key scores 0. The attended key's flag is copied:
             carry1 <- Generate, carry0 <- Kill | Propagate.
  MLP 1:     out.digit[(work_a + work_b + carry) mod 10]
lm_head reads out.digit. Greedy decoding of 11 tokens gives the sum.
"""
from __future__ import annotations

import torch
from transformers import Qwen3_5TextConfig

from ..author.qwen import (AttentionLayer, Head, Lookup, inv_freqs, pack_mlp, relative_score_table,
                           rms_gain_weight, rotary_dim, zero_head)
from ..author.space import Space
from . import tokenizer as tok

D_MODEL = 128
HEAD_DIM = 32
N_HEADS = 16
PARTIAL_ROTARY = 0.25          # as in Qwen3.5-0.8B: 8 rotary dims = 4 pairs
ROPE_THETA = 10.0              # gives pair frequencies [1, .56, .32, .18]: sharpest peaks for offsets <= 22
INTERMEDIATE = 200
QK_GAIN = 30.0                 # per-side q/k norm gain; logits = gain^2 * sqrt(head_dim) * cos(angle)
LOOKUP_K = 10.0
LM_SCALE = 10.0
GATE_LOGIT = 10.0

DELTA_A = 22
DELTA_B = 11
N_DIST = 11                    # distances 0..10

# head D table constants (unnormalized dot products; the test checks the realized attention)
D_C = 12.0                     # score of the nearest lower column
D_STEP = 1.0                   # decrease per column further down
D_BOS = 1.5                    # sink: above 0 (every irrelevant key), below C - 9*STEP = 3
D_PENALTY = 20.0               # Propagate keys

COUNTS_LAYER0_IN = 3                     # const, digit, kind
COUNTS_LAYER0_MID = 3 + 2 + N_DIST       # + work_a, work_b, kind_at[0..10]
COUNTS_LAYER1_IN = COUNTS_LAYER0_MID + 1 # + flag
COUNTS_LAYER1_MID = COUNTS_LAYER1_IN + 1 # + carry
COUNTS_FINAL = COUNTS_LAYER1_MID + 1     # + out

KIND_DIGIT, KIND_PLUS, KIND_EQ, KIND_BOS = 0, 1, 2, 3
KILL, PROP, GEN = 0, 1, 2
CARRY1, CARRY0 = 0, 1


def make_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=len(tok.VOCAB),
        hidden_size=D_MODEL,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        head_dim=HEAD_DIM,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA, "partial_rotary_factor": PARTIAL_ROTARY},
        bos_token_id=tok.BOS,
    )


def make_space() -> Space:
    sp = Space(D_MODEL)
    sp.alloc("const", 1)
    sp.alloc("digit", 10)
    sp.alloc("kind", 4)        # digit, plus, eq, bos
    sp.alloc("work_a", 10)
    sp.alloc("work_b", 10)
    for d in range(N_DIST):
        sp.alloc(f"kind_at{d}", 4)
    sp.alloc("flag", 3)        # kill, propagate, generate
    sp.alloc("carry", 2)       # [carry1, carry0]
    sp.alloc("out", 10)
    return sp


def lookahead_table() -> torch.Tensor:
    """T[dp, db]: score for a prediction at distance dp from '=' attending a b digit at distance db from '+'."""
    T = torch.zeros(N_DIST, N_DIST)
    for dp in range(N_DIST):
        for db in range(1, N_DIST):
            if db > dp:
                T[dp, db] = D_C - D_STEP * (db - dp - 1)
    return T


def build() -> tuple[Qwen3_5TextConfig, dict[str, torch.Tensor], Space]:
    cfg = make_config()
    sp = make_space()
    const, digit, kind = sp["const"], sp["digit"], sp["kind"]
    work_a, work_b, flag, carry, out = sp["work_a"], sp["work_b"], sp["flag"], sp["carry"], sp["out"]
    kind_at = [sp[f"kind_at{d}"] for d in range(N_DIST)]
    rot = rotary_dim(cfg)
    f = inv_freqs(cfg)
    pairs = {j: (float(f[j]), 1.0) for j in range(rot // 2)}
    sd: dict[str, torch.Tensor] = {}

    # ---- embeddings: every token = const + digit one-hot + kind one-hot (3 active slots)
    emb = torch.zeros(len(tok.VOCAB), D_MODEL)
    for t in range(10):
        emb[t] = sp.vec({const: 1.0, (digit, t): 1.0, (kind, KIND_DIGIT): 1.0})
    emb[tok.PLUS] = sp.vec({const: 1.0, (digit, 0): 1.0, (kind, KIND_PLUS): 1.0})
    emb[tok.EQ] = sp.vec({const: 1.0, (digit, 0): 1.0, (kind, KIND_EQ): 1.0})
    emb[tok.BOS] = sp.vec({const: 1.0, (digit, 0): 1.0, (kind, KIND_BOS): 1.0})
    sd["model.embed_tokens.weight"] = emb

    # ---- layer 0: offset-copy heads
    def offset_copy(delta: int, src, dst) -> Head:
        h = Head(sp, HEAD_DIM)
        h.relative_position_query(const, pairs, delta, rot)
        h.relative_position_key(const, pairs, rot)
        h.copy(src, dst, first_dim=rot)
        h.open_gate(const, GATE_LOGIT)
        return h

    heads0 = [offset_copy(DELTA_A, digit, work_a), offset_copy(DELTA_B, digit, work_b)]
    heads0 += [offset_copy(d, kind, kind_at[d]) for d in range(N_DIST)]
    heads0 += [zero_head(sp, HEAD_DIM) for _ in range(N_HEADS - len(heads0))]
    sd.update(AttentionLayer(heads0, QK_GAIN, QK_GAIN).pack(cfg, "model.layers.0.self_attn"))

    # ---- layer 0 MLP: flags from (tok.digit, work_b)
    lk0 = Lookup(sp, const, [digit, work_b], K=LOOKUP_K)
    for a in range(10):
        for b in range(10):
            s = a + b
            lk0.entry((a, b), sp.basis(flag, GEN if s >= 10 else (PROP if s == 9 else KILL)))
    sd.update(pack_mlp(lk0, cfg, "model.layers.0.mlp"))

    # ---- layer 1: carry-lookahead head
    T = lookahead_table()
    hD = Head(sp, HEAD_DIM)
    for i in range(1, N_DIST):                       # head dim rot+i <-> "key is at distance i from '+'"
        hD.k_read(rot + i, kind_at[i], KIND_PLUS, 1.0)
        for dp in range(N_DIST):                     # query carries row T[dp, :] when kind_at[dp] == EQ
            if T[dp, i] != 0:
                hD.q_read(rot + i, kind_at[dp], KIND_EQ, float(T[dp, i]))
    pen, sink = rot + N_DIST, rot + N_DIST + 1
    hD.q_read(pen, const, 0, -D_PENALTY); hD.k_read(pen, flag, PROP, 1.0)
    hD.q_read(sink, const, 0, D_BOS);     hD.k_read(sink, kind, KIND_BOS, 1.0)
    v1, v0 = rot + N_DIST + 2, rot + N_DIST + 3
    hD.v_read(v1, flag, GEN, 1.0); hD.write(v1, carry, CARRY1, 1.0)
    hD.v_read(v0, flag, KILL, 1.0); hD.v_read(v0, flag, PROP, 1.0); hD.write(v0, carry, CARRY0, 1.0)
    hD.open_gate(const, GATE_LOGIT)
    heads1 = [hD] + [zero_head(sp, HEAD_DIM) for _ in range(N_HEADS - 1)]
    sd.update(AttentionLayer(heads1, QK_GAIN, QK_GAIN).pack(cfg, "model.layers.1.self_attn"))

    # ---- layer 1 MLP: out digit from (work_a, work_b, carry)
    lk1 = Lookup(sp, const, [work_a, work_b, carry], K=LOOKUP_K)
    for a in range(10):
        for b in range(10):
            for c_idx, c in ((CARRY1, 1), (CARRY0, 0)):
                lk1.entry((a, b, c_idx), sp.basis(out, (a + b + c) % 10))
    sd.update(pack_mlp(lk1, cfg, "model.layers.1.mlp"))

    # ---- norms absorb the constant active-slot count at each stage
    sd["model.layers.0.input_layernorm.weight"] = rms_gain_weight(D_MODEL, COUNTS_LAYER0_IN)
    sd["model.layers.0.post_attention_layernorm.weight"] = rms_gain_weight(D_MODEL, COUNTS_LAYER0_MID)
    sd["model.layers.1.input_layernorm.weight"] = rms_gain_weight(D_MODEL, COUNTS_LAYER1_IN)
    sd["model.layers.1.post_attention_layernorm.weight"] = rms_gain_weight(D_MODEL, COUNTS_LAYER1_MID)
    sd["model.norm.weight"] = rms_gain_weight(D_MODEL, COUNTS_FINAL)

    # ---- unembedding: digits read out.digit; '+', '=', BOS stay at logit 0
    lm = torch.zeros(len(tok.VOCAB), D_MODEL)
    for t in range(10):
        lm[t] = sp.basis(out, t) * LM_SCALE
    sd["lm_head.weight"] = lm
    return cfg, sd, sp


def relative_margins() -> dict[int, float]:
    """Cosine margin of each offset-copy head's peak over its best competitor (|q| = |k| = 2)."""
    cfg = make_config()
    f = inv_freqs(cfg)
    pairs = {j: (float(f[j]), 1.0) for j in range(rotary_dim(cfg) // 2)}
    out = {}
    for delta in [DELTA_A, DELTA_B] + list(range(N_DIST)):
        t = relative_score_table(pairs, delta, tok.N_CTX - 1)
        others = torch.cat([t[:delta], t[delta + 1:]]).max().item()
        out[delta] = (t[delta].item() - others) / float(len(pairs))
    return out


def count_params(sd: dict[str, torch.Tensor]) -> dict[str, int]:
    return {"dense": sum(v.numel() for v in sd.values()), "nonzero": sum(int((v != 0).sum()) for v in sd.values())}
