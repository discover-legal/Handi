"""Authoring primitives behave exactly like the stock Qwen3.5 modules they target."""
import math

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

from handi.adder.design import make_config
from handi.author.qwen import Lookup, inv_freqs, rms_gain_weight, qk_norm_weight, relative_score_table
from handi.author.space import Space


def test_space_alloc_and_vec():
    sp = Space(8)
    a = sp.alloc("a", 3)
    b = sp.alloc("b", 1)
    v = sp.vec({(a, 2): 1.0, b: 0.5})
    assert v.tolist() == [0, 0, 1, 0.5, 0, 0, 0, 0]
    assert sp.used == 4


def test_lookup_reproduces_truth_table_under_real_swiglu():
    sp = Space(16)
    const, x, y, out = sp.alloc("const"), sp.alloc("x", 3), sp.alloc("y", 3), sp.alloc("out", 9)
    lk = Lookup(sp, const, [x, y], K=10.0)
    for i in range(3):
        for j in range(3):
            lk.entry((i, j), sp.basis(out, 3 * i + j))
    cfg = make_config()
    cfg.hidden_size, cfg.intermediate_size = 16, 9
    mlp = Qwen3_5MLP(cfg, 9)
    g, u, d = lk.pack(9)
    mlp.gate_proj.weight.data, mlp.up_proj.weight.data, mlp.down_proj.weight.data = g, u, d
    for i in range(3):
        for j in range(3):
            h = sp.vec({const: 1.0, (x, i): 1.0, (y, j): 1.0})
            o = mlp(h)[out.offset:out.offset + 9]
            assert o.argmax().item() == 3 * i + j
            assert abs(o.max().item() - 1.0) < 1e-3
            assert o.sum().item() - o.max().item() < 1e-2   # other entries ~0


def test_rms_gain_absorbs_active_count():
    d, n = 128, 7
    norm = Qwen3_5RMSNorm(d)
    norm.weight.data = rms_gain_weight(d, n)
    v = torch.zeros(d)
    v[:n] = 1.0
    assert torch.allclose(norm(v)[:n], torch.ones(n), atol=1e-4)


def test_qk_norm_gain_sets_logit_scale():
    hd, gain = 32, 30.0
    norm = Qwen3_5RMSNorm(hd)
    norm.weight.data = qk_norm_weight(hd, gain)
    q = torch.zeros(hd)
    q[0] = 2.0
    assert abs(norm(q).norm().item() - gain * math.sqrt(hd)) < 1e-3


def test_relative_score_peaks_at_delta():
    cfg = make_config()
    f = inv_freqs(cfg)
    pairs = {j: (float(f[j]), 1.0) for j in range(4)}
    for delta in [0, 5, 11, 22]:
        t = relative_score_table(pairs, delta, 33)
        assert t.argmax().item() == delta
