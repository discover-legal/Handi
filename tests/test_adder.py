"""Rung 1: the hand-authored adder, run through the stock Qwen3_5ForCausalLM with the KV cache."""
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from handi.adder import tokenizer as tok
from handi.adder.design import N_DIST, build, relative_margins
from handi.adder.run import add, add_batch, build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    m, _ = build_model()
    return m


def test_state_dict_loads_strictly_into_stock_class():
    cfg, sd, _ = build()
    m = Qwen3_5ForCausalLM(cfg)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected
    assert all(k.endswith("inv_freq") for k in missing)


def test_offset_heads_have_margin():
    for delta, margin in relative_margins().items():
        assert margin > 0.1, (delta, margin)


def test_edge_cases(model):
    fx = json.loads((ROOT / "tests" / "fixtures" / "adder_pairs.json").read_text())
    for a, b in fx["edge_cases"]:
        assert add(model, a, b) == a + b


def test_adderboard_10k(model):
    fx = json.loads((ROOT / "tests" / "fixtures" / "adder_pairs.json").read_text())
    cases = [tuple(p) for p in fx["edge_cases"]] + [tuple(p) for p in fx["random_cases"]]
    passed = 0
    for i in range(0, len(cases), 1000):
        chunk = cases[i:i + 1000]
        passed += sum(r == a + b for (a, b), r in zip(chunk, add_batch(model, chunk)))
    assert passed == len(cases), f"{passed}/{len(cases)}"


def test_attention_patterns_are_sharp():
    """Every designed head attends > 0.99 to its intended key (teacher-forced full sequence)."""
    cfg, sd, _ = build()
    cfg._attn_implementation = "eager"
    m = Qwen3_5ForCausalLM(cfg)
    m.load_state_dict(sd, strict=False)
    m.eval()
    a, b = 1_234_567_890, 9_876_543_210
    full = tok.encode_prompt(a, b) + [int(c) for c in f"{a + b:011d}"]
    with torch.no_grad():
        att = m(torch.tensor([full]), output_attentions=True).attentions
    pred = list(range(22, 33))
    for q in pred:
        assert att[0][0, 0, q, q - 22] > 0.99      # head A: a digit of the column
        assert att[0][0, 1, q, q - 11] > 0.99      # head B: b digit of the column
        for d in range(N_DIST):
            assert att[0][0, 2 + d, q, q - d] > 0.99
    # head D: nearest lower column whose digits do not propagate, else BOS
    sa, sb = f"{a:010d}", f"{b:010d}"
    for q in pred:
        k = 32 - q
        target = 0
        for j in range(k - 1, -1, -1):
            if int(sa[9 - j]) + int(sb[9 - j]) != 9:
                target = 21 - j
                break
        assert att[1][0, 0, q, target] > 0.99, (q, target, att[1][0, 0, q].argmax().item())
