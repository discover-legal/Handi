"""Build the adder in the stock Qwen3_5ForCausalLM and run it with the KV cache.

    python -m handi.adder.run 123 456
    python -m handi.adder.run --eval          # AdderBoard edge cases + 10k pairs

Also exposes the AdderBoard interface: build_model() -> (model, metadata), add(model, a, b) -> int.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM

from . import tokenizer as tok
from .design import build, count_params

ROOT = Path(__file__).resolve().parents[2]


def build_model(device: str | None = None) -> tuple[Qwen3_5ForCausalLM, dict]:
    cfg, sd, sp = build()
    model = Qwen3_5ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    meta = {
        "name": "Handi Rung-1 adder (Qwen3.5 architecture, hand-authored, zero training)",
        "author": "hand-authored by Claude Fable 5.1",
        "params": count_params(sd),
        "architecture": "Qwen3_5ForCausalLM: 2 full-attention layers, d=128, 16 heads x 32, SwiGLU 200, RMSNorm, partial RoPE 0.25",
        "tricks": ["relative-position heads via RoPE at offsets 22, 11 and 0..10 (distance-to-landmark one-hots)", "carry-lookahead attention head",
                   "SwiGLU truth-table MLPs", "RMSNorm gains absorb constant active-slot counts"],
    }
    return model, meta


def _generate(model: Qwen3_5ForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    """Greedy, KV-cached generation of the 11 sum digits.

    The attention mask is passed explicitly: prompts are all the same length and contain no padding,
    and generate() would otherwise infer a mask from pad_token_id, which must never equal a real
    prompt token ('=' ends every prompt).
    """
    return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=tok.N_SUM, do_sample=False,
                          use_cache=True, eos_token_id=None, pad_token_id=tok.BOS)


@torch.no_grad()
def add(model: Qwen3_5ForCausalLM, a: int, b: int) -> int:
    ids = torch.tensor([tok.encode_prompt(a, b)], device=model.device)
    out = _generate(model, ids)
    return tok.decode_sum(out[0, ids.shape[1]:].tolist())


@torch.no_grad()
def add_batch(model: Qwen3_5ForCausalLM, pairs: list[tuple[int, int]]) -> list[int]:
    ids = torch.tensor([tok.encode_prompt(a, b) for a, b in pairs], device=model.device)
    out = _generate(model, ids)
    res = []
    for row in out[:, ids.shape[1]:].tolist():
        try:
            res.append(tok.decode_sum(row))
        except ValueError:
            res.append(-1)
    return res


def evaluate(model, batch: int = 500) -> dict:
    fx = json.loads((ROOT / "tests" / "fixtures" / "adder_pairs.json").read_text())
    cases = [tuple(p) for p in fx["edge_cases"]] + [tuple(p) for p in fx["random_cases"]]
    passed, failures = 0, []
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        res = add_batch(model, chunk)
        for (a, b), r in zip(chunk, res):
            if r == a + b:
                passed += 1
            elif len(failures) < 20:
                failures.append((a, b, r))
    return {"passed": passed, "total": len(cases), "accuracy": 100.0 * passed / len(cases), "failures": failures}


def main(argv: list[str]) -> None:
    model, meta = build_model()
    if argv and argv[0] == "--eval":
        r = evaluate(model)
        print(json.dumps({"meta": meta, **r}, indent=2))
        return
    a, b = int(argv[0]), int(argv[1])
    print(f"{a} + {b} = {add(model, a, b)}   (true {a + b})")


if __name__ == "__main__":
    main(sys.argv[1:])
