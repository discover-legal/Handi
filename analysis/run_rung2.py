"""Rung 2 driver: build each condition's weights for each model, load into the stock HF class,
score M1-M4, write results/tables/rung2.csv.

Usage: .venv/bin/python analysis/run_rung2.py --models tinystories-1m gpt2 --conditions A0 A1 A5 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis import registry  # noqa: E402
from handi.analysis.loader import load_canonical  # noqa: E402
from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import generators  # noqa: E402
from handi.authoring.evaluate import evaluate_all, load_authored, load_trained  # noqa: E402

SRC = {"gpt2": "F32", "gpt_neo": "F32", "gpt_neox": "F16", "qwen3_5": "BF16"}


def rows_from(res: dict, w: ResultsWriter, **common) -> None:
    for r in res.get("m1", []):
        c = r["corpus"]
        w.row(metric=f"m1_ce_{c}", value=r["nats"], null_mean=r["ci_lo"], null_sd=r["ci_hi"], null_kind="bootstrap95",
              null_n=r["n_tokens"], **common)
        for ref in ("uniform", "unigram", "bigram"):
            w.row(metric=f"m1_ref_{ref}_{c}", value=r[ref], **common)
    if "m2" in res:
        w.row(metric="m2_copy_acc", value=res["m2"], **common)
    if "m3" in res:
        for k, v in res["m3"].items():
            if isinstance(v, (int, float)):
                w.row(metric=f"m3_{k}", value=v, **common)
    if "m4" in res:
        for task, r in res["m4"].items():
            w.row(metric=f"m4_{task}", value=r["accuracy"], null_mean=r["ci_lo"], null_sd=r["ci_hi"], null_kind="normal95",
                  null_n=r["n"], ks=r["majority"], **common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["tinystories-1m"])
    ap.add_argument("--conditions", nargs="*", default=["A0", "A1", "A5"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--which", nargs="*", default=["m1", "m2", "m3", "m4"])
    ap.add_argument("--m1-tokens", type=int, default=100_000)
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung2.csv"))
    a = ap.parse_args()
    with ResultsWriter(a.out) as w:
        for name in a.models:
            canon, cfg, fam = load_canonical(name)
            for cond in a.conditions:
                seeds = [0] if cond == "A5" else a.seeds
                for seed in seeds:
                    t0 = time.time()
                    if cond == "A5":
                        model, tok = load_trained(name)
                    else:
                        weights = generators.CONDITIONS[cond](canon, cfg, fam, seed)
                        model, tok = load_authored(name, weights)
                    res = evaluate_all(model, tok, a.which, a.m1_tokens)
                    rows_from(res, w, hypothesis="H13", model=name, family=fam, layer="ALL", component=cond,
                              seed=seed, src_dtype=SRC[fam])
                    m1 = {r["corpus"]: round(r["nats"], 3) for r in res.get("m1", [])}
                    print(f"{name:16s} {cond} seed={seed} m1={m1} m2={res.get('m2')} "
                          f"m4={ {k: round(v['accuracy'], 3) for k, v in res.get('m4', {}).items()} } ({time.time() - t0:.0f}s)", flush=True)
                    del model
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
