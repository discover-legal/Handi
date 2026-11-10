"""Rung 3 read-out on LexGLUE LEDGAR, beside the model (owner, 2026-11-10: LEDGAR read beside the model).

LEDGAR (coastalcph/lex_glue, config ledgar, CC BY 4.0): sort a contract provision into one of 100 types. The
output-row edit does not fit (many of the 100 type names share their first token, and 15 labelled examples
cannot cover 100 types), so this is a read-out beside the model, as for the extraction tasks: the model file is
not changed and its own reply is not used. The answer is computed outside the model from its final state.

  state     Qwen3.5-0.8B, the final state (the one the output table reads) at the last token of the prompt
            "Contract provision: {text}\\nThe type of this provision is:"; provisions are read whole (at most
            910 tokens).
  read-out  the K-label read-out of run_rung3_all.py without the row edit: one standardised ridge per type,
            from the state to +1 for that type's items and -1 for the others, lambda = n; the answer is the type
            with the highest score. Types with no fit item cannot be predicted.
  fit       1,000 provisions drawn at random from the training split (seed 300), one draw.
  scored    the whole test split (10,000 provisions).
  baseline  wordcount: the same ridges on counts of the provision's words (words in at least two fit items);
            majority: always the commonest type of the fit items.
Scores, as LexGLUE reports LEDGAR: micro-F1 (= accuracy, one type per item) and macro-F1 (mean over the 100 types
of each type's F1). Rows: results/tables/rung3_ledgar.csv (hypothesis R3L).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import evaluate as EV  # noqa: E402
from analysis.run_rung3 import collect  # noqa: E402
from analysis.run_rung3_all import ridge_any, word_preds  # noqa: E402

PROMPT = "Contract provision: {text}\nThe type of this provision is:"
K = 100


def split(name):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("coastalcph/lex_glue", f"ledgar/{name}-00000-of-00001.parquet", repo_type="dataset"))
    return df.text.astype(str).tolist(), df.label.astype(int).to_numpy()


def type_names():
    from huggingface_hub import hf_hub_download
    t = open(hf_hub_download("coastalcph/lex_glue", "README.md", repo_type="dataset")).read()
    seg = t[t.find("config_name: ledgar"):]
    names = re.findall(r"'(\d+)': (.+)", seg)[:K]
    assert [int(i) for i, _ in names] == list(range(K))
    return [n.strip() for _, n in names]


def macro_f1(p, g):
    f = []
    for c in range(K):
        tp = np.sum((p == c) & (g == c)); fp = np.sum((p == c) & (g != c)); fn = np.sum((p != c) & (g == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def states(model, tok, texts, cache):
    if cache and cache.exists():
        return np.load(cache)["H"]
    H = []
    for i in range(0, len(texts), 50):
        h, _, _ = collect(model, tok, [PROMPT.format(text=t) for t in texts[i:i + 50]], max_len=10 ** 9)
        H.append(h.astype(np.float32))
        if i % 1000 == 0:
            print(f"  states {i}/{len(texts)}", flush=True)
    H = np.concatenate(H)
    if cache:
        np.savez(cache, H=H)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--fit", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung3_ledgar.csv"))
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    t0 = time.time()
    tr_x, tr_y = split("train"); te_x, te_y = split("test"); names = type_names()
    rng = np.random.default_rng(300)
    idx = np.sort(rng.choice(len(tr_x), a.fit, replace=False))
    fx, fy = [tr_x[i] for i in idx], tr_y[idx]
    cdir = Path(a.cache) if a.cache else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    model, tok = EV.load_trained(a.model)
    Hf = states(model, tok, fx, cdir / f"fit{a.fit}.npz" if cdir else None).astype(np.float64)
    He = states(model, tok, te_x, cdir / "test.npz" if cdir else None).astype(np.float64)
    S = np.zeros((len(te_x), K)) - np.inf
    for k in np.unique(fy):
        w, b = ridge_any(Hf, 2.0 * (fy == k) - 1.0, lam=float(len(fy)))
        S[:, k] = He @ w + b
    P = {"readout": S.argmax(1)}
    # word counts: word_preds works on label indices 0..K'-1 of the types present in the fit items
    present = np.unique(fy); remap = {c: i for i, c in enumerate(present)}
    pw, nvocab = word_preds(fx, np.array([remap[c] for c in fy]), te_x, len(present))
    P["wordcount"] = present[pw]
    P["majority"] = np.full(len(te_x), np.bincount(fy, minlength=K).argmax())
    base = dict(hypothesis="R3L", model=a.model, family="qwen3_5", src_dtype="BF16", seed=300, layer="ALL")
    with ResultsWriter(a.out) as w:
        for v, p in P.items():
            w.row(metric="ledgar_micro_f1", component=v, value=float((p == te_y).mean()), null_n=len(te_y), **base)
            w.row(metric="ledgar_macro_f1", component=v, value=macro_f1(p, te_y), null_n=len(te_y), **base)
        for n, val in [("n_fit", a.fit), ("types_in_fit", len(present)), ("n_test", len(te_y)), ("vocab", nvocab)]:
            w.row(metric="info_ledgar", component=n, value=val, **base)
    for v, p in P.items():
        print(f"{v}: micro-F1 {np.mean(p == te_y):.3f} macro-F1 {macro_f1(p, te_y):.3f}")
    worst = sorted(range(K), key=lambda c: np.mean(P['readout'][te_y == c] == c) if np.any(te_y == c) else 1)[:5]
    print("types in fit:", len(present), "| hardest types:", [names[c] for c in worst], f"| {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
