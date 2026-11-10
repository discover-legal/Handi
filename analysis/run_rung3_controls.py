"""Control tests for the Rung 3 formula-written edit, on the three tasks that have never-seen items
(overruling, definition_classification, cuad_governing_law). Same fit sample (500 items) and same never-seen
items as run_rung3_transfer.py; same formula (fit_edit). All scoring is on the never-seen items.

  1. shuffled   The Yes/No labels of the 500 fit items are permuted (20 permutations). Calibration does not use
                labels, so the read-out alone (sign of w.h + b) is reported next to the full edit.
  2. wordcount  The same ridge (standardised, lambda = n) fitted on word counts of the item text instead of the
                model's state, with the same 500 labels. Also the best lambda of a grid picked on the never-seen
                items themselves, which favours the baseline.
  3. fewer      The edit fitted on 10, 25, 50, 100, 250 of the 500 fit items (20 random subsets each).
  4. nearcopy   Never-seen items that resemble a fit item (word-set Jaccard >= 0.9 / 0.7 / 0.5) are dropped; for
                governing law also every never-seen clause whose contract supplied a fit clause.
  5. untrained  The same recipe on the same architecture with fresh-initialised weights (the existing A0 block).

Rows: results/tables/rung3_controls.csv (hypothesis R3X). component = variant; value = accuracy on never-seen
items; ks = majority rate; null_mean / null_sd = min / max over repeats where there are repeats (null_kind minmax).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis.loader import load_canonical  # noqa: E402
from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import evaluate as EV  # noqa: E402
from handi.authoring import generators  # noqa: E402
from analysis.run_rung3 import collect, ridge  # noqa: E402
from analysis.run_rung3_transfer import fit_edit, apply  # noqa: E402

TASKS = ["overruling", "definition_classification", "cuad_governing_law"]
REPEATS = 20
SIZES = [10, 25, 50, 100, 250, 500]
THRESHOLDS = [0.9, 0.7, 0.5]
LAM_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]          # multiples of n
FT = [0, 1]                                        # label columns of the compact logit table


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def items(task):
    """The Rung 3 fit sample and the never-seen items, exactly as run_rung3_transfer.py draws them."""
    all_rows = EV.legalbench_items(task, 10 ** 9); rows = EV.legalbench_items(task, 500)
    labels = sorted({r["answer"] for r in all_rows})
    seen = {tuple(sorted(r.items())) for r in rows}
    fresh = [r for r in all_rows if tuple(sorted(r.items())) not in seen]
    rng = np.random.default_rng(1)
    if len(fresh) > 500:
        fresh = [fresh[i] for i in sorted(rng.choice(len(fresh), 500, replace=False))]
    return rows, fresh, labels


def compact(model, tok, task, rows, labels):
    """H [n, d] and a three-column logit table [label0, label1, log-sum-exp of every other token], so that the
    log-sum-exp over its columns equals the model's and fit_edit / apply work unchanged with label columns 0, 1."""
    template = (ROOT / "tests/fixtures/legalbench_prompts" / f"{task}.txt").read_text()
    ft = [tok(" " + lab, add_special_tokens=False)["input_ids"][0] for lab in labels]
    H, LOG, _ = collect(model, tok, [EV.fill_prompt(template, r) for r in rows])
    lse = np.logaddexp.reduce(LOG, axis=1)
    lab = LOG[:, ft]
    rest = lse + np.log(np.clip(1.0 - np.exp(lab[:, 0] - lse) - np.exp(lab[:, 1] - lse), 1e-300, None))
    return H, np.column_stack([lab, rest])


def acc(pred, gold):
    return float((pred == gold).mean())


def scores(H, L, gold, Hf, Lf, goldf, lam=None):
    """Fit on (H, L, gold); score on the never-seen (Hf, Lf, goldf): as is, calibrated, read-out alone, edited."""
    e = fit_edit(H, L, gold, FT, lam=lam)
    n = len(gold)
    mu_hat = H.mean(0); mu_hat /= np.linalg.norm(mu_hat)
    lp = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
    m = lp[:, FT].mean(0); shift = -(m - m.mean()) / (H @ mu_hat).mean()
    cal = (Lf[:, 1] - Lf[:, 0]) + (shift[1] - shift[0]) * (Hf @ mu_hat)
    w, b = ridge(H, 2.0 * gold - 1.0, lam=float(n) if lam is None else lam)
    Le = apply(Lf, Hf, e)
    return dict(shipped=acc((Lf[:, 1] > Lf[:, 0]).astype(int), goldf), calibrated=acc((cal > 0).astype(int), goldf),
                readout=acc((Hf @ w + b > 0).astype(int), goldf), edited=acc((Le[:, 1] > Le[:, 0]).astype(int), goldf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--init-seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung3_controls.csv"))
    a = ap.parse_args()
    t0 = time.time()
    I = {t: items(t) for t in TASKS}
    model, tok = EV.load_trained(a.model)
    D = {}
    for task, (rows, fresh, labels) in I.items():
        H, L = compact(model, tok, task, rows, labels); Hf, Lf = compact(model, tok, task, fresh, labels)
        D[task] = dict(H=H, L=L, gold=np.array([labels.index(r["answer"]) for r in rows]), Hf=Hf, Lf=Lf,
                       goldf=np.array([labels.index(r["answer"]) for r in fresh]))
        print(f"collected {task}: {len(rows)} fit, {len(fresh)} never seen ({time.time() - t0:.0f}s)", flush=True)
    del model; torch.cuda.empty_cache()

    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3X", model=a.model, family="qwen3_5", src_dtype="BF16", layer="ALL")

        def put(test, task, variant, value, n, maj, lo=None, hi=None, seed=0):
            w.row(metric=f"{test}_{task}", component=variant, value=value, ks=maj, null_n=n, seed=seed,
                  null_kind="minmax" if lo is not None else None, null_mean=lo, null_sd=hi, **base)

        for task in TASKS:
            T = D[task]; rows, fresh, labels = I[task]
            H, L, gold, Hf, Lf, goldf = T["H"], T["L"], T["gold"], T["Hf"], T["Lf"], T["goldf"]
            n, nf = len(gold), len(goldf); maj = float(max(goldf.mean(), 1 - goldf.mean()))
            ref = scores(H, L, gold, Hf, Lf, goldf)
            print(f"\n[{task}] never-seen n={nf} majority={maj:.3f} | trained model: as shipped={ref['shipped']:.3f} "
                  f"calibrated={ref['calibrated']:.3f} read-out alone={ref['readout']:.3f} edited={ref['edited']:.3f}", flush=True)
            for k, v in ref.items():
                put("reference", task, k, v, nf, maj)

            # 1. shuffled labels
            rng = np.random.default_rng(100)
            S = [scores(H, L, rng.permutation(gold), Hf, Lf, goldf) for _ in range(REPEATS)]
            for k in ("readout", "edited"):
                v = np.array([s[k] for s in S])
                put("shuffled", task, k, float(v.mean()), nf, maj, float(v.min()), float(v.max()))
                print(f"  1 shuffled labels ({REPEATS} permutations): {k:8s} mean={v.mean():.3f} min={v.min():.3f} max={v.max():.3f}", flush=True)

            # 2. word-count baseline
            vocab = {}
            for r in rows:
                for t in set(words(r["text"])):
                    vocab[t] = vocab.get(t, 0) + 1
            vocab = {t: i for i, t in enumerate(sorted(t for t, c in vocab.items() if c >= 2))}

            def counts(rs):
                X = np.zeros((len(rs), len(vocab)))
                for i, r in enumerate(rs):
                    for t in words(r["text"]):
                        j = vocab.get(t)
                        if j is not None:
                            X[i, j] += 1
                return X
            X, Xf = counts(rows), counts(fresh)
            grid = {}
            for mlt in LAM_GRID:
                wv, b = ridge(X, 2.0 * gold - 1.0, lam=mlt * n)
                grid[mlt] = acc((Xf @ wv + b > 0).astype(int), goldf)
            best = max(grid, key=grid.get)
            put("wordcount", task, "same_formula", grid[1.0], nf, maj)
            put("wordcount", task, f"best_lambda_x{best:g}", grid[best], nf, maj)
            print(f"  2 word counts ({len(vocab)} words in >= 2 fit items): same formula={grid[1.0]:.3f}; best lambda of grid "
                  f"(picked on the never-seen items) x{best:g}={grid[best]:.3f} | model state: read-out alone={ref['readout']:.3f} edited={ref['edited']:.3f}", flush=True)

            # 3. fewer labels
            rng = np.random.default_rng(200)
            for k in SIZES:
                reps = 1 if k == n else REPEATS
                v = []
                for _ in range(reps):
                    idx = np.arange(n) if k == n else rng.choice(n, k, replace=False)
                    v.append(scores(H[idx], L[idx], gold[idx], Hf, Lf, goldf)["edited"])
                v = np.array(v)
                put("fewer", task, f"n{k}", float(v.mean()), nf, maj, float(v.min()), float(v.max()))
                print(f"  3 fitted on {k:3d} items ({reps:2d} subsets): edited mean={v.mean():.3f} min={v.min():.3f} max={v.max():.3f}", flush=True)

            # 4. near-copies removed
            e = fit_edit(H, L, gold, FT); Le = apply(Lf, Hf, e)
            pe = (Le[:, 1] > Le[:, 0]).astype(int); p0 = (Lf[:, 1] > Lf[:, 0]).astype(int)
            ws = [set(words(r["text"])) for r in rows]
            sim = np.array([max(len(s & t) / max(1, len(s | t)) for t in ws) for s in (set(words(r["text"])) for r in fresh)])
            for th in THRESHOLDS:
                keep = sim < th
                put("nearcopy", task, f"jaccard_lt_{th}", acc(pe[keep], goldf[keep]), int(keep.sum()), float(max(goldf[keep].mean(), 1 - goldf[keep].mean())))
                print(f"  4 never-seen items with no fit item at Jaccard >= {th}: kept {int(keep.sum()):3d} of {nf}; as shipped={acc(p0[keep], goldf[keep]):.3f} "
                      f"edited={acc(pe[keep], goldf[keep]):.3f}", flush=True)
            if "document_name" in rows[0]:
                docs = {r["document_name"] for r in rows}
                keep = np.array([r["document_name"] not in docs for r in fresh])
                if keep.any():
                    put("nearcopy", task, "other_contracts_only", acc(pe[keep], goldf[keep]), int(keep.sum()), float(max(goldf[keep].mean(), 1 - goldf[keep].mean())))
                print(f"  4 never-seen clauses from contracts with no clause in the fit sample: kept {int(keep.sum()):3d} of {nf}; "
                      + (f"as shipped={acc(p0[keep], goldf[keep]):.3f} edited={acc(pe[keep], goldf[keep]):.3f}" if keep.any() else "none"), flush=True)

        # 5. untrained model, same architecture (A0: fresh init by the family's init rule)
        canon, cfg, fam = load_canonical(a.model)
        for seed in a.init_seeds:
            weights = generators.CONDITIONS["A0"](canon, cfg, fam, seed, model=a.model, report={})
            model, tok = EV.load_authored(a.model, weights)
            print(f"\n5 untrained model (fresh init, seed {seed}):", flush=True)
            for task, (rows, fresh, labels) in I.items():
                H, L = compact(model, tok, task, rows, labels); Hf, Lf = compact(model, tok, task, fresh, labels)
                T = D[task]; maj = float(max(T["goldf"].mean(), 1 - T["goldf"].mean()))
                s = scores(H, L, T["gold"], Hf, Lf, T["goldf"])
                for k, v in s.items():
                    put("untrained", task, k, v, len(T["goldf"]), maj, seed=seed)
                print(f"  {task:28s} as is={s['shipped']:.3f} calibrated={s['calibrated']:.3f} read-out alone={s['readout']:.3f} "
                      f"edited={s['edited']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
