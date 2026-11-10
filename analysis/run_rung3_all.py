"""The Rung 3 formula-written edit on every two-label LegalBench task (132 of 162; the other 30 have more than
two labels, which the two-row edit does not cover), with a word-count baseline beside each.

Per task, on the trained Qwen3.5-0.8B, prompts = the task's LegalBench base_prompt.txt:
  twofold  the Rung 3 sample (all items, or 500 drawn with seed 0) split by item parity; every item is scored by
           an edit fitted on the other half.
  fresh    tasks with more than 500 items only: edit fitted on the whole 500-item sample, scored on up to 500
           other items (drawn with seed 1) that the fit never saw.
  few15    the edit fitted on 15 labelled items drawn at random from all collected items of the task (20 draws,
           seed 300), scored on every other collected item; value = mean over draws, null_mean / null_sd = min /
           max (null_kind minmax). The shipped row of few15 is the model as is on all collected items.
Variants: shipped (model as is), calibrated (E1, uses no labels), edited (E1 + E2, the row edit: fit_edit / apply
of run_rung3_transfer.py, which is what a patched model computes), wordcount (the same standardised ridge,
lambda = n, on counts of the words of the item's own fields, words present in at least two fit items).
Rows: results/tables/rung3_all.csv (hypothesis R3A). metric = {protocol}_{task} (accuracy) and
{protocol}_bal_{task} (balanced accuracy = mean of the two per-label recalls); ks = majority rate; null_n = items
scored. info_{task} rows: n_total, truncated (prompts cut to the last 1024 tokens), multi_token_label.
Resumable: a task with a done_{task} row in the output file is skipped.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import evaluate as EV  # noqa: E402
from analysis.run_rung3 import collect  # noqa: E402
from analysis.run_rung3_transfer import fit_edit, apply  # noqa: E402

PROMPTS = ROOT / "tests/fixtures/legalbench_prompts"
FT = [0, 1]
MAX_LEN = 1024
CHUNK = 50
FEW = 15
REPEATS = 20


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def ridge_any(X, y, lam):
    """The standardised ridge of run_rung3.ridge; solved in the n x n form when there are more features than
    items (the same solution: (Z'Z + lam I)^-1 Z' = Z' (ZZ' + lam I)^-1)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    yc = y - y.mean()
    if Z.shape[1] <= Z.shape[0]:
        w_z = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ yc)
    else:
        w_z = Z.T @ np.linalg.solve(Z @ Z.T + lam * np.eye(Z.shape[0]), yc)
    w = w_z / sd
    return w, float(y.mean() - mu @ w)


def states(model, tok, template, rows, ft):
    """H [n, d]; logit table [label0, label1, log-sum-exp of every other token]; number of truncated prompts."""
    prompts = [EV.fill_prompt(template, r) for r in rows]
    Hs, Ls, cut = [], [], 0
    for i in range(0, len(prompts), CHUNK):
        ch = prompts[i:i + CHUNK]
        cut += sum(len(tok(p, add_special_tokens=False)["input_ids"]) > MAX_LEN for p in ch)
        H, LOG, _ = collect(model, tok, ch, max_len=MAX_LEN)
        lse = np.logaddexp.reduce(LOG, axis=1); lab = LOG[:, ft]
        rest = lse + np.log(np.clip(1.0 - np.exp(lab[:, 0] - lse) - np.exp(lab[:, 1] - lse), 1e-300, None))
        Hs.append(H); Ls.append(np.column_stack([lab, rest]))
    return np.concatenate(Hs), np.concatenate(Ls), cut


def model_preds(H, L, gold, He, Le):
    """Fit on (H, L, gold); predictions on (He, Le): shipped, calibrated, edited."""
    e = fit_edit(H, L, gold, FT)
    mu_hat = H.mean(0); mu_hat /= np.linalg.norm(mu_hat)
    lp = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
    m = lp[:, FT].mean(0); shift = -(m - m.mean()) / (H @ mu_hat).mean()
    cal = (Le[:, 1] - Le[:, 0]) + (shift[1] - shift[0]) * (He @ mu_hat)
    Led = apply(Le, He, e)
    return dict(shipped=(Le[:, 1] > Le[:, 0]).astype(int), calibrated=(cal > 0).astype(int),
                edited=(Led[:, 1] > Led[:, 0]).astype(int))


def word_preds(texts, gold, texts_e):
    seen = {}
    for t in texts:
        for x in set(words(t)):
            seen[x] = seen.get(x, 0) + 1
    vocab = {x: i for i, x in enumerate(sorted(x for x, c in seen.items() if c >= 2))}
    if not vocab or len(set(gold.tolist())) < 2:
        return np.full(len(texts_e), int(round(gold.mean())), dtype=int), len(vocab)

    def counts(ts):
        X = np.zeros((len(ts), len(vocab)))
        for i, t in enumerate(ts):
            for x in words(t):
                j = vocab.get(x)
                if j is not None:
                    X[i, j] += 1
        return X
    w, b = ridge_any(counts(texts), 2.0 * gold - 1.0, lam=float(len(gold)))
    return (counts(texts_e) @ w + b > 0).astype(int), len(vocab)


def acc(p, g):
    return float((p == g).mean())


def bal(p, g):
    r = [float((p[g == c] == c).mean()) for c in (0, 1) if (g == c).any()]
    return float(np.mean(r))


def done_tasks(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {r["metric"][5:] for r in csv.DictReader(fh) if r["metric"].startswith("done_")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--tasks", nargs="*", default=sorted(p.stem for p in PROMPTS.glob("*.txt")))
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung3_all.csv"))
    a = ap.parse_args()
    skip = done_tasks(Path(a.out))
    model, tok = EV.load_trained(a.model)
    t_start = time.time()
    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3A", model=a.model, family="qwen3_5", src_dtype="BF16", seed=0, layer="ALL")
        for k, task in enumerate(a.tasks):
            if task in skip:
                continue
            t0 = time.time()
            try:
                template = (PROMPTS / f"{task}.txt").read_text()
                fields = [f for f in dict.fromkeys(re.findall(r"\{\{(\w+)\}\}", template))]
                all_rows = EV.legalbench_items(task, 10 ** 9); rows = EV.legalbench_items(task, 500)
                labels = sorted({r["answer"] for r in all_rows})
                if len(labels) != 2:
                    print(f"[{task}] {len(labels)} labels, skipped", flush=True); continue
                toks = [tok(" " + lab, add_special_tokens=False)["input_ids"] for lab in labels]
                ft = [t[0] for t in toks]
                if ft[0] == ft[1]:
                    print(f"[{task}] labels share their first token, skipped", flush=True); continue
                seen = {tuple(sorted(r.items())) for r in rows}
                fresh = [r for r in all_rows if tuple(sorted(r.items())) not in seen]
                if len(fresh) > 500:
                    rng = np.random.default_rng(1)
                    fresh = [fresh[i] for i in sorted(rng.choice(len(fresh), 500, replace=False))]
                text = lambda r: " ".join(str(r.get(f, "")) for f in fields)
                gold = np.array([labels.index(r["answer"]) for r in rows]); n = len(gold)
                H, L, cut = states(model, tok, template, rows, ft)
                tx = [text(r) for r in rows]
                out = []

                def put(protocol, variant, p, g):
                    maj = float(max(g.mean(), 1 - g.mean()))
                    w.row(metric=f"{protocol}_{task}", component=variant, value=acc(p, g), ks=maj, null_n=len(g), **base)
                    w.row(metric=f"{protocol}_bal_{task}", component=variant, value=bal(p, g), ks=0.5, null_n=len(g), **base)
                    return acc(p, g)

                # two-fold on the sample
                fold = np.arange(n) % 2
                P = {v: np.zeros(n, int) for v in ("shipped", "calibrated", "edited", "wordcount")}
                for f in (0, 1):
                    fi, ev = fold != f, fold == f
                    for v, p in model_preds(H[fi], L[fi], gold[fi], H[ev], L[ev]).items():
                        P[v][ev] = p
                    P["wordcount"][ev], _ = word_preds([t for t, m in zip(tx, fi) if m], gold[fi], [t for t, m in zip(tx, ev) if m])
                maj = float(max(gold.mean(), 1 - gold.mean()))
                s = {v: put("twofold", v, p, gold) for v, p in P.items()}
                line = (f"twofold n={n} maj={maj:.3f} shipped={s['shipped']:.3f} cal={s['calibrated']:.3f} "
                        f"edited={s['edited']:.3f} words={s['wordcount']:.3f}")
                # never-seen items
                if fresh:
                    gf = np.array([labels.index(r["answer"]) for r in fresh])
                    Hf, Lf, cutf = states(model, tok, template, fresh, ft); cut += cutf
                    Pf = model_preds(H, L, gold, Hf, Lf)
                    Pf["wordcount"], _ = word_preds(tx, gold, [text(r) for r in fresh])
                    sf = {v: put("fresh", v, p, gf) for v, p in Pf.items()}
                    line += (f" | fresh n={len(gf)} maj={max(gf.mean(), 1 - gf.mean()):.3f} shipped={sf['shipped']:.3f} "
                             f"cal={sf['calibrated']:.3f} edited={sf['edited']:.3f} words={sf['wordcount']:.3f}")
                # 15 labelled examples: fit on a random draw of FEW items, score every other collected item
                Hp, Lp, gp, tp = H, L, gold, tx
                if fresh:
                    Hp, Lp, gp = np.concatenate([H, Hf]), np.concatenate([L, Lf]), np.concatenate([gold, gf])
                    tp = tx + [text(r) for r in fresh]
                majp = float(max(gp.mean(), 1 - gp.mean())); ns = len(gp) - FEW
                rng = np.random.default_rng(300)
                R = {v: [] for v in ("calibrated", "edited", "wordcount")}; RB = {v: [] for v in R}
                for _ in range(REPEATS):
                    m = np.zeros(len(gp), bool); m[rng.choice(len(gp), FEW, replace=False)] = True
                    Pd = model_preds(Hp[m], Lp[m], gp[m], Hp[~m], Lp[~m])
                    Pd["wordcount"], _ = word_preds([t for t, q in zip(tp, m) if q], gp[m], [t for t, q in zip(tp, m) if not q])
                    for v in R:
                        R[v].append(acc(Pd[v], gp[~m])); RB[v].append(bal(Pd[v], gp[~m]))
                p_ship = (Lp[:, 1] > Lp[:, 0]).astype(int)
                w.row(metric=f"few15_{task}", component="shipped", value=acc(p_ship, gp), ks=majp, null_n=len(gp), **base)
                w.row(metric=f"few15_bal_{task}", component="shipped", value=bal(p_ship, gp), ks=0.5, null_n=len(gp), **base)
                for v in R:
                    w.row(metric=f"few15_{task}", component=v, value=float(np.mean(R[v])), null_kind="minmax", null_mean=float(np.min(R[v])),
                          null_sd=float(np.max(R[v])), ks=majp, null_n=ns, **base)
                    w.row(metric=f"few15_bal_{task}", component=v, value=float(np.mean(RB[v])), null_kind="minmax", null_mean=float(np.min(RB[v])),
                          null_sd=float(np.max(RB[v])), ks=0.5, null_n=ns, **base)
                line = (f"few15 scored={ns} maj={majp:.3f} shipped={acc(p_ship, gp):.3f} cal={np.mean(R['calibrated']):.3f} "
                        f"edited={np.mean(R['edited']):.3f} [{np.min(R['edited']):.3f}..{np.max(R['edited']):.3f}] words={np.mean(R['wordcount']):.3f} || " + line)
                for name, val in [("n_total", len(all_rows)), ("truncated", cut), ("multi_token_label", int(any(len(t) > 1 for t in toks)))]:
                    w.row(metric=f"info_{task}", component=name, value=val, **base)
                w.row(metric=f"done_{task}", component="done", value=1, **base)
                print(f"[{k + 1}/{len(a.tasks)} {task}] {line} | cut={cut} | {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            except Exception:
                print(f"[{task}] FAILED\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
