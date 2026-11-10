"""Does the Rung 3 formula-written edit (E1 calibration + E2 ridge readout, written into the label rows of
lm_head) hold up on examples it never saw, and what does it do to the other tasks?

Two tests, both exact offline computations on the trained Qwen3.5-0.8B (an edit of the label rows changes the
logit of label t by delta_t . h for the final-norm state h, which is what run_rung3.py verified against the live
model):
  1. Fresh items. For every task with more than 500 test items, the edit is fitted on the whole 500-item sample
     used in Rung 3 and scored on up to 500 OTHER items from the same task that were never seen.
  2. Cross-task effect. The label rows (" Yes" / " No" and the rest) are shared by every task that uses them.
     Each task's edit (fitted on all its sampled items) is applied and every other task is rescored: fraction of
     answers changed and accuracy before / after. Also the "all edits at once" case: every task's row changes added
     together (as one model would have to carry them).
Rows: results/tables/rung3_transfer.csv (hypothesis R3T).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import evaluate as EV  # noqa: E402
from analysis.run_rung3 import collect, ridge  # noqa: E402


def fit_edit(H, LOG, gold, first_tok, lam=None):
    """E1 + E2 on (H, LOG, gold). Returns {token_id: delta_row [d]} for the two label tokens."""
    n, d = H.shape
    mu_hat = H.mean(0); mu_hat /= np.linalg.norm(mu_hat)
    proj = H @ mu_hat
    lp = LOG - np.logaddexp.reduce(LOG, axis=1, keepdims=True)
    mean_lp = lp[:, first_tok].mean(0)
    delta = -(mean_lp - mean_lp.mean())
    shift = delta / proj.mean()
    gap0 = LOG[:, first_tok[1]] - LOG[:, first_tok[0]]
    gap1 = gap0 + (shift[1] - shift[0]) * proj
    y = 2.0 * gold - 1.0
    wv, b = ridge(H, y, lam=float(n) if lam is None else lam)
    alpha = 2.0 * gap1.std() / ((H @ wv + b).std() + 1e-9)
    rows = {}
    for j, t in enumerate(first_tok):
        sgn = [-1.0, 1.0][j]
        rows[t] = shift[j] * mu_hat + sgn * alpha * wv + sgn * alpha * b / proj.mean() * mu_hat
    return rows


def apply(LOG, H, rows):
    L = LOG.copy()
    for t, dr in rows.items():
        L[:, t] += H @ dr
    return L


def acc_of(LOG, first_tok, gold):
    return float(((LOG[:, first_tok[1]] > LOG[:, first_tok[0]]).astype(int) == gold).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--tasks", nargs="*", default=EV.LEGALBENCH_TASKS)
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung3_transfer.csv"))
    a = ap.parse_args()
    model, tok = EV.load_trained(a.model)
    t0 = time.time()
    D = {}
    for task in a.tasks:
        template = (ROOT / "tests/fixtures/legalbench_prompts" / f"{task}.txt").read_text()
        all_rows = EV.legalbench_items(task, 10 ** 9)
        rows = EV.legalbench_items(task, 500)
        labels = sorted({r["answer"] for r in all_rows})
        if len(labels) != 2:
            print(f"[{task}] {len(labels)} labels, skipped", flush=True); continue
        first_tok = [tok(" " + lab, add_special_tokens=False)["input_ids"][0] for lab in labels]
        seen = {tuple(sorted(r.items())) for r in rows}
        fresh = [r for r in all_rows if tuple(sorted(r.items())) not in seen]
        rng = np.random.default_rng(1)
        if len(fresh) > 500:
            fresh = [fresh[i] for i in sorted(rng.choice(len(fresh), 500, replace=False))]
        H, LOG, _ = collect(model, tok, [EV.fill_prompt(template, r) for r in rows])
        gold = np.array([labels.index(r["answer"]) for r in rows])
        Hf = LOGf = goldf = None
        if fresh:
            Hf, LOGf, _ = collect(model, tok, [EV.fill_prompt(template, r) for r in fresh])
            goldf = np.array([labels.index(r["answer"]) for r in fresh])
        D[task] = dict(labels=labels, ft=first_tok, H=H, LOG=LOG, gold=gold, Hf=Hf, LOGf=LOGf, goldf=goldf)
        print(f"collected {task}: {len(rows)} sampled, {len(fresh)} fresh of {len(all_rows)} total; labels {labels} ({time.time() - t0:.0f}s)", flush=True)

    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3T", model=a.model, family="qwen3_5", src_dtype="BF16", seed=0, layer="ALL")
        edits = {}
        print("\n1. Fresh items (edit fitted on the full 500-item sample, scored on items never seen):", flush=True)
        for task, T in D.items():
            edits[task] = fit_edit(T["H"], T["LOG"], T["gold"], T["ft"])
            if T["Hf"] is None:
                continue
            maj = max(T["goldf"].mean(), 1 - T["goldf"].mean())
            a0 = acc_of(T["LOGf"], T["ft"], T["goldf"])
            a1 = acc_of(apply(T["LOGf"], T["Hf"], edits[task]), T["ft"], T["goldf"])
            n = len(T["goldf"])
            w.row(metric=f"fresh_{task}", component="E0", value=a0, ks=maj, null_n=n, **base)
            w.row(metric=f"fresh_{task}", component="E2", value=a1, ks=maj, null_n=n, **base)
            print(f"  {task:36s} n={n:3d} majority={maj:.3f} as shipped={a0:.3f} edited={a1:.3f}", flush=True)

        print("\n2. Cross-task effect of each task's edit (rows: edit fitted on; columns: task scored):", flush=True)
        names = list(D)
        print("  " + " " * 36 + " ".join(f"{t[:10]:>10s}" for t in names), flush=True)
        for src in names:
            line = []
            for dst in names:
                T = D[dst]
                L0, L1 = T["LOG"], apply(T["LOG"], T["H"], edits[src])
                p0 = (L0[:, T["ft"][1]] > L0[:, T["ft"][0]]); p1 = (L1[:, T["ft"][1]] > L1[:, T["ft"][0]])
                changed = float((p0 != p1).mean()); a0 = float((p0 == T["gold"]).mean()); a1 = float((p1 == T["gold"]).mean())
                w.row(metric=f"cross_changed_{dst}", component=f"edit:{src}", value=changed, ks=a1 - a0, null_n=len(T["gold"]), **base)
                shared = bool(set(D[src]["ft"]) & set(T["ft"]))
                line.append(f"{changed:5.2f}{'*' if src == dst else ('' if shared else ' ')}" .rjust(10))
            print(f"  {src:36s}" + " ".join(line), flush=True)
        print("  (fraction of answers changed; * = the task the edit was fitted on, in-sample; blank-marked = no shared label token)", flush=True)

        print("\n3. All ten edits applied at once (one model carrying every task's row changes):", flush=True)
        combined = {}
        for src in names:
            for t, dr in edits[src].items():
                combined[t] = combined.get(t, 0) + dr
        for dst in names:
            T = D[dst]
            a0 = acc_of(T["LOG"], T["ft"], T["gold"]); a1 = acc_of(apply(T["LOG"], T["H"], edits[dst]), T["ft"], T["gold"])
            ac = acc_of(apply(T["LOG"], T["H"], combined), T["ft"], T["gold"])
            w.row(metric=f"combined_{dst}", component="all_edits", value=ac, ks=a1, null_n=len(T["gold"]), **base)
            print(f"  {dst:36s} as shipped={a0:.3f} own edit (in-sample)={a1:.3f} all edits={ac:.3f}", flush=True)


if __name__ == "__main__":
    main()
