"""The Rung 3 formula-written edit on every LegalBench task with a fixed set of answer labels, with a word-count
baseline beside each. 132 tasks have two labels; 21 have three to eight (the K-label form below). The other 9
have no fixed label set or labels the first-token edit cannot separate, and are not run (GAPS below).

The K-label form (owner, 2026-11-10: extend the formula). It is the two-label formula written per label, and for
K = 2 it is the same edit to the last bit (checked against fit_edit of run_rung3_transfer.py):
  calibration  U[t_k] += shift_k mu_hat, so every label has the same mean log-probability on the fit items
               (unchanged: it was already written per label).
  read-out     one standardised ridge per label, from h to +1 for that label's items and -1 for the others
               (lambda = n); U[t_k] += alpha w_k + (alpha b_k / p_mean) mu_hat. For two labels w_No = -w_Yes.
  alpha        2 x (spread of the calibrated gap) / (spread of the read-out), both taken as the root mean square
               over label pairs of sd(score_k - score_j), the read-out's halved (for two labels: 2 sd(gap) / sd(r)).
The answer is the label whose first token has the highest score. Only the K label rows change.

Per task, on the trained Qwen3.5-0.8B, prompts = the task's LegalBench base_prompt.txt:
  twofold  the Rung 3 sample (all items, or 500 drawn with seed 0) split by item parity; every item is scored by
           an edit fitted on the other half.
  fresh    tasks with more than 500 items only: edit fitted on the whole 500-item sample, scored on up to 500
           other items (drawn with seed 1) that the fit never saw.
  few15    the edit fitted on 15 labelled items drawn at random from all collected items of the task (20 draws,
           seed 300), scored on every other collected item; value = mean over draws, null_mean / null_sd = min /
           max (null_kind minmax). The shipped row of few15 is the model as is on all collected items.
Variants: shipped (model as is, the answer restricted to the label tokens), calibrated (uses no labels), edited
(calibration + read-out: what a patched model computes), wordcount (the same per-label standardised ridge,
lambda = n, on counts of the words of the item's own fields, words present in at least two fit items).
Rows: results/tables/rung3_all.csv (hypothesis R3A). metric = {protocol}_{task} (accuracy) and
{protocol}_bal_{task} (balanced accuracy = mean of the per-label recalls over the labels present in the scored
items); ks = majority rate (accuracy rows) or 1 / K (balanced rows); null_n = items scored. info_{task} rows:
n_total, truncated (prompts cut to the last 1024 tokens), multi_token_label, and for tasks with more than two
labels n_labels and space_token (1 when the labels are bare digits, which this tokenizer writes as a space token
followed by the digit: the prompt then ends with that space and the digit tokens are the label rows).
Resumable: a task with a done_{task} row in the output file is skipped.

Options (owner, 2026-11-10: more labelled examples for the many-label tasks; do not cut long prompts):
  --sizes N ...  the few-label protocol at each N (default 15): metric few{N}_{task}, same 20 draws rule, seed
                 300 for 15 and 300 + N otherwise; a size is skipped when fewer than 30 items would be left to
                 score. few15 is unchanged by this option.
  --max-len N    prompts longer than N tokens keep their last N (default 1024, as in every earlier run); 0 = the
                 whole prompt is read. Recorded in the info_{task} row max_len.
  --cache DIR    saves the collected states and label scores per task (npz), so later analyses need no model.
  --hf MODEL_ID  another trained model (owner, 2026-11-10: try another model, a Nemotron or a Phi) loaded from
                 Hugging Face in bfloat16 instead of the registry's Qwen. Same prompts, same formula. Its layers
                 run on the GPU; its token table (input embedding and output table) stays on the CPU, because a
                 4B-class model does not fit an 8 GB card whole. Label scores are the output table times the
                 final state, computed on the CPU in float32.
  --no-fresh     skips the never-seen items of the large tasks (and so their fresh rows): the few-label draws
                 then come from the 500-item sample only.
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
from analysis.run_rung3 import collect, ridge  # noqa: E402

PROMPTS = ROOT / "tests/fixtures/legalbench_prompts"
MAX_LABELS = 12
GAPS = {  # not run: the row edit needs a fixed set of labels with distinct first tokens
    "citation_prediction_open": "open answer (a case name)", "definition_extraction": "open answer (a term from the text)",
    "rule_qa": "open answer (free text)", "sara_numeric": "open answer (a dollar amount)",
    "ssla_company_defendants": "open answer (names)", "ssla_individual_defendants": "open answer (names)",
    "ssla_plaintiff": "open answer (names)", "successor_liability": "answers are sets of several labels",
    "unfair_tos": "two of nine labels (Unilateral change / Unilateral termination) share their first token",
}
MIN_SCORED = 30
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


def load_hf(model_id):
    """(model, tok, U): layers on the GPU, token table on the CPU, U = output table in float32 on the CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
    if getattr(model.lm_head, "bias", None) is not None:
        raise RuntimeError("the output table has a bias; the row edit as written does not cover it")
    body = model.model; emb = body.get_input_embeddings()
    for _, child in body.named_children():
        if child is not emb:
            child.to("cuda")
    return model, tok, model.lm_head.weight.detach().float()


def make_collect_hf(U):
    import torch

    @torch.no_grad()
    def collect_hf(model, tok, prompts, max_len=1024):
        body = model.model; emb = body.get_input_embeddings()
        H, LOG = [], []
        for p in prompts:
            ids = tok(p, add_special_tokens=False)["input_ids"][-max_len:]
            x = emb(torch.tensor([ids])).to("cuda")
            h = body(inputs_embeds=x).last_hidden_state[0, -1].float().cpu()    # the state lm_head reads
            H.append(h); LOG.append(U @ h)
        return torch.stack(H).numpy().astype(np.float64), torch.stack(LOG).numpy().astype(np.float64), None
    return collect_hf


COLLECT = collect


def states(model, tok, template, rows, ft, max_len):
    """H [n, d]; logit table [one column per label, log-sum-exp of every other token]; number of truncated prompts."""
    prompts = [EV.fill_prompt(template, r) for r in rows]
    Hs, Ls, cut = [], [], 0
    for i in range(0, len(prompts), CHUNK):
        ch = prompts[i:i + CHUNK]
        cut += sum(len(tok(p, add_special_tokens=False)["input_ids"]) > max_len for p in ch)
        H, LOG, _ = COLLECT(model, tok, ch, max_len=max_len)
        lse = np.logaddexp.reduce(LOG, axis=1); lab = LOG[:, ft]
        rest = lse + np.log(np.clip(1.0 - np.exp(lab - lse[:, None]).sum(1), 1e-300, None))
        Hs.append(H); Ls.append(np.column_stack([lab, rest]))
    return np.concatenate(Hs), np.concatenate(Ls), cut


def fit_edit_k(H, L, gold, K, lam=None):
    """Calibration + read-out for K labels on (H, L, gold); L columns 0..K-1 are the label logits, column K the
    rest. Returns (D [K, d] = what is added to each label row, shift [K], mu_hat [d])."""
    n, d = H.shape
    mu_hat = H.mean(0); mu_hat /= np.linalg.norm(mu_hat)
    proj = H @ mu_hat; p_mean = proj.mean()
    lp = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
    mean_lp = lp[:, :K].mean(0)
    shift = -(mean_lp - mean_lp.mean()) / p_mean
    cal = L[:, :K] + proj[:, None] * shift[None, :]
    W, B = np.zeros((K, d)), np.zeros(K)
    for k in range(K):
        W[k], B[k] = ridge_any(H, 2.0 * (gold == k) - 1.0, lam=float(n) if lam is None else lam)
    R = H @ W.T + B
    pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
    s_cal = np.sqrt(np.mean([(cal[:, k] - cal[:, j]).std() ** 2 for j, k in pairs]))
    s_read = np.sqrt(np.mean([((R[:, k] - R[:, j]).std() / 2.0) ** 2 for j, k in pairs]))
    alpha = 2.0 * s_cal / (s_read + 1e-9)
    D = shift[:, None] * mu_hat[None, :] + alpha * W + (alpha * B / p_mean)[:, None] * mu_hat[None, :]
    return D, shift, mu_hat


def model_preds(H, L, gold, He, Le, K):
    """Fit on (H, L, gold); predictions on (He, Le): shipped, calibrated, edited."""
    D, shift, mu_hat = fit_edit_k(H, L, gold, K)
    cal = Le[:, :K] + (He @ mu_hat)[:, None] * shift[None, :]
    return dict(shipped=Le[:, :K].argmax(1), calibrated=cal.argmax(1), edited=(Le[:, :K] + He @ D.T).argmax(1))


def word_preds(texts, gold, texts_e, K):
    seen = {}
    for t in texts:
        for x in set(words(t)):
            seen[x] = seen.get(x, 0) + 1
    vocab = {x: i for i, x in enumerate(sorted(x for x, c in seen.items() if c >= 2))}
    if not vocab or len(set(gold.tolist())) < 2:
        return np.full(len(texts_e), int(np.bincount(gold, minlength=K).argmax()), dtype=int), len(vocab)

    def counts(ts):
        X = np.zeros((len(ts), len(vocab)))
        for i, t in enumerate(ts):
            for x in words(t):
                j = vocab.get(x)
                if j is not None:
                    X[i, j] += 1
        return X
    X, Xe = counts(texts), counts(texts_e)
    S = np.zeros((len(texts_e), K))
    for k in range(K):
        w, b = ridge_any(X, 2.0 * (gold == k) - 1.0, lam=float(len(gold)))
        S[:, k] = Xe @ w + b
    return S.argmax(1), len(vocab)


def acc(p, g):
    return float((p == g).mean())


def bal(p, g):
    r = [float((p[g == c] == c).mean()) for c in np.unique(g)]
    return float(np.mean(r))


def majority(g):
    return float(np.bincount(g).max() / len(g))


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
    ap.add_argument("--sizes", nargs="*", type=int, default=[FEW])
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--hf", default=None)
    ap.add_argument("--no-fresh", action="store_true")
    a = ap.parse_args()
    max_len = a.max_len if a.max_len > 0 else 10 ** 9
    if a.cache:
        Path(a.cache).mkdir(parents=True, exist_ok=True)
    skip = done_tasks(Path(a.out))
    global COLLECT
    if a.hf:
        model, tok, U = load_hf(a.hf); COLLECT = make_collect_hf(U)
    else:
        model, tok = EV.load_trained(a.model)
    t_start = time.time()
    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3A", model=a.model, family="qwen3_5", src_dtype="BF16", seed=0, layer="ALL")
        if a.hf:
            base.update(model=a.hf, family=model.config.model_type)
        for k, task in enumerate(a.tasks):
            if task in skip:
                continue
            t0 = time.time()
            try:
                template = (PROMPTS / f"{task}.txt").read_text()
                fields = [f for f in dict.fromkeys(re.findall(r"\{\{(\w+)\}\}", template))]
                all_rows = EV.legalbench_items(task, 10 ** 9); rows = EV.legalbench_items(task, 500)
                labels = sorted({r["answer"] for r in all_rows})
                K = len(labels)
                if task in GAPS or K > MAX_LABELS:
                    print(f"[{task}] not run: {GAPS.get(task, f'{K} distinct answers')}", flush=True); continue
                toks = [tok(" " + lab, add_special_tokens=False)["input_ids"] for lab in labels]
                ft = [t[0] for t in toks]; space_token = 0
                if len(set(ft)) == 1 and all(len(t) == 2 for t in toks) and tok.decode([ft[0]]) == " ":
                    template += " "; ft = [t[1] for t in toks]; space_token = 1    # 'Answer: 3' = ... ':', ' ', '3'
                if len(set(ft)) != K:
                    print(f"[{task}] labels share their first token, not run", flush=True); continue
                seen = {tuple(sorted(r.items())) for r in rows}
                fresh = [] if a.no_fresh else [r for r in all_rows if tuple(sorted(r.items())) not in seen]
                if len(fresh) > 500:
                    rng = np.random.default_rng(1)
                    fresh = [fresh[i] for i in sorted(rng.choice(len(fresh), 500, replace=False))]
                text = lambda r: " ".join(str(r.get(f, "")) for f in fields)
                gold = np.array([labels.index(r["answer"]) for r in rows]); n = len(gold)
                H, L, cut = states(model, tok, template, rows, ft, max_len)
                tx = [text(r) for r in rows]
                out = []

                def put(protocol, variant, p, g):
                    w.row(metric=f"{protocol}_{task}", component=variant, value=acc(p, g), ks=majority(g), null_n=len(g), **base)
                    w.row(metric=f"{protocol}_bal_{task}", component=variant, value=bal(p, g), ks=1.0 / K, null_n=len(g), **base)
                    return acc(p, g)

                # two-fold on the sample
                fold = np.arange(n) % 2
                P = {v: np.zeros(n, int) for v in ("shipped", "calibrated", "edited", "wordcount")}
                for f in (0, 1):
                    fi, ev = fold != f, fold == f
                    for v, p in model_preds(H[fi], L[fi], gold[fi], H[ev], L[ev], K).items():
                        P[v][ev] = p
                    P["wordcount"][ev], _ = word_preds([t for t, m in zip(tx, fi) if m], gold[fi], [t for t, m in zip(tx, ev) if m], K)
                maj = majority(gold)
                s = {v: put("twofold", v, p, gold) for v, p in P.items()}
                line = (f"twofold n={n} maj={maj:.3f} shipped={s['shipped']:.3f} cal={s['calibrated']:.3f} "
                        f"edited={s['edited']:.3f} words={s['wordcount']:.3f}")
                # never-seen items
                if fresh:
                    gf = np.array([labels.index(r["answer"]) for r in fresh])
                    Hf, Lf, cutf = states(model, tok, template, fresh, ft, max_len); cut += cutf
                    Pf = model_preds(H, L, gold, Hf, Lf, K)
                    Pf["wordcount"], _ = word_preds(tx, gold, [text(r) for r in fresh], K)
                    sf = {v: put("fresh", v, p, gf) for v, p in Pf.items()}
                    line += (f" | fresh n={len(gf)} maj={majority(gf):.3f} shipped={sf['shipped']:.3f} "
                             f"cal={sf['calibrated']:.3f} edited={sf['edited']:.3f} words={sf['wordcount']:.3f}")
                # 15 labelled examples: fit on a random draw of FEW items, score every other collected item
                Hp, Lp, gp, tp = H, L, gold, tx
                if fresh:
                    Hp, Lp, gp = np.concatenate([H, Hf]), np.concatenate([L, Lf]), np.concatenate([gold, gf])
                    tp = tx + [text(r) for r in fresh]
                majp = majority(gp)
                if a.cache:
                    np.savez(Path(a.cache) / f"{task}.npz", H=Hp, L=Lp, gold=gp, n_sample=n, labels=np.array(labels), texts=np.array(tp))
                p_ship = Lp[:, :K].argmax(1); few_line = ""
                for size in a.sizes:
                    ns = len(gp) - size
                    if size != FEW and ns < MIN_SCORED:
                        continue
                    rng = np.random.default_rng(300 if size == FEW else 300 + size)
                    R = {v: [] for v in ("calibrated", "edited", "wordcount")}; RB = {v: [] for v in R}
                    for _ in range(REPEATS):
                        m = np.zeros(len(gp), bool); m[rng.choice(len(gp), size, replace=False)] = True
                        Pd = model_preds(Hp[m], Lp[m], gp[m], Hp[~m], Lp[~m], K)
                        Pd["wordcount"], _ = word_preds([t for t, q in zip(tp, m) if q], gp[m], [t for t, q in zip(tp, m) if not q], K)
                        for v in R:
                            R[v].append(acc(Pd[v], gp[~m])); RB[v].append(bal(Pd[v], gp[~m]))
                    w.row(metric=f"few{size}_{task}", component="shipped", value=acc(p_ship, gp), ks=majp, null_n=len(gp), **base)
                    w.row(metric=f"few{size}_bal_{task}", component="shipped", value=bal(p_ship, gp), ks=1.0 / K, null_n=len(gp), **base)
                    for v in R:
                        w.row(metric=f"few{size}_{task}", component=v, value=float(np.mean(R[v])), null_kind="minmax", null_mean=float(np.min(R[v])),
                              null_sd=float(np.max(R[v])), ks=majp, null_n=ns, **base)
                        w.row(metric=f"few{size}_bal_{task}", component=v, value=float(np.mean(RB[v])), null_kind="minmax", null_mean=float(np.min(RB[v])),
                              null_sd=float(np.max(RB[v])), ks=1.0 / K, null_n=ns, **base)
                    few_line += (f"few{size} scored={ns} maj={majp:.3f} shipped={acc(p_ship, gp):.3f} cal={np.mean(R['calibrated']):.3f} "
                                 f"edited={np.mean(R['edited']):.3f} [{np.min(R['edited']):.3f}..{np.max(R['edited']):.3f}] words={np.mean(R['wordcount']):.3f} "
                                 f"bal: shipped={bal(p_ship, gp):.3f} edited={np.mean(RB['edited']):.3f} words={np.mean(RB['wordcount']):.3f} || ")
                line = few_line + line
                info = [("n_total", len(all_rows)), ("truncated", cut), ("multi_token_label", int(any(len(t) > 1 for t in toks))),
                        ("max_len", a.max_len)]
                if K > 2:
                    info += [("n_labels", K), ("space_token", space_token)]; line = f"K={K} " + line
                for name, val in info:
                    w.row(metric=f"info_{task}", component=name, value=val, **base)
                w.row(metric=f"done_{task}", component="done", value=1, **base)
                print(f"[{k + 1}/{len(a.tasks)} {task}] {line} | cut={cut} | {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            except Exception:
                print(f"[{task}] FAILED\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
