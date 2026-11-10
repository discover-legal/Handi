"""Rung 3 read-out for the LegalBench extraction tasks (owner, 2026-11-10: build the extraction read-out).

Tasks: definition_extraction and the three ssla tasks, whose answers are words copied from the input. This is a
read-out beside the model, not an edit of what the model writes: the model's own reply is unchanged.

Per item, on the trained Qwen3.5-0.8B, prompt = the task's LegalBench base_prompt.txt with the item's text, read
whole (no cut unless --max-len is given):
  states   the final state (the one the output table reads) at the last token of every word of the item's
           text; words are runs of letters and digits.
  labels   a word is +1 when it lies inside an occurrence of the answer in the text, -1 otherwise. The answer is
           looked for whole and as its comma-separated pieces (for the ssla tasks only pieces of two or more
           words, so that "Inc." alone marks nothing), matched without case and with any non-word characters
           between its words. ssla_plaintiff answers are written ['Name']; the brackets and quotes are dropped.
  read-out one standardised ridge (lambda = number of fit words), from the state to the word's label, fitted on
           all words of the labelled items, in two versions: weighted so the two kinds of word count equally,
           and unweighted (_plain). Both are reported because neither suits every task: answer words are rare
           (17 of 6,138 words on 20 ssla_plaintiff items); unweighted no score reached 0 there, so nothing was
           picked, and weighted too many words were picked. A word is picked when its score is above 0; runs of picked words next to each other
           are the extracted spans.
  answer   definition_extraction: the one span with the highest word score (the top word if none is picked).
           ssla: every span, repeats dropped, joined with ", "; none picked = "" (ssla_plaintiff: "Not named").
Variants (component):
  shipped    the model as is: its greedy reply to the prompt, up to 64 tokens, cut at the first line break.
  readout, readout_plain      the read-out above, weighted and unweighted.
  wordcount, wordcount_plain  the same ridge on counts instead of states: the word itself, the word before, the word after
             (lower case) and whether it starts with a capital, features present in at least two fit items.
  ceiling    the labels themselves turned into an answer by the same rule: the best any per-word pick can do.
Scores: LegalBench's own scorers (evaluation.py of HazyResearch/legalbench, copied below): definition_extraction
accuracy after Porter stemming; ssla F1 pooled over items.
Protocols, as in run_rung3_all.py: few{N} = fitted on N labelled items drawn at random (20 draws, seed 300 for 15
and 300 + N otherwise), scored on every other collected item, value = mean over draws, null_mean / null_sd =
min / max; twofold = the sample split by item parity, every item scored by a fit on the other half. The sample
is all items, or 500 drawn with seed 0.
Rows: results/tables/rung3_extract.csv (hypothesis R3X), metric = {protocol}_{task}, null_n = items scored.
info_{task} rows: n_total, n_sample, answer_found (share of items whose answer is found in the text, items
answered "Not named" left out), words_mean, prompt_tokens_max, max_len. Resumable on done_{task} rows.
"""
from __future__ import annotations

import argparse
import re
import string
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import evaluate as EV  # noqa: E402
from analysis.run_rung3_all import PROMPTS, done_tasks, ridge_any  # noqa: E402,F401

TASKS = ["definition_extraction", "ssla_company_defendants", "ssla_individual_defendants", "ssla_plaintiff"]
WORD = re.compile(r"\w+")
REPEATS = 20
FEW = 15
MAX_NEW = 64


# --- LegalBench scorers, copied from HazyResearch/legalbench evaluation.py ---------------------------------------
def normalize(text: str, stem: bool) -> str:
    from nltk.stem.porter import PorterStemmer
    text = str(text).translate(str.maketrans("", "", string.punctuation)).strip().lower()
    return PorterStemmer().stem(text) if stem else text


def evaluate_definition_extraction(generations, answers):
    total = correct = 0
    for g, a in zip(generations, answers):
        na = [normalize(x, stem=True) for x in a.split(",")]
        total += 1
        if any(normalize(x, stem=True) in na for x in g.split(",")):
            correct += 1
    return correct / total


def evaluate_ssla(generations, answers):
    tp = fp = fn = 0
    for g, a in zip(generations, answers):
        na = [normalize(x, stem=False) for x in a.split(",")]
        ng = [normalize(x, stem=False) for x in str(g).split(",")]
        for x in na:
            found = False
            for j in range(len(ng)):
                if x in ng[j]:
                    tp += 1; found = True
                    break
            if found:
                del ng[j]
            else:
                fn += 1
            fp += len(ng)          # as in LegalBench (inside the answer loop)
    return 2 * tp / (2 * tp + fp + fn)


def score(task, gens, answers):
    return evaluate_definition_extraction(gens, answers) if task == "definition_extraction" else evaluate_ssla(gens, answers)


# --- labels -----------------------------------------------------------------------------------------------------
def answer_pieces(task, ans):
    a = ans.strip()
    if task == "ssla_plaintiff":
        a = re.sub(r"^\[|\]$", "", a).replace("'", "")
    ps = [a] + [p.strip() for p in re.split(r"[,\n]", a)]
    keep = []
    for p in ps:
        n = len(WORD.findall(p))
        if n and (task == "definition_extraction" or n >= 2) and p.lower() != "not named":
            keep.append(p)
    return list(dict.fromkeys(keep))


def word_labels(task, text, ans, spans):
    y = -np.ones(len(spans))
    found = False
    for p in answer_pieces(task, ans):
        pat = r"\b" + r"\W+".join(map(re.escape, WORD.findall(p))) + r"\b"
        for m in re.finditer(pat, text, re.I):
            found = True
            for i, (s, e) in enumerate(spans):
                if s >= m.start() and e <= m.end():
                    y[i] = 1.0
    return y, found


# --- from word scores to an answer ------------------------------------------------------------------------------
def decode(task, text, spans, s):
    pick = np.flatnonzero(s > 0)
    runs = []
    for i in pick:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    if task == "definition_extraction":
        if not runs:
            runs = [[int(np.argmax(s))]] if len(s) else []
        if not runs:
            return ""
        r = max(runs, key=lambda r: s[r].max())
        return text[spans[r[0]][0]:spans[r[-1]][1]]
    out = list(dict.fromkeys(text[spans[r[0]][0]:spans[r[-1]][1]] for r in runs))
    if not out:
        return "Not named" if task == "ssla_plaintiff" else ""
    return ", ".join(out)


def balance(y):
    """Weights making the +1 and -1 words count equally (mean weight 1); all 1 when one kind is missing."""
    pos = y > 0; n, k = len(y), int(pos.sum())
    if k == 0 or k == n:
        return np.ones(n)
    return np.where(pos, n / (2.0 * k), n / (2.0 * (n - k)))


def wridge(X, y, lam, v):
    """The standardised ridge of run_rung3.ridge with item weights v: weighted means and sds, weighted squares."""
    sw = v.sum()
    mu = (v @ X) / sw
    sd = np.sqrt(np.maximum((v @ (X * X)) / sw - mu ** 2, 0)) + 1e-6
    Z = (X - mu) / sd
    ym = float(v @ y / sw)
    w_z = np.linalg.solve(Z.T @ (Z * v[:, None]) + lam * np.eye(Z.shape[1]), Z.T @ (v * (y - ym)))
    w = w_z / sd
    return w, ym - mu @ w


# --- word-count features ----------------------------------------------------------------------------------------
def word_feats(text, spans):
    ws = [text[s:e].lower() for s, e in spans]
    return [[f"w={w}", f"p={ws[i - 1] if i else '<s>'}", f"n={ws[i + 1] if i + 1 < len(ws) else '</s>'}"]
            + (["cap"] if text[spans[i][0]].isupper() else []) for i, w in enumerate(ws)]


def wordcount_scores(F_fit, item_fit, y_fit, F_eval, weighted=True):
    """Weighted standardised ridge (lambda = n words) on sparse 0/1 features, solved by lsqr with implicit
    centring: minimise sum_i v_i (z_i w - y_i + ym)^2 + lambda |w|^2 as rows scaled by sqrt(v_i)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, lsqr
    seen = {}
    for item_feats in item_fit:
        for f in item_feats:
            seen[f] = seen.get(f, 0) + 1
    vocab = {f: j for j, f in enumerate(sorted(f for f, c in seen.items() if c >= 2))}

    def mat(F):
        r, c = [], []
        for i, fs in enumerate(F):
            for f in fs:
                j = vocab.get(f)
                if j is not None:
                    r.append(i); c.append(j)
        return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(F), len(vocab)))
    X, Xe = mat(F_fit), mat(F_eval)
    n = X.shape[0]
    v = balance(y_fit) if weighted else np.ones(n); sw = v.sum(); r = np.sqrt(v)
    mu = (X.T @ v) / sw
    sd = np.sqrt(np.maximum((X.multiply(X)).T @ v / sw - mu ** 2, 0)) + 1e-6
    Z = LinearOperator((n, len(vocab)), matvec=lambda q: r * (X @ (q / sd) - mu @ (q / sd)),
                       rmatvec=lambda u: (X.T @ (r * u) - mu * (r * u).sum()) / sd, dtype=float)
    ym = float(v @ y_fit / sw)
    w_z = lsqr(Z, r * (y_fit - ym), damp=np.sqrt(n), atol=1e-8, btol=1e-8, iter_lim=5000)[0]
    w = w_z / sd
    return Xe @ w + (ym - mu @ w)


# --- model pass -------------------------------------------------------------------------------------------------
def collect_item(model, tok, prefix, text, suffix, max_len):
    """(states at the last token of each word [n_words, d], word spans in text, greedy reply, prompt tokens)."""
    import torch
    prompt = prefix + text + suffix
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], np.array(enc["offset_mapping"])
    n_tok = len(ids)
    drop = max(0, n_tok - max_len)
    spans = [(m.start(), m.end()) for m in WORD.finditer(text)]
    starts = offs[:, 0]
    pos = np.searchsorted(starts, [len(prefix) + e for _, e in spans], side="left") - 1
    keep = pos >= drop
    spans = [sp_ for sp_, k in zip(spans, keep) if k]
    pos = pos[keep] - drop
    with torch.no_grad():
        x = torch.tensor([ids[drop:]], device=EV.DEVICE)
        out = model.model(x, use_cache=True)
        hs = out.last_hidden_state[0]
        H = hs[torch.as_tensor(pos, device=hs.device)].float().cpu().numpy() if len(pos) else np.zeros((0, hs.shape[-1]))
        past, h, gen = out.past_key_values, hs[-1:], []
        for _ in range(MAX_NEW):
            t = int(model.lm_head(h)[-1].argmax())
            if t == tok.eos_token_id:
                break
            gen.append(t)
            if "\n" in tok.decode(gen):
                break
            o = model.model(torch.tensor([[t]], device=EV.DEVICE), past_key_values=past, use_cache=True)
            past, h = o.past_key_values, o.last_hidden_state[0]
    reply = tok.decode(gen).split("\n")[0].strip()
    return H.astype(np.float32), spans, reply, n_tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung3_extract.csv"))
    ap.add_argument("--sizes", nargs="*", type=int, default=[15, 30, 60, 120])
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--items", type=int, default=500, help="sample size (500 as in every Rung 3 run; smaller for a smoke test)")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    max_len = a.max_len if a.max_len > 0 else 10 ** 9
    if a.cache:
        Path(a.cache).mkdir(parents=True, exist_ok=True)
    skip = done_tasks(Path(a.out))
    model, tok = EV.load_trained(a.model)
    t_start = time.time()
    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3X", model=a.model, family="qwen3_5", src_dtype="BF16", seed=0, layer="ALL")
        for k, task in enumerate(a.tasks):
            if task in skip:
                continue
            t0 = time.time()
            try:
                template = (PROMPTS / f"{task}.txt").read_text()
                prefix, suffix = template.split("{{text}}")
                n_total = len(EV.legalbench_items(task, 10 ** 9))
                rows = EV.legalbench_items(task, a.items)
                answers = [r["answer"] for r in rows]
                cache = Path(a.cache) / f"{task}.npz" if a.cache else None
                if cache and cache.exists():
                    z = np.load(cache, allow_pickle=True)
                    Hs = list(np.split(z["H"], z["cuts"])); spans = list(z["spans"]); replies = list(z["replies"])
                    n_toks = z["n_toks"]
                else:
                    Hs, spans, replies, n_toks = [], [], [], []
                    for r in rows:
                        H, sp_, rep, nt = collect_item(model, tok, prefix, r["text"], suffix, max_len)
                        Hs.append(H); spans.append(sp_); replies.append(rep); n_toks.append(nt)
                    n_toks = np.array(n_toks)
                    if cache:
                        np.savez(cache, H=np.concatenate(Hs).astype(np.float16), cuts=np.cumsum([len(h) for h in Hs])[:-1],
                                 spans=np.array(spans, dtype=object), replies=np.array(replies), n_toks=n_toks)
                texts = [r["text"] for r in rows]
                lab = [word_labels(task, t, ans, s) for t, ans, s in zip(texts, answers, spans)]
                Y = [y for y, _ in lab]
                named = [i for i, ans in enumerate(answers) if "not named" not in ans.lower()]
                found = float(np.mean([lab[i][1] for i in named])) if named else float("nan")
                F = [word_feats(t, s) for t, s in zip(texts, spans)]
                n = len(rows)

                def run(fit, ev):
                    Hf = np.concatenate([Hs[i] for i in fit]).astype(np.float64); yf = np.concatenate([Y[i] for i in fit])
                    cuts = np.cumsum([len(Hs[i]) for i in ev])[:-1]
                    He = np.concatenate([Hs[i] for i in ev]).astype(np.float64)
                    ans_e = [answers[i] for i in ev]
                    res = dict(shipped=score(task, [replies[i] for i in ev], ans_e))
                    for tag, weighted in (("", True), ("_plain", False)):
                        W, B = wridge(Hf, yf, float(len(yf)), balance(yf) if weighted else np.ones(len(yf)))
                        s_read = np.split(He @ W + B, cuts)
                        s_word = np.split(wordcount_scores([f for i in fit for f in F[i]], [set(f for fs in F[i] for f in fs) for i in fit],
                                                           yf, [f for i in ev for f in F[i]], weighted), cuts)
                        res["readout" + tag] = score(task, [decode(task, texts[i], spans[i], s) for i, s in zip(ev, s_read)], ans_e)
                        res["wordcount" + tag] = score(task, [decode(task, texts[i], spans[i], s) for i, s in zip(ev, s_word)], ans_e)
                    res["ceiling"] = score(task, [decode(task, texts[i], spans[i], Y[i]) for i in ev], ans_e)
                    return res

                line = ""
                for size in a.sizes:
                    if n - size < 30 and size != FEW:
                        continue
                    rng = np.random.default_rng(300 if size == FEW else 300 + size)
                    R = {v: [] for v in ("shipped", "readout", "readout_plain", "wordcount", "wordcount_plain", "ceiling")}
                    for _ in range(REPEATS):
                        m = np.zeros(n, bool); m[rng.choice(n, size, replace=False)] = True
                        for v, x in run(np.flatnonzero(m), np.flatnonzero(~m)).items():
                            R[v].append(x)
                    for v, xs in R.items():
                        w.row(metric=f"few{size}_{task}", component=v, value=float(np.mean(xs)), null_kind="minmax",
                              null_mean=float(np.min(xs)), null_sd=float(np.max(xs)), null_n=n - size, **base)
                    line += (f"few{size} shipped={np.mean(R['shipped']):.3f} readout={np.mean(R['readout']):.3f} "
                             f"[{np.min(R['readout']):.3f}..{np.max(R['readout']):.3f}] plain={np.mean(R['readout_plain']):.3f} words={np.mean(R['wordcount']):.3f} "
                             f"words_plain={np.mean(R['wordcount_plain']):.3f} "
                             f"ceiling={np.mean(R['ceiling']):.3f} || ")
                idx = np.arange(n); halves = [idx[idx % 2 == 0], idx[idx % 2 == 1]]
                r0, r1 = run(halves[1], halves[0]), run(halves[0], halves[1])
                # twofold: the average of the two half scores, weighted by items scored
                for v in r0:
                    val = (r0[v] * len(halves[0]) + r1[v] * len(halves[1])) / n
                    w.row(metric=f"twofold_{task}", component=v, value=val, null_n=n, **base)
                line += "twofold " + " ".join(f"{v}={(r0[v] * len(halves[0]) + r1[v] * len(halves[1])) / n:.3f}" for v in r0)
                info = [("n_total", n_total), ("n_sample", n), ("answer_found", found),
                        ("words_mean", float(np.mean([len(s) for s in spans]))), ("prompt_tokens_max", int(np.max(n_toks))),
                        ("max_len", a.max_len)]
                for name, val in info:
                    w.row(metric=f"info_{task}", component=name, value=val, **base)
                w.row(metric=f"done_{task}", component="done", value=1, **base)
                print(f"[{k + 1}/{len(a.tasks)} {task}] {line} | found={found:.3f} tokens_max={int(np.max(n_toks))} | "
                      f"{time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            except Exception:
                print(f"[{task}] FAILED\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
