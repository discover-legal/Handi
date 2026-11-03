"""Metrics M1-M4 for any weight set loaded into its stock Hugging Face class.

M1  cross-entropy (nats/token) on WikiText-103 test and TinyStories validation, bootstrap CI over
    1k-token chunks; reference lines: uniform ln V, unigram and bigram cross-entropy of models fit
    on a disjoint training sample with the same tokenizer.
M2  copy accuracy: 200 sequences of 50 random tokens repeated twice, accuracy on the second half.
M3  sampled text: 50k tokens at temperature 1; KL(sample bigrams || corpus bigrams) compared with
    unigram sampling; fraction of whole dictionary words.
M4  LegalBench: ten binary tasks (docs/decisions.md), zero-shot with the base prompt, scored by
    comparing the log-probability of the two label strings.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

from ..analysis import registry  # noqa: E402
from ..canon import from_canonical  # noqa: E402

LEGALBENCH_TASKS = ["hearsay", "overruling", "nys_judicial_ethics", "definition_classification",
                    "citation_prediction_classification", "proa", "diversity_1", "diversity_6",
                    "cuad_governing_law", "contract_nli_limited_use"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------

def load_trained(name: str):
    snap = str(registry.snapshot_dir(name))
    cls = Qwen3_5ForCausalLM if registry.family(name) == "qwen3_5" else AutoModelForCausalLM
    model = cls.from_pretrained(snap, dtype=torch.float32).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(snap)
    return model, tok


def load_authored(name: str, canon: dict[str, np.ndarray | torch.Tensor]):
    """A fresh instance of `name`'s architecture with the canonical weights `canon` loaded in."""
    snap = str(registry.snapshot_dir(name))
    fam = registry.family(name)
    cls = Qwen3_5ForCausalLM if fam == "qwen3_5" else AutoModelForCausalLM
    model = cls.from_pretrained(snap, dtype=torch.float32)   # architecture + config; weights overwritten below
    cfg = registry.load_config(name)
    ct = {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v).float() for k, v in canon.items()}
    kw = {}
    if fam == "gpt2" and not any(k.startswith("transformer.") for k in model.state_dict()):
        kw["prefix"] = ""
    sd = from_canonical(fam, ct, cfg, **kw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not re.search(r"\.(attn|attention)\.bias$|masked_bias|inv_freq|lm_head\.weight$", m)]
    if missing or unexpected:
        raise RuntimeError(f"authored load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    tok = AutoTokenizer.from_pretrained(snap)
    return model.to(DEVICE).eval(), tok


# --------------------------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------------------------

_CORPUS_CACHE: dict[tuple, np.ndarray] = {}


def corpus_text(corpus: str, split: str) -> str:
    from datasets import load_dataset
    if corpus == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        return "".join(r["text"] for r in ds)
    if corpus == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split=split)
        return "\n\n".join(r["text"] for r in ds)
    raise ValueError(corpus)


def corpus_tokens(corpus: str, split: str, tok, max_chars: int) -> np.ndarray:
    key = (corpus, split, tok.name_or_path, max_chars)
    if key not in _CORPUS_CACHE:
        text = corpus_text(corpus, split)[:max_chars]
        ids = tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        _CORPUS_CACHE[key] = np.asarray(ids, dtype=np.int64)
    return _CORPUS_CACHE[key]


# --------------------------------------------------------------------------------------------
# M1
# --------------------------------------------------------------------------------------------

@dataclass
class M1Result:
    corpus: str
    nats: float
    ci_lo: float
    ci_hi: float
    n_tokens: int
    uniform: float
    unigram: float
    bigram: float


@torch.no_grad()
def cross_entropy(model, ids: np.ndarray, ctx: int = 512, batch: int = 2, n_tokens: int = 100_000) -> tuple[float, float, float, int]:
    ids = ids[: n_tokens + 1]
    chunks = [ids[i:i + ctx + 1] for i in range(0, len(ids) - ctx - 1, ctx)]
    losses = []
    for b in range(0, len(chunks), batch):
        x = torch.tensor(np.stack(chunks[b:b + batch]), device=DEVICE)
        logits = model(x[:, :-1]).logits
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1),
                                                reduction="none").view(x.shape[0], -1)   # [B, ctx]
        del logits
        losses.append(nll.cpu().numpy())
    nll = np.concatenate(losses).reshape(-1)
    per_chunk = nll.reshape(-1, ctx).mean(1) if len(nll) % ctx == 0 else np.array([nll.mean()])
    rng = np.random.default_rng(0)
    boots = [rng.choice(per_chunk, len(per_chunk), replace=True).mean() for _ in range(1000)]
    return float(nll.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(len(nll))


def ngram_reference(train_ids: np.ndarray, test_ids: np.ndarray, V: int) -> tuple[float, float]:
    """Cross-entropy of add-k smoothed unigram and bigram models fit on train_ids, scored on test_ids."""
    uni = np.bincount(train_ids, minlength=V).astype(np.float64) + 0.5
    uni /= uni.sum()
    h_uni = float(-np.log(uni[test_ids[1:]]).mean())
    pairs = Counter(zip(train_ids[:-1].tolist(), train_ids[1:].tolist()))
    ctx_counts = np.bincount(train_ids[:-1], minlength=V).astype(np.float64)
    lam = 0.7   # interpolated bigram: lam * ML bigram + (1 - lam) * unigram
    nll = 0.0
    for a, b in zip(test_ids[:-1].tolist(), test_ids[1:].tolist()):
        p_bi = pairs.get((a, b), 0) / ctx_counts[a] if ctx_counts[a] > 0 else 0.0
        nll -= math.log(lam * p_bi + (1 - lam) * uni[b])
    return h_uni, nll / (len(test_ids) - 1)


def m1(model, tok, corpora=("wikitext", "tinystories"), n_tokens: int = 100_000) -> list[M1Result]:
    V = model.config.vocab_size
    out = []
    for c in corpora:
        test = corpus_tokens(c, "test" if c == "wikitext" else "validation", tok, max_chars=2_000_000)
        train = corpus_tokens(c, "train", tok, max_chars=8_000_000)
        nats, lo, hi, n = cross_entropy(model, test, n_tokens=n_tokens)
        h_uni, h_bi = ngram_reference(train, test[: n_tokens + 1], V)
        out.append(M1Result(c, nats, lo, hi, n, math.log(V), h_uni, h_bi))
    return out


# --------------------------------------------------------------------------------------------
# M2, M3
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def m2_copy(model, V: int | None = None, n_seq: int = 200, length: int = 50, seed: int = 0) -> float:
    V = V or model.config.vocab_size
    rng = np.random.default_rng(seed)
    x = rng.integers(0, V, size=(n_seq, length))
    x = np.concatenate([x, x], 1)
    correct = 0
    for b in range(0, n_seq, 20):
        t = torch.tensor(x[b:b + 20], device=DEVICE)
        pred = model(t[:, :-1]).logits.argmax(-1)[:, length - 1:]      # predictions for positions length..2*length-1
        correct += int((pred == t[:, length:]).sum())
    return correct / (n_seq * length)


@torch.no_grad()
def m3_sample(model, tok, corpora=("wikitext", "tinystories"), n_tokens: int = 50_000, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    V = model.config.vocab_size
    bos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
    ids = torch.full((50, 1), bos, device=DEVICE)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                         max_new_tokens=n_tokens // 50, pad_token_id=bos, eos_token_id=None)[:, 1:]
    sample = out.reshape(-1).cpu().numpy()

    def bigram_dist(a):
        return Counter(zip(a[:-1].tolist(), a[1:].tolist()))

    def kl_to(pb, ref):
        qb = bigram_dist(ref)
        nq, np_ = sum(qb.values()), sum(pb.values())
        eps = 0.5 / nq
        return sum(c / np_ * math.log((c / np_) / (qb.get(k, 0) / nq + eps)) for k, c in pb.items())

    res = {}
    pb = bigram_dist(sample)
    rng = np.random.default_rng(seed)
    for c in corpora:
        ref = corpus_tokens(c, "train", tok, max_chars=8_000_000)
        uni = np.bincount(ref, minlength=V).astype(np.float64)
        uni /= uni.sum()
        us = rng.choice(V, size=len(sample), p=uni)
        res[f"kl_bigram_{c}"] = kl_to(pb, ref)
        res[f"kl_bigram_{c}_unigram_sampling"] = kl_to(bigram_dist(us), ref)
    words_path = Path("/usr/share/dict/words")
    dictionary = set(w.strip().lower() for w in words_path.read_text().splitlines()) if words_path.exists() else set()
    text = tok.decode(sample.tolist())
    toks = re.findall(r"[A-Za-z]+", text)
    res["dict_word_fraction"] = (sum(w.lower() in dictionary for w in toks) / len(toks)) if (toks and dictionary) else float("nan")
    res["n_alpha_words"] = len(toks)
    res["preview"] = text[:300]
    return res


# --------------------------------------------------------------------------------------------
# M4 LegalBench
# --------------------------------------------------------------------------------------------

def legalbench_items(task: str, max_items: int = 500, seed: int = 0) -> list[dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("nguha/legalbench", f"data/{task}/test.tsv", repo_type="dataset")
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if len(rows) > max_items:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), max_items, replace=False))]
    return rows


def fill_prompt(template: str, row: dict) -> str:
    def rep(m):
        return row.get(m.group(1), "")
    return re.sub(r"\{\{(\w+)\}\}", rep, template)


@torch.no_grad()
def _label_logprob(model, tok, prompts: list[str], label: str, max_len: int = 1024) -> np.ndarray:
    """Sum log-prob of ` label` continuation after each prompt (left-truncated to max_len)."""
    lab = tok(" " + label, add_special_tokens=False)["input_ids"]
    out = []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"][-(max_len - len(lab)):] + lab
        x = torch.tensor([ids], device=DEVICE)
        lp = torch.log_softmax(model(x).logits[0].float(), -1)
        out.append(float(sum(lp[len(ids) - len(lab) - 1 + i, lab[i]] for i in range(len(lab)))))
    return np.asarray(out)


def m4_legalbench(model, tok, tasks=LEGALBENCH_TASKS, max_items: int = 500) -> dict[str, dict]:
    res = {}
    for task in tasks:
        template = (ROOT / "tests/fixtures/legalbench_prompts" / f"{task}.txt").read_text()
        rows = legalbench_items(task, max_items)
        labels = sorted({r["answer"] for r in rows})
        prompts = [fill_prompt(template, r) for r in rows]
        scores = np.stack([_label_logprob(model, tok, prompts, lab) for lab in labels], 1)
        pred = [labels[i] for i in scores.argmax(1)]
        gold = [r["answer"] for r in rows]
        acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
        majority = max(Counter(gold).values()) / len(gold)
        # 95% CI by normal approximation
        half = 1.96 * math.sqrt(acc * (1 - acc) / len(gold)) if len(gold) else float("nan")
        res[task] = {"accuracy": acc, "ci_lo": acc - half, "ci_hi": acc + half, "majority": majority,
                     "n": len(gold), "labels": labels}
    return res


# --------------------------------------------------------------------------------------------

def evaluate_all(model, tok, which=("m1", "m2", "m3", "m4"), m1_tokens: int = 100_000) -> dict:
    out = {}
    if "m1" in which:
        out["m1"] = [r.__dict__ for r in m1(model, tok, n_tokens=m1_tokens)]
    if "m2" in which:
        out["m2"] = m2_copy(model)
    if "m3" in which:
        out["m3"] = m3_sample(model, tok)
    if "m4" in which:
        out["m4"] = m4_legalbench(model, tok)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--which", nargs="*", default=["m1", "m2", "m3", "m4"])
    ap.add_argument("--m1-tokens", type=int, default=100_000)
    a = ap.parse_args()
    model, tok = load_trained(a.model)
    print(json.dumps(evaluate_all(model, tok, a.which, a.m1_tokens), indent=1, default=str))
