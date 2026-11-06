"""Corpus bigram statistics, the model's direct path (embedding -> final norm -> unembedding), and the
rank-d factorisation of the smoothed log-bigram matrix used by H17 and condition A8.

Counts come from the M1 WikiText-103 train tokens (first 8M characters, the model's own tokenizer) and are
cached under results/raw/bigram/<model>.npz. Everything here is closed-form: no gradient step anywhere.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..analysis import registry

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path(__file__).resolve().parents[2] / "results" / "raw" / "bigram"
LAMBDA = 0.7          # interpolated bigram weight, as evaluate.ngram_reference
ALPHA = 0.5           # add-alpha unigram
TRAIN_CHARS = 8_000_000
TEMPS = np.logspace(math.log10(0.05), math.log10(500.0), 33)


# ----------------------------------------------------------------------------- counts
def _tokens(model: str, split: str, max_chars: int) -> np.ndarray:
    from .evaluate import corpus_tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(registry.snapshot_dir(model)))
    return corpus_tokens("wikitext", split, tok, max_chars)


def counts(model: str, V: int) -> dict:
    """{'uni': [V] train unigram counts, 'rows','cols','vals': train bigram COO, 'calib': train tokens 100k-200k,
    'test': first 100k+1 test tokens}."""
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{model}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["V"]) == V:
            return {k: z[k] for k in ("uni", "rows", "cols", "vals", "calib", "test")}
    train = _tokens(model, "train", TRAIN_CHARS)
    test = _tokens(model, "test", 2_000_000)[:100_001]
    uni = np.bincount(train, minlength=V)[:V].astype(np.int64)
    pair = train[:-1].astype(np.int64) * V + train[1:]
    u, c = np.unique(pair, return_counts=True)
    out = dict(uni=uni, rows=(u // V).astype(np.int64), cols=(u % V).astype(np.int64), vals=c.astype(np.int64),
               calib=train[100_000:200_001], test=test)
    np.savez(f, V=V, **out)
    return out


def references(c: dict, V: int) -> dict:
    """Cross-entropy of uniform, add-alpha unigram and the interpolated bigram on the test pairs."""
    uni = c["uni"].astype(np.float64) + ALPHA
    p = uni / uni.sum()
    test = c["test"]
    a, b = test[:-1], test[1:]
    h_uni = float(-np.log(p[b]).mean())
    ctx = c["uni"].astype(np.float64)
    import scipy.sparse as sp
    M = sp.csr_matrix((c["vals"].astype(np.float64), (c["rows"], c["cols"])), shape=(V, V))
    cab = np.asarray(M[a, b]).ravel()
    pbi = np.where(ctx[a] > 0, cab / np.maximum(ctx[a], 1), 0.0)
    h_bi = float(-np.log(LAMBDA * pbi + (1 - LAMBDA) * p[b]).mean())
    return dict(uniform=math.log(V), unigram=h_uni, bigram=h_bi)


# ----------------------------------------------------------------------------- direct path
def final_norm(x: torch.Tensor, canon: dict, fam: str) -> torch.Tensor:
    g = torch.as_tensor(np.asarray(canon["ln_f.w"], dtype=np.float32), device=x.device)
    if fam == "qwen3_5":
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + g)
    b = torch.as_tensor(np.asarray(canon["ln_f.b"], dtype=np.float32), device=x.device)
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) * torch.rsqrt(var + 1e-5) * g + b


def unembedding(canon: dict) -> np.ndarray:
    return canon["unembed.w"] if "unembed.w" in canon else canon["embed.tok"]


@torch.no_grad()
def direct_path_ce(canon: dict, fam: str, pairs_calib: np.ndarray, pairs_test: np.ndarray, E=None, U=None,
                   temps=TEMPS, chunk: int = 1024) -> tuple[float, float]:
    """(best temperature on the calibration pairs, test cross-entropy at that temperature) of softmax(l_a / T)."""
    E = torch.as_tensor(np.asarray(canon["embed.tok"] if E is None else E, dtype=np.float32), device=DEVICE)
    U = torch.as_tensor(np.asarray(unembedding(canon) if U is None else U, dtype=np.float32), device=DEVICE)
    temps = torch.as_tensor(np.asarray(temps, dtype=np.float32), device=DEVICE)

    def ce_per_temp(tokens: np.ndarray) -> torch.Tensor:
        a, b = tokens[:-1], tokens[1:]
        ua, inv = np.unique(a, return_inverse=True)
        tot = torch.zeros(len(temps), dtype=torch.float64, device=DEVICE)
        bt = torch.as_tensor(b, device=DEVICE)
        inv_t = torch.as_tensor(inv, device=DEVICE)
        for s in range(0, len(ua), chunk):
            ids = torch.as_tensor(ua[s:s + chunk], device=DEVICE)
            h = final_norm(E[ids], canon, fam)
            logits = h @ U.T                                        # [c, V]
            sel = (inv_t >= s) & (inv_t < s + chunk)
            ra, tgt = inv_t[sel] - s, bt[sel]
            for i, T in enumerate(temps):
                lp = torch.log_softmax(logits / T, -1)
                tot[i] -= lp[ra, tgt].sum()
            del logits
        return tot / (len(tokens) - 1)

    cal = ce_per_temp(pairs_calib)
    i = int(cal.argmin())
    test = ce_per_temp(pairs_test)
    return float(temps[i]), float(test[i])


# ----------------------------------------------------------------------------- factorisation (A8)
class _LogBigramOperator:
    """T = 1 log(p)^T + S with S_ab = log(1 + lambda c_ab / ((1 - lambda) c_a p_b)) sparse; matvec on the GPU."""

    def __init__(self, c: dict, V: int):
        uni = c["uni"].astype(np.float64) + ALPHA
        self.logp = torch.as_tensor(np.log(uni / uni.sum()), dtype=torch.float32, device=DEVICE)
        ctx = np.maximum(c["uni"].astype(np.float64), 1.0)
        p = uni / uni.sum()
        s_vals = np.log1p(LAMBDA * c["vals"] / ((1 - LAMBDA) * ctx[c["rows"]] * p[c["cols"]]))
        idx = torch.as_tensor(np.stack([c["rows"], c["cols"]]), device=DEVICE)
        vals = torch.as_tensor(s_vals, dtype=torch.float32, device=DEVICE)
        self.S = torch.sparse_coo_tensor(idx, vals, (V, V)).coalesce().to_sparse_csr()
        self.St = torch.sparse_coo_tensor(idx.flip(0), vals, (V, V)).coalesce().to_sparse_csr()
        self.V = V

    def mm(self, X: torch.Tensor) -> torch.Tensor:        # T @ X
        return self.S @ X + torch.ones(self.V, 1, device=DEVICE) @ (self.logp[None, :] @ X)

    def tmm(self, X: torch.Tensor) -> torch.Tensor:       # T^T @ X
        return self.St @ X + self.logp[:, None] @ (torch.ones(1, self.V, device=DEVICE) @ X)


@torch.no_grad()
def factorise(c: dict, V: int, d: int, tied: bool, seed: int = 0, power: int = 10, over: int = 16):
    """Rank-d factors of the smoothed log-bigram target. Untied: T ~ E U^T (randomized SVD).
    Tied: (T + T^T)/2 ~ E diag(sign) E^T from its d eigenpairs of largest |lambda| (randomized, symmetric);
    the sign goes into the final-norm gain (decisions.md, A8 run note, decision of 2026-11-05).
    Returns (E [V, d], U [V, d] = E * sign for tied, fit ratio on a sampled block, sign [d] in {-1, +1})."""
    op = _LogBigramOperator(c, V)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    k = d + over
    Q = torch.randn(V, k, generator=gen, device=DEVICE)
    if tied:
        sym = lambda X: 0.5 * (op.mm(X) + op.tmm(X))
        Q, _ = torch.linalg.qr(sym(Q))
        for _ in range(power):
            Q, _ = torch.linalg.qr(sym(Q))
        B = Q.T @ sym(Q)                                 # k x k
        B = 0.5 * (B + B.T)
        lam, W = torch.linalg.eigh(B)
        order = torch.argsort(lam.abs(), descending=True)[:d]
        lam, W = lam[order], W[:, order]
        sign = torch.where(lam >= 0, 1.0, -1.0)
        E = (Q @ W) * lam.abs().sqrt()[None, :]
        U = E * sign[None, :]
    else:
        Q, _ = torch.linalg.qr(op.mm(Q))
        for _ in range(power):
            Q, _ = torch.linalg.qr(op.tmm(Q))
            Q, _ = torch.linalg.qr(op.mm(Q))
        B = op.tmm(Q).T                                  # k x V  (= Q^T T)
        Ub, s, Vh = torch.linalg.svd(B, full_matrices=False)
        E = (Q @ Ub[:, :d]) * s[:d].sqrt()[None, :]
        U = Vh[:d].T * s[:d].sqrt()[None, :]
        sign = torch.ones(d, device=DEVICE)
    # fit quality on a random 2048 x 2048 block (columns in chunks of 256: V x 256 dense at a time)
    ia = torch.randint(0, V, (2048,), generator=gen, device=DEVICE)
    ib = torch.randint(0, V, (2048,), generator=gen, device=DEVICE)
    blocks = []
    for s0 in range(0, 2048, 256):
        onehot = torch.zeros(V, 256, device=DEVICE); onehot[ib[s0:s0 + 256], torch.arange(256, device=DEVICE)] = 1
        blk = op.mm(onehot) if not tied else 0.5 * (op.mm(onehot) + op.tmm(onehot))
        blocks.append(blk[ia]); del onehot, blk
    T_blk = torch.cat(blocks, 1)                         # [2048, 2048]
    fit = T_blk - E[ia] @ U[ib].T
    ratio = float(1 - fit.pow(2).sum() / (T_blk - T_blk.mean()).pow(2).sum())
    return (E.cpu().numpy().astype(np.float32), U.cpu().numpy().astype(np.float32), ratio,
            sign.cpu().numpy().astype(np.float32))
