"""Rung 2 weight generators: each condition returns a full canonical weight dict for a model shape.

A0  fresh init by the family's init rule (control).
A1  spectrum-matched random: every 2-D matrix = U diag(sigma_trained) V^T with Haar U, V;
    token embeddings covariance-matched (N6) with random token assignment; 1-D vectors (norm gains,
    biases) keep their trained marginal under a random permutation (control; expected == A0).
A2  A1 + ordered-axis patterns (H10 smooth positions, H5 outlier dims, H4 frequency direction).
A3  A2 + cross-layer subspace alignment (H7) and QK/OV identity components (H3).
A4  A3 + explicit previous-token / induction circuits where H6 finds them.
A5  trained weights (upper bound); A5' / A5'' are produced by analysis/h02_lowrank_function.py.
Conditions A2-A4 take their parameters from the confirmed-pattern tables and are filled in only
after those hypotheses are evaluated (see docs/decisions.md for what was confirmed).
"""
from __future__ import annotations

import numpy as np
import torch

from ..analysis.nulls import n4_fresh_init, n5_spectrum_matched, n6_cov_matched

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _haar(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    g = torch.randn(m, k, generator=gen).to(DEVICE)
    q, r = torch.linalg.qr(g)
    return q * torch.sign(torch.diagonal(r))[None, :]


def spectrum_matched_gpu(W: np.ndarray, gen: torch.Generator) -> np.ndarray:
    """U diag(sigma_W) V^T with Haar U, V, computed on the GPU (Gram path for very tall matrices)."""
    T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).to(DEVICE)
    m, n = T.shape
    if max(m, n) / min(m, n) >= 16:
        G = (T.T @ T) if m >= n else (T @ T.T)
        s = torch.linalg.eigvalsh(G.double()).flip(0).clamp_min(0).sqrt().float()
    else:
        s = torch.linalg.svdvals(T)
    k = len(s)
    out = (_haar(m, k, gen) * s[None, :]) @ _haar(n, k, gen).T
    return out.cpu().numpy()


def a0_fresh(canon: dict[str, np.ndarray], cfg: dict, fam: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    fresh = n4_fresh_init(fam, cfg, rng)
    out = dict(canon)
    for k in canon:
        if k in fresh and fresh[k].shape == canon[k].shape:
            out[k] = fresh[k].astype(np.float32)
        elif canon[k].ndim == 1:
            # norm gains / biases the init table does not cover: family init value
            out[k] = _init_vector(k, canon[k], fam)
        else:
            out[k] = (rng.standard_normal(canon[k].shape) * 0.02).astype(np.float32)
    return out


def _init_vector(name: str, v: np.ndarray, fam: str) -> np.ndarray:
    if name.endswith(".b"):
        return np.zeros_like(v)
    if fam == "qwen3_5":
        return np.zeros_like(v)            # zero-centered RMSNorm: weight 0 means gain 1
    return np.ones_like(v)                 # LayerNorm gain 1


def a1_spectrum(canon: dict[str, np.ndarray], cfg: dict, fam: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, W in canon.items():
        if k in ("embed.tok", "unembed.w"):
            out[k] = n6_cov_matched(W, rng).astype(np.float32)
        elif W.ndim == 2:
            out[k] = spectrum_matched_gpu(W, gen)
        elif W.ndim == 1:
            out[k] = rng.permutation(W).astype(np.float32)
        else:
            out[k] = rng.permutation(W.reshape(-1)).reshape(W.shape).astype(np.float32)
    if fam == "qwen3_5" and "unembed.w" not in canon:
        pass   # tied: the HF class ties lm_head to embed.tok after loading
    return out


CONDITIONS = {"A0": a0_fresh, "A1": a1_spectrum}
