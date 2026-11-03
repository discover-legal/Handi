"""H2: is the model's function carried by the top-k singular directions (the spikes) or by the bulk?

For each matrix group (attn.qk = q and k projections; attn.vo = v and o; mlp = up/gate/down;
embed = token embeddings (+ unembed when untied)) and for every matrix in the group at once:
    keep_spikes : top-k singular triplets kept exactly, bulk replaced by Haar-random directions with
                  the bulk singular values (in the orthogonal complement of the kept directions)
    keep_bulk   : bulk kept exactly, top-k replaced by Haar-random directions with the top-k values
k = max(spike count above the Marchenko-Pastur edge, 1), capped at 20% of the directions.
Metric: M1 cross-entropy on WikiText-103 test and TinyStories validation (50k tokens each).
Null expectation on a fresh-init model: both ablations change nothing (loss already ~ ln V).

Usage: .venv/bin/python analysis/h02_lowrank_function.py --models gpt2 pythia-160m tinystories-33m qwen3.5-0.8b
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

from handi.analysis import registry, spectra  # noqa: E402
from handi.analysis.loader import load_canonical  # noqa: E402
from handi.analysis.nulls import n4_fresh_init  # noqa: E402
from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring.evaluate import cross_entropy, corpus_tokens, load_authored  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GROUPS = {
    "attn.qk": lambda n: n.endswith(("attn.q.w", "attn.k.w")),
    "attn.vo": lambda n: n.endswith(("attn.v.w", "attn.o.w")),
    "mlp": lambda n: ".mlp." in n and n.endswith(".w"),
    "embed": lambda n: n in ("embed.tok", "unembed.w"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def haar(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    g = torch.randn(m, k, generator=gen, device="cpu").to(DEVICE)
    q, r = torch.linalg.qr(g)
    return q * torch.sign(torch.diagonal(r))[None, :]


def split_and_rebuild(W: np.ndarray, mode: str, k: int, gen: torch.Generator) -> tuple[np.ndarray, int]:
    """Return (ablated W, k used). SVD on GPU in float32 (Gram path for very tall matrices)."""
    T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).to(DEVICE)
    m, n = T.shape
    tall = m >= n
    if max(m, n) / min(m, n) >= 16:
        G = (T.T @ T) if tall else (T @ T.T)
        ev, Vs = torch.linalg.eigh(G.double())
        ev, Vs = ev.flip(0).clamp_min(0), Vs.flip(1)
        s = ev.sqrt().float()
        small = Vs.float()                                   # [min, min]
        big = (T @ small if tall else T.T @ small) / s.clamp_min(1e-12)[None, :]   # [max, min]
        U, V = (big, small) if tall else (small, big)
    else:
        U, s, Vh = torch.linalg.svd(T, full_matrices=False)
        V = Vh.T
    r = len(s)
    k = max(1, min(k, r // 5))
    if mode == "keep_spikes":
        Uk, sk, Vk = U[:, :k], s[:k], V[:, :k]
        # Haar directions in the complement of the kept ones, carrying the bulk singular values
        Ub = haar(m, r - k, gen); Ub = Ub - Uk @ (Uk.T @ Ub); Ub, _ = torch.linalg.qr(Ub)
        Vb = haar(n, r - k, gen); Vb = Vb - Vk @ (Vk.T @ Vb); Vb, _ = torch.linalg.qr(Vb)
        W2 = Uk @ torch.diag(sk) @ Vk.T + Ub @ torch.diag(s[k:]) @ Vb.T
    elif mode == "keep_bulk":
        Ub, sb, Vb = U[:, k:], s[k:], V[:, k:]
        Uk = haar(m, k, gen); Uk = Uk - Ub @ (Ub.T @ Uk); Uk, _ = torch.linalg.qr(Uk)
        Vk = haar(n, k, gen); Vk = Vk - Vb @ (Vb.T @ Vk); Vk, _ = torch.linalg.qr(Vk)
        W2 = Uk @ torch.diag(s[:k]) @ Vk.T + Ub @ torch.diag(sb) @ Vb.T
    else:
        raise ValueError(mode)
    return W2.cpu().numpy(), k


def spike_k(W: np.ndarray) -> int:
    s = spectra.singular_values(W)
    eig = spectra.esd(W, s)
    gamma = spectra.aspect(W)
    return int(spectra.spike_count(eig, gamma))


def run_model(name: str, writer: ResultsWriter, seed: int, n_tokens: int, groups: list[str]) -> None:
    canon, cfg, fam = load_canonical(name)
    tok = AutoTokenizer.from_pretrained(str(registry.snapshot_dir(name)))
    tests = {c: corpus_tokens(c, "test" if c == "wikitext" else "validation", tok, 2_000_000) for c in ("wikitext", "tinystories")}
    V = cfg["vocab_size"]
    src_dtype = {"gpt2": "F32", "gpt_neo": "F32", "gpt_neox": "F16", "qwen3_5": "BF16"}[fam]

    def score(weights: dict, label: str) -> dict[str, float]:
        model, _ = load_authored(name, weights)
        out = {}
        for c, ids in tests.items():
            nats, lo, hi, n = cross_entropy(model, ids, n_tokens=n_tokens)
            out[c] = nats
            writer.row(hypothesis="H02", model=name, family=fam, layer="ALL", component=label, metric=f"ce_{c}",
                         value=nats, null_kind="", null_n=0, null_mean="", null_sd="", z="", ks="", ks_null_median="",
                         n_rows="", n_cols="", src_dtype=src_dtype, seed=seed)
        del model
        torch.cuda.empty_cache()
        print(f"  {name:16s} {label:28s} " + "  ".join(f"{c}={v:.3f}" for c, v in out.items()), flush=True)
        return out

    print(f"[{name}] trained baseline", flush=True)
    base = score(canon, "trained")
    fresh = n4_fresh_init(fam, cfg, np.random.default_rng(seed))
    fresh = {k: v for k, v in fresh.items() if k in canon}
    fresh_full = dict(canon); fresh_full.update(fresh)
    score(fresh_full, "fresh_init")
    for g in groups:
        names = [n for n in canon if GROUPS[g](n) and canon[n].ndim == 2]
        if not names:
            continue
        ks = {n: spike_k(canon[n]) for n in names}
        writer.row(hypothesis="H02", model=name, family=fam, layer="ALL", component=g, metric="median_spike_k",
                     value=float(np.median(list(ks.values()))), null_kind="", null_n=0, null_mean="", null_sd="",
                     z="", ks="", ks_null_median="", n_rows="", n_cols="", src_dtype=src_dtype, seed=seed)
        for mode in ("keep_spikes", "keep_bulk"):
            gen = torch.Generator().manual_seed(seed)
            w2 = dict(canon)
            used = []
            t0 = time.time()
            for n in names:
                w2[n], kk = split_and_rebuild(canon[n], mode, ks[n], gen)
                used.append(kk)
            print(f"  [{g}:{mode}] {len(names)} matrices, k median {np.median(used):.0f}, {time.time() - t0:.0f}s", flush=True)
            res = score(w2, f"{g}:{mode}")
            for c in res:
                writer.row(hypothesis="H02", model=name, family=fam, layer="ALL", component=f"{g}:{mode}",
                             metric=f"delta_ce_{c}_vs_trained", value=res[c] - base[c], null_kind="", null_n=0,
                             null_mean="", null_sd="", z="", ks="", ks_null_median="", n_rows="", n_cols="",
                             src_dtype=src_dtype, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gpt2", "pythia-160m", "tinystories-33m", "qwen3.5-0.8b"])
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=50_000)
    ap.add_argument("--out", default=str(ROOT / "results/tables/h02.csv"))
    a = ap.parse_args()
    with ResultsWriter(a.out) as w:
        for m in a.models:
            run_model(m, w, a.seed, a.tokens, a.groups)


if __name__ == "__main__":
    main()
