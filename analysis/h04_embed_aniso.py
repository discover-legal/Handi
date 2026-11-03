"""H4: token-embedding anisotropy with a dominant common direction tied to frequency.

Per model, on embed.tok (and unembed.w where untied):
  * mean pairwise cosine over 10^6 random token pairs;
  * PCA variance ratios of the top-10 PCs vs the MP edge of a V x d Gaussian
    (gamma = d/V, sigma2 = ||E||_F^2 / (V d)); the edge as a variance share is
    lambda_+ / trace = (1 + sqrt(gamma))^2 / d;
  * Spearman of the projection on PC1 (sign fixed to agree with the mean direction)
    and on the mean direction against token id and against corpus unigram counts
    (first 200k characters of WikiText-103 test, tokenized with the model's tokenizer).
Nulls: N1 (Gaussian, same shape and norm) and N6 (covariance-matched, random token
assignment), 20 draws (5 for Qwen). PCA always goes through the d x d covariance.

Usage: .venv/bin/python analysis/h04_embed_aniso.py [--models ...] [--nulls N] [--seed S] [--out CSV]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from handi.analysis import nulls, plots, registry, spectra  # noqa: E402
from handi.analysis.loader import load_canonical  # noqa: E402
from handi.analysis.results import ResultsWriter, save_raw  # noqa: E402
from handi.analysis.stats import ks_2samp, null_summary, zscore  # noqa: E402

HYP = "H04"
DEFAULT_MODELS = ["gpt2", "pythia-70m", "pythia-160m", "qwen3.5-0.8b",
                  "tinystories-1m", "tinystories-3m", "tinystories-8m", "tinystories-33m"]
SRC_DTYPE = {"gpt2": "F32", "pythia-70m": "F16", "pythia-160m": "F16", "qwen3.5-0.8b": "BF16",
             "tinystories-1m": "F32", "tinystories-3m": "F32", "tinystories-8m": "F32", "tinystories-33m": "F32"}
N_PAIRS = 1_000_000
CORPUS_CHARS = 200_000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- corpus
def unigram_counts(model: str, V: int) -> np.ndarray:
    from datasets import load_dataset
    from tokenizers import Tokenizer

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "".join(ds["text"])[:CORPUS_CHARS]
    tok = Tokenizer.from_file(str(registry.snapshot_dir(model) / "tokenizer.json"))
    ids = np.asarray(tok.encode(text).ids, dtype=np.int64)
    return np.bincount(ids, minlength=V)[:V]


# ----------------------------------------------------------------------------- stats
def _pairs(rng: np.random.Generator, V: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    i = rng.integers(0, V, size=n)
    j = rng.integers(0, V, size=n)
    keep = i != j
    return i[keep], j[keep]


CHUNK = 32_768


def embed_stats(E: np.ndarray, pairs, counts: np.ndarray | None) -> dict:
    """All H4 statistics for one V x d matrix. Returns scalars plus the cosine sample
    and the PC1 / mean projections (for figures). All GPU work is chunked over rows
    (the GPU is shared), float64 for the d x d covariance."""
    V, d = E.shape
    E32 = np.ascontiguousarray(E, dtype=np.float32)
    T = torch.from_numpy(E32).to(DEV)                       # V x d float32 (1 GB for Qwen)
    row_norm = T.norm(dim=1).clamp_min(1e-12)
    # mean pairwise cosine on the sample of pairs (chunked gather)
    i, j = pairs
    cos = np.empty(i.size, dtype=np.float32)
    for a in range(0, i.size, 200_000):
        ii = torch.from_numpy(i[a:a + 200_000]).to(DEV)
        jj = torch.from_numpy(j[a:a + 200_000]).to(DEV)
        c = (T[ii] * T[jj]).sum(1) / (row_norm[ii] * row_norm[jj])
        cos[a:a + 200_000] = c.cpu().numpy()
    # PCA through the d x d covariance (centered), float64 accumulated in chunks
    mu = torch.zeros(d, dtype=torch.float64, device=DEV)
    for a in range(0, V, CHUNK):
        mu += T[a:a + CHUNK].double().sum(0)
    mu /= V
    C = torch.zeros(d, d, dtype=torch.float64, device=DEV)
    fro2 = 0.0
    sq_rows = torch.empty(V, dtype=torch.float64, device=DEV)
    for a in range(0, V, CHUNK):
        X = T[a:a + CHUNK].double()
        fro2 += float((X ** 2).sum())
        sq_rows[a:a + CHUNK] = (X ** 2).sum(1)
        X = X - mu
        C += X.T @ X
    C /= (V - 1)
    evals, evecs = torch.linalg.eigh(C)
    evals = evals.flip(0).clamp_min(0.0)
    evecs = evecs.flip(1)
    share = (evals / evals.sum()).cpu().numpy()
    pc1 = evecs[:, 0]
    if torch.dot(pc1, mu) < 0:
        pc1 = -pc1
    mu_dir = mu / mu.norm().clamp_min(1e-12)
    proj_pc1 = np.empty(V, dtype=np.float64)
    proj_mean = np.empty(V, dtype=np.float64)
    for a in range(0, V, CHUNK):
        X = T[a:a + CHUNK].double()
        proj_pc1[a:a + CHUNK] = ((X - mu) @ pc1).cpu().numpy()
        proj_mean[a:a + CHUNK] = (X @ mu_dir).cpu().numpy()
    # MP edge for an iid V x d matrix with the same Frobenius norm
    sigma2 = fro2 / (V * d)
    gamma = min(V, d) / max(V, d)
    lam_minus, lam_plus = spectra.mp_edges(gamma, sigma2)
    mp_share = lam_plus / (d * sigma2)          # = (1 + sqrt(gamma))^2 / d
    tok_id = np.arange(V)
    out = {
        "mean_cos": float(cos.mean()),
        "cos_sd": float(cos.std()),
        "pc_share": share[:10],
        "mp_edge_share": float(mp_share),
        "pc1_over_mp": float(share[0] / mp_share),
        "spearman_pc1_tokid": float(sps.spearmanr(proj_pc1, tok_id).statistic),
        "spearman_mean_tokid": float(sps.spearmanr(proj_mean, tok_id).statistic),
        "mean_norm_over_rms": float(mu.norm() / torch.sqrt(sq_rows.mean())),
        "_cos": cos,
        "_proj_pc1": proj_pc1,
        "_proj_mean": proj_mean,
    }
    if counts is not None:
        m = counts > 0
        out["spearman_pc1_unigram"] = float(sps.spearmanr(proj_pc1[m], counts[m]).statistic)
        out["spearman_mean_unigram"] = float(sps.spearmanr(proj_mean[m], counts[m]).statistic)
        out["n_unigram"] = int(m.sum())
    del T, C, row_norm, sq_rows
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return out


SCALARS = ["mean_cos", "cos_sd", "mp_edge_share", "pc1_over_mp", "spearman_pc1_tokid",
           "spearman_mean_tokid", "mean_norm_over_rms", "spearman_pc1_unigram", "spearman_mean_unigram"]


# ----------------------------------------------------------------------------- figures
def figure(model: str, comp: str, tr: dict, n1: list[dict], n6: list[dict], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    lo = min(tr["_cos"].min(), n1[0]["_cos"].min(), n6[0]["_cos"].min())
    hi = max(tr["_cos"].max(), n1[0]["_cos"].max(), n6[0]["_cos"].max())
    bins = np.linspace(lo, hi, 120)
    ax.hist(tr["_cos"], bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=pal[0])
    ax.hist(tr["_cos"], bins=bins, density=True, histtype="step", color=pal[0], lw=2, label="trained")
    ax.hist(n1[0]["_cos"], bins=bins, density=True, histtype="step", color=pal[1], lw=2, label="N1 Gaussian (1 draw)")
    ax.hist(n6[0]["_cos"], bins=bins, density=True, histtype="step", color=pal[2], lw=2, label="N6 cov-matched (1 draw)")
    ax.axvline(0, color=plots.INK_SECONDARY, lw=1)
    ax.set_xlabel("pairwise cosine (10^6 random token pairs)")
    ax.set_ylabel("density")
    ax.set_title(f"{model} {comp}: mean cos = {tr['mean_cos']:.3f}")
    ax.legend()
    ax = axes[1]
    V = tr["_proj_pc1"].size
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(V, size=min(V, 6000), replace=False))
    ax.scatter(np.log10(idx + 1), tr["_proj_pc1"][idx], s=6, color=pal[0], alpha=0.45, linewidths=0, rasterized=True)
    ax.set_xlabel("log10(token id + 1)")
    ax.set_ylabel("projection on PC1 (centered)")
    ax.set_title(f"PC1 vs token id: Spearman rho = {tr['spearman_pc1_tokid']:.3f} (subsample)")
    ax.grid(True, axis="both")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"h04_{model}_{comp.replace('.', '_')}.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run_model(model: str, n_null: int, seed: int, w: ResultsWriter, fig_dir: Path) -> dict:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    comps = ["embed.tok"] + (["unembed.w"] if "unembed.w" in weights else [])
    V, d = weights["embed.tok"].shape
    draws = min(n_null, 5) if model == "qwen3.5-0.8b" else n_null
    counts = unigram_counts(model, V)
    rng = np.random.default_rng(seed)
    pairs = _pairs(rng, V, N_PAIRS)
    headline = {}
    for comp in comps:
        E = weights[comp]
        tr = embed_stats(E, pairs, counts)
        n1, n6 = [], []
        for k in range(draws):
            r1 = np.random.default_rng([seed, 1, k])
            n1.append(embed_stats(nulls.n1_gaussian(E, r1), pairs, counts))
            r6 = np.random.default_rng([seed, 6, k])
            n6.append(embed_stats(nulls.n6_cov_matched(E, r6), pairs, counts))
            print(f"  {model} {comp} null draw {k + 1}/{draws}  ({time.time() - t0:.0f}s)", flush=True)
        base = dict(hypothesis=HYP, model=model, family=fam, layer="", component=comp,
                    n_rows=V, n_cols=d, src_dtype=SRC_DTYPE[model], seed=seed)
        for null_kind, nl in (("N1", n1), ("N6", n6)):
            for metric in SCALARS:
                if metric not in tr:
                    continue
                nv = [x[metric] for x in nl]
                s = null_summary(nv)
                w.row(**base, metric=metric, value=tr[metric], null_kind=null_kind, null_n=s["n"],
                      null_mean=s["mean"], null_sd=s["sd"], z=zscore(tr[metric], nv))
            for k in range(10):
                nv = [x["pc_share"][k] for x in nl]
                s = null_summary(nv)
                w.row(**base, metric=f"pc{k + 1}_share", value=float(tr["pc_share"][k]), null_kind=null_kind,
                      null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(float(tr["pc_share"][k]), nv))
            # distribution statistic: KS of the pairwise-cosine sample vs null, and null-vs-null median
            ks_tr = [ks_2samp(tr["_cos"], x["_cos"])[0] for x in nl]
            ks_nn = [ks_2samp(nl[a]["_cos"], nl[b]["_cos"])[0] for a in range(len(nl)) for b in range(a + 1, len(nl))]
            w.row(**base, metric="ks_cos", value=float(np.median(ks_tr)), null_kind=null_kind, null_n=len(nl),
                  ks=float(np.median(ks_tr)), ks_null_median=float(np.median(ks_nn)) if ks_nn else "")
        w.row(**base, metric="n_unigram_tokens", value=tr["n_unigram"], null_kind="", null_n=0)
        save_raw(HYP, model, comp, pc_share=tr["pc_share"], proj_pc1=tr["_proj_pc1"], proj_mean=tr["_proj_mean"],
                 cos_sample=tr["_cos"][:100000], unigram_counts=counts,
                 n1_mean_cos=[x["mean_cos"] for x in n1], n6_mean_cos=[x["mean_cos"] for x in n6],
                 n1_pc_share=np.array([x["pc_share"] for x in n1]), n6_pc_share=np.array([x["pc_share"] for x in n6]),
                 n6_spearman_pc1_unigram=[x["spearman_pc1_unigram"] for x in n6],
                 n6_spearman_pc1_tokid=[x["spearman_pc1_tokid"] for x in n6])
        figure(model, comp, tr, n1, n6, fig_dir)
        headline[comp] = dict(mean_cos=tr["mean_cos"], z_cos_N1=zscore(tr["mean_cos"], [x["mean_cos"] for x in n1]),
                              pc1=float(tr["pc_share"][0]), mp=tr["mp_edge_share"],
                              rho_id=tr["spearman_pc1_tokid"], z_id_N6=zscore(tr["spearman_pc1_tokid"], [x["spearman_pc1_tokid"] for x in n6]),
                              rho_uni=tr["spearman_pc1_unigram"], z_uni_N6=zscore(tr["spearman_pc1_unigram"], [x["spearman_pc1_unigram"] for x in n6]),
                              rho_mean_uni=tr["spearman_mean_unigram"])
    print(f"{model} done in {time.time() - t0:.0f}s: {headline}", flush=True)
    return headline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--nulls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "h04.csv"))
    args = ap.parse_args()
    fig_dir = ROOT / "results" / "figures" / "h04"
    with ResultsWriter(args.out) as w:
        for m in args.models:
            run_model(m, args.nulls, args.seed, w, fig_dir)


if __name__ == "__main__":
    main()
