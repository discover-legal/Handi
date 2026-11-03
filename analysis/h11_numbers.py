"""H11: number / modular / elliptic-curve structure in token embeddings.

Integer tokens: for GPT-2 BPE (gpt2, tinystories-*) and NeoX BPE (pythia-*) the tokens whose
decoded string is exactly an integer 0..999 without leading zeros, as two sets: "nospace" ("123")
and "space" (" 123"). For Qwen (digits tokenized singly): "digits" = the ten single-digit tokens
(n = 10 limits every test here; the helix basis has 10 columns, so R^2 is ~1 by construction and
only the null comparison is meaningful), and "digitsum" = 0..99 where 10..99 is represented by
the SUM of the two digit embeddings (a stated adaptation, not a token).

Tests (per set, E[n] rows = embeddings of the available n):
  A  helix fit: least squares of E[n] (all d dims) on [1, n, cos(2 pi n/T), sin(2 pi n/T)]
     for T in {2, 5, 10, 100}; total R^2; vs 20 draws of 4 random periods in (3, 200) and vs
     N7 (shuffled n -> embedding assignment, 100 draws). A linear-only [1, n] R^2 is reported too.
  B  DFT along n of the top-10 PCs (on the longest contiguous run 0..m-1 of available n):
     fraction of each PC's energy at periods 10, 100, 2, 5 (explicit sinusoid projection), vs N7.
  C  elliptic curve: project to the top-2 PCs (standardized x, y); least squares (a, b) in
     y^2 = x^3 + a x + b, residual RMS; vs an implicit 2-parameter helix analogue y^2 = a x^2 + b
     (an ellipse about the origin; the same residual functional and parameter count), vs N1
     (Gaussian embeddings of the same shape and norm, 100 draws). N7 leaves the point cloud and
     hence this statistic unchanged (reported as degenerate, z = 0).

Usage: .venv/bin/python analysis/h11_numbers.py [--models ...] [--nulls N] [--seed S] [--out CSV]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis import nulls, plots, registry  # noqa: E402
from handi.analysis.loader import load_canonical  # noqa: E402
from handi.analysis.results import ResultsWriter, save_raw  # noqa: E402
from handi.analysis.stats import ks_2samp, null_summary, zscore  # noqa: E402

HYP = "H11"
DEFAULT_MODELS = ["gpt2", "pythia-70m", "pythia-160m", "qwen3.5-0.8b",
                  "tinystories-1m", "tinystories-3m", "tinystories-8m", "tinystories-33m"]
SRC_DTYPE = {"gpt2": "F32", "pythia-70m": "F16", "pythia-160m": "F16", "qwen3.5-0.8b": "BF16",
             "tinystories-1m": "F32", "tinystories-3m": "F32", "tinystories-8m": "F32", "tinystories-33m": "F32"}
PERIODS = (2.0, 5.0, 10.0, 100.0)
N_RANDOM_PERIODS = 20
N_PCS = 10
INT_RE = re.compile(r"^(0|[1-9][0-9]{0,2})$")


# ----------------------------------------------------------------------------- token sets
def number_sets(model: str, E: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{set name: (n values, embedding rows N x d)}."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(registry.snapshot_dir(model) / "tokenizer.json"))
    V = min(tok.get_vocab_size(), E.shape[0])
    nospace, space = {}, {}
    for i in range(V):
        s = tok.decode([i])
        if INT_RE.match(s):
            nospace[int(s)] = i
        elif s.startswith(" ") and INT_RE.match(s[1:]):
            space[int(s[1:])] = i
    sets = {}
    if model.startswith("qwen"):
        digits = {k: nospace[k] for k in range(10) if k in nospace}
        if len(digits) != 10:
            raise RuntimeError(f"{model}: expected 10 digit tokens, found {sorted(digits)}")
        n = np.arange(10)
        sets["digits"] = (n, E[[digits[k] for k in n]])
        n2 = np.arange(100)
        rows = [E[digits[k]] if k < 10 else E[digits[k // 10]] + E[digits[k % 10]] for k in n2]
        sets["digitsum"] = (n2, np.stack(rows))
    else:
        for name, m in (("nospace", nospace), ("space", space)):
            n = np.array(sorted(m))
            sets[name] = (n, E[[m[k] for k in n]])
    return sets


def contiguous_prefix(n: np.ndarray) -> int:
    m = 0
    while m < n.size and n[m] == m:
        m += 1
    return m


# ----------------------------------------------------------------------------- A: helix
def basis(n: np.ndarray, periods) -> np.ndarray:
    cols = [np.ones_like(n, dtype=np.float64), n.astype(np.float64)]
    for T in periods:
        cols.append(np.cos(2 * np.pi * n / T))
        cols.append(np.sin(2 * np.pi * n / T))
    return np.stack(cols, 1)


def r2_fit(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Total R^2 and per-dim R^2 of least squares Y ~ X."""
    Yc = Y - Y.mean(0)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    R = Y - X @ beta
    ss_res = (R ** 2).sum(0)
    ss_tot = (Yc ** 2).sum(0)
    return float(1 - ss_res.sum() / ss_tot.sum()), 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


# ----------------------------------------------------------------------------- B: DFT
def top_pcs(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    Yc = Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    k = min(k, S.size)
    return Yc @ Vt[:k].T, S[:k] ** 2 / (S ** 2).sum()


def period_power(x: np.ndarray, T: float) -> float:
    """Fraction of the (centered) series energy at period T: |<x, e^{-2 pi i n / T}>|^2 / (N sum x^2)."""
    n = np.arange(x.size)
    x = x - x.mean()
    c = np.sum(x * np.exp(-2j * np.pi * n / T))
    return float(np.abs(c) ** 2 / (x.size * np.sum(x ** 2) + 1e-30))


def dft_stats(Y: np.ndarray) -> dict:
    ts, share = top_pcs(Y, N_PCS)
    P = np.array([[period_power(ts[:, k], T) for T in PERIODS] for k in range(ts.shape[1])])   # pcs x periods
    wmean = (P * share[:, None]).sum(0) / share.sum()
    return {"per_pc": P, "weighted": wmean, "share": share, "ts": ts}


# ----------------------------------------------------------------------------- C: cubic
def cubic_stats(Y: np.ndarray) -> dict:
    ts, _ = top_pcs(Y, 2)
    x, y = ts[:, 0], ts[:, 1]
    x = (x - x.mean()) / (x.std() + 1e-30)
    y = (y - y.mean()) / (y.std() + 1e-30)
    t = y ** 2 - x ** 3
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    r_cubic = t - A @ (a, b)
    A2 = np.stack([x ** 2, np.ones_like(x)], 1)
    (a2, b2), *_ = np.linalg.lstsq(A2, y ** 2, rcond=None)
    r_ell = y ** 2 - A2 @ (a2, b2)
    return {"cubic_rms": float(np.sqrt(np.mean(r_cubic ** 2))), "ellipse_rms": float(np.sqrt(np.mean(r_ell ** 2))),
            "a": float(a), "b": float(b), "x": x, "y": y, "r_cubic": r_cubic, "r_ell": r_ell}


# ----------------------------------------------------------------------------- driver
def run_model(model: str, n_null: int, seed: int, w: ResultsWriter, fig_dir: Path) -> dict:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    E = weights["embed.tok"].astype(np.float64)
    d = E.shape[1]
    head = {}
    for set_name, (n, Y) in number_sets(model, E).items():
        N = n.size
        comp = f"embed.tok/{set_name}"
        base = dict(hypothesis=HYP, model=model, family=fam, layer="", component=comp, n_rows=N, n_cols=d,
                    src_dtype=SRC_DTYPE[model], seed=seed)

        def row(metric, value, null_kind, nv, **kw):
            s = null_summary(nv)
            w.row(**{**base, **kw}, metric=metric, value=value, null_kind=null_kind, null_n=s["n"], null_mean=s["mean"],
                  null_sd=s["sd"], z=zscore(value, nv) if s["n"] >= 2 else "")

        rng = np.random.default_rng([seed, 11])
        idx = np.arange(N)
        n7 = [nulls.n7_assignment_shuffle(Y, np.random.default_rng([seed, 7, k]), idx) for k in range(n_null)]

        # ---- A
        X = basis(n, PERIODS)
        r2, r2_dim = r2_fit(X, Y)
        r2_lin, _ = r2_fit(basis(n, ()), Y)
        r2_rand = [r2_fit(basis(n, rng.uniform(3, 200, size=len(PERIODS))), Y)[0] for _ in range(N_RANDOM_PERIODS)]
        r2_n7 = [r2_fit(X, Yn)[0] for Yn in n7]
        r2_lin_n7 = [r2_fit(basis(n, ()), Yn)[0] for Yn in n7]
        row("r2_helix", r2, "random_periods", r2_rand)
        row("r2_helix", r2, "N7", r2_n7)
        row("r2_linear", r2_lin, "N7", r2_lin_n7)
        row("r2_helix_minus_linear", r2 - r2_lin, "N7", [a - b for a, b in zip(r2_n7, r2_lin_n7)])
        # per-period contribution: R^2 of helix minus R^2 with that period removed
        for T in PERIODS:
            others = tuple(p for p in PERIODS if p != T)
            v = r2 - r2_fit(basis(n, others), Y)[0]
            nv = [r2_fit(X, Yn)[0] - r2_fit(basis(n, others), Yn)[0] for Yn in n7[:min(n_null, 50)]]
            row(f"r2_gain_T{int(T)}", v, "N7", nv)
        # distribution statistic: per-dim R^2 vs N7
        r2_dim_n7 = [r2_fit(X, Yn)[1] for Yn in n7[:20]]
        ks_tr = [ks_2samp(r2_dim[np.isfinite(r2_dim)], q[np.isfinite(q)])[0] for q in r2_dim_n7]
        ks_nn = [ks_2samp(r2_dim_n7[a], r2_dim_n7[b])[0] for a in range(len(r2_dim_n7)) for b in range(a + 1, len(r2_dim_n7))]
        w.row(**base, metric="ks_r2_per_dim", value=float(np.median(ks_tr)), null_kind="N7", null_n=len(r2_dim_n7),
              ks=float(np.median(ks_tr)), ks_null_median=float(np.median(ks_nn)))

        # ---- B (contiguous prefix only)
        m = contiguous_prefix(n)
        dft = None
        if m >= 20:
            Ym = Y[:m]
            dft = dft_stats(Ym)
            dft_n7 = [dft_stats(nulls.n7_assignment_shuffle(Ym, np.random.default_rng([seed, 7, k]), np.arange(m))) for k in range(n_null)]
            for ti, T in enumerate(PERIODS):
                row(f"dft_power_T{int(T)}_weighted", float(dft["weighted"][ti]), "N7", [x["weighted"][ti] for x in dft_n7],
                    n_rows=m)
                for k in range(dft["per_pc"].shape[0]):
                    row(f"dft_power_T{int(T)}_pc{k + 1}", float(dft["per_pc"][k, ti]), "N7", [x["per_pc"][k, ti] for x in dft_n7],
                        n_rows=m)
        else:
            w.row(**base, metric="dft_skipped_contiguous_run_too_short", value=m, null_kind="", null_n=0)

        # ---- C
        cub = cubic_stats(Y)
        n1 = [cubic_stats(nulls.n1_gaussian(Y, np.random.default_rng([seed, 1, k]))) for k in range(n_null)]
        row("cubic_rms", cub["cubic_rms"], "N1", [x["cubic_rms"] for x in n1])
        row("ellipse_rms", cub["ellipse_rms"], "N1", [x["ellipse_rms"] for x in n1])
        row("cubic_minus_ellipse_rms", cub["cubic_rms"] - cub["ellipse_rms"], "N1", [x["cubic_rms"] - x["ellipse_rms"] for x in n1])
        w.row(**base, metric="cubic_rms", value=cub["cubic_rms"], null_kind="N7_degenerate", null_n=n_null,
              null_mean=cub["cubic_rms"], null_sd=0.0, z=0.0)
        ks_tr = [ks_2samp(np.abs(cub["r_cubic"]), np.abs(x["r_cubic"]))[0] for x in n1[:20]]
        ks_nn = [ks_2samp(np.abs(n1[a]["r_cubic"]), np.abs(n1[b]["r_cubic"]))[0] for a in range(20) for b in range(a + 1, 20)]
        w.row(**base, metric="ks_cubic_residual", value=float(np.median(ks_tr)), null_kind="N1", null_n=20,
              ks=float(np.median(ks_tr)), ks_null_median=float(np.median(ks_nn)))

        save_raw(HYP, model, comp, n=n, r2_helix=r2, r2_linear=r2_lin, r2_random_periods=r2_rand, r2_n7=r2_n7,
                 r2_per_dim=r2_dim, pc_xy=np.stack([cub["x"], cub["y"]], 1), cubic_ab=[cub["a"], cub["b"]],
                 cubic_rms=cub["cubic_rms"], ellipse_rms=cub["ellipse_rms"],
                 cubic_rms_n1=[x["cubic_rms"] for x in n1], ellipse_rms_n1=[x["ellipse_rms"] for x in n1],
                 **({"dft_per_pc": dft["per_pc"], "dft_weighted": dft["weighted"], "pc_timeseries": dft["ts"], "pc_share": dft["share"]} if dft else {}))
        figure(model, set_name, n, cub, fig_dir)
        head[set_name] = dict(N=N, r2=round(r2, 4), r2_rand=round(float(np.mean(r2_rand)), 4), r2_n7=round(float(np.mean(r2_n7)), 4),
                              z_rand=round(zscore(r2, r2_rand), 1), z_n7=round(zscore(r2, r2_n7), 1), r2_lin=round(r2_lin, 4),
                              dft=None if dft is None else {int(T): (round(float(dft["weighted"][i]), 4), round(zscore(float(dft["weighted"][i]), [x["weighted"][i] for x in dft_n7]), 1)) for i, T in enumerate(PERIODS)},
                              cubic=round(cub["cubic_rms"], 3), ellipse=round(cub["ellipse_rms"], 3),
                              cubic_n1=round(float(np.mean([x["cubic_rms"] for x in n1])), 3),
                              z_cubic_N1=round(zscore(cub["cubic_rms"], [x["cubic_rms"] for x in n1]), 1))
        print(f"  {model} {set_name}: {head[set_name]} ({time.time() - t0:.0f}s)", flush=True)
    return head


def figure(model: str, set_name: str, n: np.ndarray, cub: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    x, y = cub["x"], cub["y"]
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 7, width_ratios=[2.2] + [1] * 5 + [0.08], hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    sc = ax.scatter(x, y, c=n, cmap="Blues", s=14, linewidths=0.6, edgecolors=plots.SURFACE, vmin=-0.15 * n.max(), vmax=n.max())
    # the fitted cubic y^2 = x^3 + a x + b
    xs = np.linspace(x.min(), x.max(), 400)
    rhs = xs ** 3 + cub["a"] * xs + cub["b"]
    ok = rhs >= 0
    ax.plot(xs[ok], np.sqrt(rhs[ok]), color=pal[7], lw=1.4, label=f"fitted cubic (RMS {cub['cubic_rms']:.2f})")
    ax.plot(xs[ok], -np.sqrt(rhs[ok]), color=pal[7], lw=1.4)
    ax.set_xlabel("PC1 (standardized)")
    ax.set_ylabel("PC2 (standardized)")
    ax.set_title(f"{model} {set_name}: number embeddings\n(top-2 PCs, colour = n, N = {n.size})")
    ax.legend(loc="lower left")
    ax.grid(True, axis="both")
    cax = fig.add_subplot(gs[:, 6])
    fig.colorbar(sc, cax=cax, label="n")
    for r in range(10):
        axr = fig.add_subplot(gs[r // 5, 1 + r % 5])
        m = (n % 10) == r
        axr.scatter(x[~m], y[~m], s=5, color=plots.GRID, linewidths=0)
        axr.scatter(x[m], y[m], s=9, color=pal[0], linewidths=0)
        axr.set_title(f"n mod 10 = {r}", fontsize=9)
        axr.set_xticks([]); axr.set_yticks([]); axr.grid(False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"h11_{model}_{set_name}_pc12.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--nulls", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "h11.csv"))
    args = ap.parse_args()
    fig_dir = ROOT / "results" / "figures" / "h11"
    with ResultsWriter(args.out) as w:
        for m in args.models:
            run_model(m, args.nulls, args.seed, w, fig_dir)


if __name__ == "__main__":
    main()
