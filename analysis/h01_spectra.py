"""H1: heavy-tailed spectra with spikes above the MP edge (docs/hypotheses.md section 4, H1).

For every 2-D canonical weight matrix of every model: ESD, KS to MP (sigma^2 from the Frobenius
norm), spike count (sigma^2 from the bulk median, iterated), CSN power-law alpha, stable rank.
The same on N1 (Gaussian), N2 (element shuffle) and N4 (fresh init) nulls. One row per
(model, layer, component, metric, null_kind) plus layer="ALL" aggregates (median over layers,
null distribution = per-draw median over layers). Raw singular values go to results/raw/h01/.

Run:  .venv/bin/python analysis/h01_spectra.py [--models ...] [--nulls N] [--seed S] [--out CSV]
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from handi.analysis import nulls, plots, registry, results, spectra, stats  # noqa: E402
from handi.analysis.loader import load_canonical, n_layers  # noqa: E402

HYP = "H01"
METRICS = ("ks_mp", "spikes", "alpha", "stable_rank")
NULL_KINDS = ("N1", "N2", "N4")
BIG_ELEMENTS = 100_000_000   # above this (Qwen 248k x 1024 embedding) draw at most 5 nulls
BIG_NULLS = 5
MAX_NULL_PAIRS = 200         # null-vs-null KS pairs sampled per (matrix, null kind)
RAW_NULL_KEEP = 5            # N1 draws whose singular values are stored in results/raw (for H8)


def split_name(name: str) -> tuple[str, str]:
    """'layers.3.attn.q.w' -> ('3', 'attn.q.w'); 'embed.tok' -> ('', 'embed.tok')."""
    if name.startswith("layers."):
        _, i, rest = name.split(".", 2)
        return i, rest
    return "", name


def matrices(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """2-D canonical weights only (no vectors, no conv kernels)."""
    return {k: v for k, v in weights.items() if v.ndim == 2}


def metrics_of(W: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """(metrics, singular values, eigenvalues of W^T W / N)."""
    s = spectra.singular_values(W)
    e = spectra.esd(W, s)
    gamma = spectra.aspect(W)
    m, n = W.shape
    sigma2_frob = float(np.sum(s**2) / (m * n))
    out = {
        "ks_mp": spectra.ks_to_mp(e, gamma, sigma2_frob),
        "spikes": float(spectra.spike_count(e, gamma)),
        "alpha": float(spectra.powerlaw_fit(e)["alpha"]),
        "stable_rank": spectra.stable_rank(W, s),
    }
    return out, s, e


def norm_eigs(e: np.ndarray) -> np.ndarray:
    """Eigenvalues divided by their mean, so trained-vs-N4 ESD comparison is scale-free."""
    return e / e.mean()


def ks_block(e_tr: np.ndarray, e_null: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    """(median KS(trained, null_i), median KS(null_i, null_j)) on mean-normalized eigenvalues."""
    t = norm_eigs(e_tr)
    ks_tn = [stats.ks_2samp(t, norm_eigs(x))[0] for x in e_null]
    n = len(e_null)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > MAX_NULL_PAIRS:
        idx = rng.choice(len(pairs), MAX_NULL_PAIRS, replace=False)
        pairs = [pairs[i] for i in idx]
    ks_nn = [stats.ks_2samp(norm_eigs(e_null[i]), norm_eigs(e_null[j]))[0] for i, j in pairs]
    return C.finite_median(ks_tn), (C.finite_median(ks_nn) if ks_nn else float("nan"))


def run_model(model: str, args, writer: results.ResultsWriter) -> None:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    mats = matrices(weights)
    n_null = C.n_nulls(args, model)
    seed = C.model_seed(args.seed, model)
    rng = np.random.default_rng(seed)
    dtype = C.SRC_DTYPE[model]
    print(f"[{model}] {len(mats)} matrices, {n_null} nulls, family {fam}", flush=True)

    trained: dict[str, dict] = {}
    null_vals: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    null_eigs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    draws_of = {k: (BIG_NULLS if v.size > BIG_ELEMENTS else n_null) for k, v in mats.items()}

    # trained + N1 + N2 (matrix-level nulls)
    for name, W in mats.items():
        m, s, e = metrics_of(W)
        trained[name] = {"metrics": m, "s": s, "e": e}
        raw_n1 = []
        for kind, gen in (("N1", nulls.n1_gaussian), ("N2", nulls.n2_shuffle)):
            for i in range(draws_of[name]):
                Z = gen(W, rng)
                mz, sz, ez = metrics_of(Z)
                for k, v in mz.items():
                    null_vals[name][kind][k].append(v)
                null_eigs[name][kind].append(ez)
                if kind == "N1" and i < RAW_NULL_KEEP:
                    raw_n1.append(sz)
                del Z
        results.save_raw(HYP.lower(), model, name, s_trained=s, s_n1=np.stack(raw_n1), gamma=spectra.aspect(W),
                         shape=np.array(W.shape), eig_trained=e)
        print(f"  {name:32s} {str(W.shape):16s} ks_mp={m['ks_mp']:.3f} spikes={m['spikes']:.0f} "
              f"alpha={m['alpha']:.2f} srank={m['stable_rank']:.1f}  [{time.time() - t0:.0f}s]", flush=True)

    # N4: fresh init of the whole architecture, one state dict per draw
    for i in range(n_null):
        fresh = nulls.n4_fresh_init(fam, cfg, np.random.default_rng([seed, 4, i]))
        for name in mats:
            if name not in fresh or i >= draws_of[name]:
                continue
            mz, _, ez = metrics_of(fresh[name])
            for k, v in mz.items():
                null_vals[name]["N4"][k].append(v)
            null_eigs[name]["N4"].append(ez)
        del fresh
    print(f"  N4 done [{time.time() - t0:.0f}s]", flush=True)

    # rows per matrix
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"trained": defaultdict(list), "null": defaultdict(lambda: defaultdict(list)),
                                                           "ks": defaultdict(list), "ksn": defaultdict(list), "shape": None})
    for name, W in mats.items():
        layer, comp = split_name(name)
        tr = trained[name]
        for kind in NULL_KINDS:
            if not null_eigs[name][kind]:
                continue
            ks_tn, ks_nn = ks_block(tr["e"], null_eigs[name][kind], rng)
            a = agg[(comp, kind)]
            a["ks"][kind].append(ks_tn)
            a["ksn"][kind].append(ks_nn)
            a["shape"] = W.shape
            for met in METRICS:
                nv = np.asarray(null_vals[name][kind][met], dtype=np.float64)
                sm = stats.null_summary(nv)
                v = tr["metrics"][met]
                writer.row(hypothesis=HYP, model=model, family=fam, layer=layer, component=comp, metric=met,
                           value=v, null_kind=kind, null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"],
                           z=stats.zscore(v, nv), ks=ks_tn, ks_null_median=ks_nn,
                           n_rows=W.shape[0], n_cols=W.shape[1], src_dtype=dtype, seed=seed)
                a["trained"][met].append(v)
                a["null"][met][kind].append(nv)

    # aggregate rows: layer = ALL, median over layers; null = per-draw median over layers
    for (comp, kind), a in agg.items():
        for met in METRICS:
            tv = np.asarray(a["trained"][met])
            draws = a["null"][met][kind]
            n_min = min(len(x) for x in draws)
            block = np.stack([x[:n_min] for x in draws])      # [layers, draws]
            per_draw = np.nanmedian(block, axis=0)
            v = C.finite_median(tv)
            sm = stats.null_summary(per_draw)
            writer.row(hypothesis=HYP, model=model, family=fam, layer="ALL", component=comp, metric=met,
                       value=v, null_kind=kind, null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"],
                       z=stats.zscore(v, per_draw), ks=C.finite_median(a["ks"][kind]),
                       ks_null_median=C.finite_median(a["ksn"][kind]),
                       n_rows=a["shape"][0], n_cols=a["shape"][1], src_dtype=dtype, seed=seed)

    if not args.no_figures:
        figures(model, fam, cfg, mats, trained, null_eigs)
    print(f"[{model}] done in {time.time() - t0:.0f}s", flush=True)


def representative(fam: str, cfg: dict, mats: dict) -> list[str]:
    L = n_layers(cfg, fam)
    mid = L // 2
    names = ["embed.tok"]
    if fam == "qwen3_5":
        attn = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
        lin = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
        names += [f"layers.{attn[len(attn) // 2]}.attn.q.w", f"layers.{lin[len(lin) // 2]}.linear.in_qkv.w"]
    else:
        names += [f"layers.{mid}.attn.q.w"]
    names += [f"layers.{mid}.mlp.up.w", f"layers.{mid}.mlp.down.w"]
    return [n for n in names if n in mats]


def figures(model, fam, cfg, mats, trained, null_eigs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    out_dir = C.FIGURES / "h01"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in representative(fam, cfg, mats):
        W = mats[name]
        e = trained[name]["e"]
        gamma = spectra.aspect(W)
        m, n = W.shape
        sigma2 = float(np.sum(e) * max(m, n) / (m * n))   # sum s^2 / (m n)
        lo, hi = spectra.mp_edges(gamma, sigma2)
        nulls1 = null_eigs[name]["N1"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        # left: ESD histogram around the bulk with MP overlay and N1 band
        xmax = 1.6 * hi
        bins = np.linspace(0, xmax, 60)
        h_tr, _ = np.histogram(e, bins=bins, density=True)
        h_null = np.stack([np.histogram(x, bins=bins, density=True)[0] for x in nulls1])
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax1.fill_between(centers, h_null.min(0), h_null.max(0), color=plots.INK_SECONDARY, alpha=0.18, lw=0,
                         label=f"N1 band (n={len(nulls1)})", step="mid")
        ax1.bar(centers, h_tr, width=(bins[1] - bins[0]) * 0.9, color=pal[0], alpha=0.85, label="trained ESD")
        xs = np.linspace(lo, hi, 400)
        ax1.plot(xs, spectra.mp_pdf(xs, gamma, sigma2), color=pal[1], lw=2, label="MP (Frobenius sigma^2)")
        n_out = int(np.sum(e > xmax))
        ax1.set_xlabel("eigenvalue of W^T W / N")
        ax1.set_ylabel("density")
        ax1.set_title("ESD near the bulk" + (f" ({n_out} eigenvalues beyond the axis)" if n_out else ""))
        fig.suptitle(f"{model}  {name}  {m} x {n}", x=0.01, ha="left", fontsize=11)
        ax1.legend(loc="upper right")
        # right: rank plot, log-log, trained vs N1 band
        ranks = np.arange(1, e.size + 1)
        e_desc = e[::-1]
        null_desc = np.stack([x[::-1] for x in nulls1])
        ax2.fill_between(ranks, null_desc.min(0), null_desc.max(0), color=plots.INK_SECONDARY, alpha=0.18, lw=0,
                         label="N1 band")
        ax2.plot(ranks, e_desc, color=pal[0], lw=2, label="trained")
        ax2.axhline(hi, color=pal[1], lw=1.2, label="MP upper edge")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("rank i")
        ax2.set_ylabel("eigenvalue")
        ax2.set_title("rank plot (spikes above the MP edge)")
        ax2.legend(loc="lower left")
        for ax in (ax1, ax2):
            ax.grid(True, axis="y")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / f"{model}_esd_{name}.png")
        plt.close(fig)


def main() -> None:
    args = C.parser(HYP).parse_args()
    out = C.prepare_out(args)
    with results.ResultsWriter(out) as writer:
        for model in args.models:
            run_model(model, args, writer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
