"""H12: MLP neuron geometry (hypotheses.md H12): superposition signature and in/out alignment.

Per MLP layer, W_in = mlp.up.w [I, d] (and mlp.gate.w for SwiGLU/Qwen), W_out[i] = mlp.down.w[:, i].
  pairwise: cosine of 10^6 random row pairs of W_in -> excess kurtosis, frac |cos| > 3/sqrt(d),
            frac |cos| > 5/sqrt(d).   Nulls: N1 (Gaussian, same shape) and N5 (spectrum-matched).
  in/out:   per-neuron cos(W_in[i], W_out[i]) -> mean, frac |cos| > 3/sqrt(d), skew, excess kurtosis.
            Null: N1 for both factors.
Rows per layer (component "mlp.up.w" / "mlp.gate.w" for pairwise; "mlp.up_down" / "mlp.gate_down" for
in/out) and layer="ALL" aggregates (median over layers, null = per-draw median over layers). The ks
columns compare the trained cosine sample with the null draws' cosine samples (median KS trained vs
draw, median KS draw vs draw). Figures: cosine histograms vs null for the middle layer.

Run:  .venv/bin/python analysis/h12_mlp_geometry.py [--models ...] [--nulls N] [--seed S] [--out CSV]
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from handi.analysis import nulls, plots, results, stats  # noqa: E402
from handi.analysis.loader import load_canonical, n_layers  # noqa: E402

HYP = "H12"
N_PAIRS = 1_000_000
PAIR_METRICS = ("kurtosis", "frac_gt3", "frac_gt5")
IO_METRICS = ("mean", "frac_gt3", "skew", "kurtosis")
MAX_NULL_PAIRS = 100
KS_SUBSAMPLE = 100_000    # cosines used in the two-sample KS (10^6 x 10^6 would be slow)


def pair_cosines(W: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray, dev) -> np.ndarray:
    T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).to(dev)
    T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (T[idx_i] * T[idx_j]).sum(dim=1).cpu().numpy().astype(np.float64)


def inout_cosines(W_in: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """cos(W_in[i], W_out[:, i]) per neuron i."""
    a = W_in.astype(np.float64)
    b = W_out.T.astype(np.float64)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-30)


def pair_stats(c: np.ndarray, d: int) -> dict[str, float]:
    return {
        "kurtosis": float(sps.kurtosis(c, fisher=True)),
        "frac_gt3": float(np.mean(np.abs(c) > 3 / np.sqrt(d))),
        "frac_gt5": float(np.mean(np.abs(c) > 5 / np.sqrt(d))),
    }


def io_stats(c: np.ndarray, d: int) -> dict[str, float]:
    return {
        "mean": float(c.mean()),
        "frac_gt3": float(np.mean(np.abs(c) > 3 / np.sqrt(d))),
        "skew": float(sps.skew(c)),
        "kurtosis": float(sps.kurtosis(c, fisher=True)),
    }


def ks_cols(c_tr, c_null, rng) -> tuple[float, float]:
    sub = lambda x: x if x.size <= KS_SUBSAMPLE else rng.choice(x, KS_SUBSAMPLE, replace=False)
    t = sub(c_tr)
    ns = [sub(x) for x in c_null]
    ks_tn = [stats.ks_2samp(t, x)[0] for x in ns]
    pairs = [(i, j) for i in range(len(ns)) for j in range(i + 1, len(ns))]
    if len(pairs) > MAX_NULL_PAIRS:
        pairs = [pairs[k] for k in rng.choice(len(pairs), MAX_NULL_PAIRS, replace=False)]
    ks_nn = [stats.ks_2samp(ns[i], ns[j])[0] for i, j in pairs]
    return C.finite_median(ks_tn), C.finite_median(ks_nn) if ks_nn else float("nan")


def run_model(model: str, args, writer: results.ResultsWriter) -> None:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    L = n_layers(cfg, fam)
    n_null = C.n_nulls(args, model)
    seed = C.model_seed(args.seed, model)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = C.SRC_DTYPE[model]
    in_names = ["up"] + (["gate"] if "layers.0.mlp.gate.w" in weights else [])
    print(f"[{model}] {L} MLP layers, W_in = {in_names}, {n_null} nulls", flush=True)

    # agg[(component, null_kind, metric)] -> trained per layer, null [layer][draw]
    agg = defaultdict(lambda: {"tr": [], "nl": [], "ks": [], "ksn": [], "shape": None})
    hist_keep: dict = {}
    mid = L // 2
    for layer in range(L):
        W_out = weights[f"layers.{layer}.mlp.down.w"]
        for nm in in_names:
            W_in = weights[f"layers.{layer}.mlp.{nm}.w"]
            I, d = W_in.shape
            idx_i = torch.from_numpy(rng.integers(0, I, N_PAIRS)).to(dev)
            idx_j = torch.from_numpy(rng.integers(0, I, N_PAIRS)).to(dev)
            same = idx_i == idx_j
            idx_j = torch.where(same, (idx_j + 1) % I, idx_j)
            # pairwise cosines, trained
            c_tr = pair_cosines(W_in, idx_i, idx_j, dev)
            m_tr = pair_stats(c_tr, d)
            comp = f"mlp.{nm}.w"
            sigma = None
            for kind in ("N1", "N5"):
                null_metrics = defaultdict(list)
                null_cos = []
                for k in range(n_null):
                    if kind == "N1":
                        Z = nulls.n1_gaussian(W_in, rng)
                    else:
                        if sigma is None:
                            sigma = torch.linalg.svdvals(torch.from_numpy(W_in).to(dev)).double().cpu().numpy()
                        Z = nulls.n5_spectrum_matched(W_in, rng, sigma=sigma)
                    c = pair_cosines(Z, idx_i, idx_j, dev)
                    for mk, v in pair_stats(c, d).items():
                        null_metrics[mk].append(v)
                    null_cos.append(c)
                ks_tn, ks_nn = ks_cols(c_tr, null_cos, rng)
                if layer == mid:
                    hist_keep[(comp, kind)] = (c_tr, null_cos[0], d)
                for mk in PAIR_METRICS:
                    nv = np.asarray(null_metrics[mk])
                    sm = stats.null_summary(nv)
                    writer.row(hypothesis=HYP, model=model, family=fam, layer=layer, component=comp, metric=mk,
                               value=m_tr[mk], null_kind=kind, null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"],
                               z=stats.zscore(m_tr[mk], nv), ks=ks_tn, ks_null_median=ks_nn,
                               n_rows=I, n_cols=d, src_dtype=dtype, seed=seed)
                    a = agg[(comp, kind, mk)]
                    a["tr"].append(m_tr[mk]); a["nl"].append(nv); a["ks"].append(ks_tn); a["ksn"].append(ks_nn); a["shape"] = (I, d)
            # in/out cosine, trained vs N1 on both factors
            c_io = inout_cosines(W_in, W_out)
            m_io = io_stats(c_io, d)
            comp_io = f"mlp.{nm}_down"
            null_metrics = defaultdict(list)
            null_cos = []
            for k in range(n_null):
                c = inout_cosines(nulls.n1_gaussian(W_in, rng), nulls.n1_gaussian(W_out, rng))
                for mk, v in io_stats(c, d).items():
                    null_metrics[mk].append(v)
                null_cos.append(c)
            ks_tn, ks_nn = ks_cols(c_io, null_cos, rng)
            if layer == mid:
                hist_keep[(comp_io, "N1")] = (c_io, np.concatenate(null_cos), d)
            for mk in IO_METRICS:
                nv = np.asarray(null_metrics[mk])
                sm = stats.null_summary(nv)
                writer.row(hypothesis=HYP, model=model, family=fam, layer=layer, component=comp_io, metric=mk,
                           value=m_io[mk], null_kind="N1", null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"],
                           z=stats.zscore(m_io[mk], nv), ks=ks_tn, ks_null_median=ks_nn,
                           n_rows=I, n_cols=d, src_dtype=dtype, seed=seed)
                a = agg[(comp_io, "N1", mk)]
                a["tr"].append(m_io[mk]); a["nl"].append(nv); a["ks"].append(ks_tn); a["ksn"].append(ks_nn); a["shape"] = (I, d)
            print(f"  layer {layer} {nm}: pair kurt={m_tr['kurtosis']:.2f} frac>3={m_tr['frac_gt3']:.4f} "
                  f"| in/out mean={m_io['mean']:.4f} frac>3={m_io['frac_gt3']:.3f} [{time.time() - t0:.0f}s]", flush=True)

    for (comp, kind, mk), a in agg.items():
        block = np.stack(a["nl"])                 # [layers, draws]
        per_draw = np.nanmedian(block, axis=0)
        v = C.finite_median(a["tr"])
        sm = stats.null_summary(per_draw)
        writer.row(hypothesis=HYP, model=model, family=fam, layer="ALL", component=comp, metric=mk,
                   value=v, null_kind=kind, null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"],
                   z=stats.zscore(v, per_draw), ks=C.finite_median(a["ks"]), ks_null_median=C.finite_median(a["ksn"]),
                   n_rows=a["shape"][0], n_cols=a["shape"][1], src_dtype=dtype, seed=seed)
        print(f"  ALL {comp:14s} {kind} {mk:9s} median={v:.4f} null={sm['mean']:.4f}+-{sm['sd']:.4f} z={stats.zscore(v, per_draw):.1f}",
              flush=True)
    if not args.no_figures:
        figure(model, mid, hist_keep)
    print(f"[{model}] done in {time.time() - t0:.0f}s", flush=True)


def figure(model: str, mid: int, keep: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    out_dir = C.FIGURES / "h12"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [k for k in keep if k[0].endswith(".w")] + [k for k in keep if not k[0].endswith(".w")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), squeeze=False)
    for ax, key in zip(axes[0], panels):
        comp, kind = key
        c_tr, c_nl, d = keep[key]
        lim = max(np.percentile(np.abs(c_tr), 99.9), 6 / np.sqrt(d))
        bins = np.linspace(-lim, lim, 81)
        ax.hist(c_nl, bins=bins, density=True, color=plots.INK_SECONDARY, alpha=0.25, label=f"{kind} null")
        ax.hist(c_tr, bins=bins, density=True, color=pal[0], alpha=0.85, rwidth=0.9, label="trained")
        for s in (-3, 3):
            ax.axvline(s / np.sqrt(d), color=pal[1], lw=1, label="+-3/sqrt(d)" if s == 3 else None)
        ax.set_yscale("log")
        ax.set_xlabel("cosine" if comp.endswith(".w") else "cos(W_in[i], W_out[i])")
        ax.set_ylabel("density (log)")
        ax.set_title(f"{comp} vs {kind}" + ("  (row pairs)" if comp.endswith(".w") else "  (per neuron)"))
        ax.legend(loc="upper right")
    fig.suptitle(f"{model}: MLP layer {mid} cosine distributions", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / f"{model}_mlp_cosines.png")
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
