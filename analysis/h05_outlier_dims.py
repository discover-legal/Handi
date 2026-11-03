"""H5: outlier residual dimensions, consistent across layers.

Per layer and per matrix: a profile over the d residual dims
  * readers (attn.q/k/v.w, mlp.up.w, mlp.gate.w; Qwen DeltaNet layers also linear.in_qkv/in_z.w):
    column norms (norm over the output axis, one value per residual dim);
  * writers (attn.o.w, mlp.down.w; Qwen linear.out.w): row norms;
  * norm gain ln1.w / ln2.w (Qwen: 1 + w) and bias ln1.b / ln2.b where present.
Statistics: per-profile excess kurtosis; top-5 dims per layer per matrix type; the number of
dims that are top-5 in >= L/2 layers, against a simulated null of random top-5 sets
(10^4 simulations; the hypergeometric reference). Nulls: N1 for norm profiles (20 draws per
matrix), N4 fresh init for gains/biases (constant under every family's init rule, so their
kurtosis is undefined: reported as such).

Usage: .venv/bin/python analysis/h05_outlier_dims.py [--models ...] [--nulls N] [--seed S] [--out CSV]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis import nulls, plots  # noqa: E402
from handi.analysis.loader import load_canonical, n_layers  # noqa: E402
from handi.analysis.results import ResultsWriter, save_raw  # noqa: E402
from handi.analysis.stats import null_summary, zscore  # noqa: E402

HYP = "H05"
DEFAULT_MODELS = ["gpt2", "pythia-70m", "pythia-160m", "qwen3.5-0.8b",
                  "tinystories-1m", "tinystories-3m", "tinystories-8m", "tinystories-33m"]
SRC_DTYPE = {"gpt2": "F32", "pythia-70m": "F16", "pythia-160m": "F16", "qwen3.5-0.8b": "BF16",
             "tinystories-1m": "F32", "tinystories-3m": "F32", "tinystories-8m": "F32", "tinystories-33m": "F32"}
READERS = ["attn.q.w", "attn.k.w", "attn.v.w", "mlp.up.w", "mlp.gate.w", "linear.in_qkv.w", "linear.in_z.w"]
WRITERS = ["attn.o.w", "mlp.down.w", "linear.out.w"]
GAINS = ["ln1.w", "ln2.w"]
BIASES = ["ln1.b", "ln2.b"]
TOPK = 5
N_SIM = 10_000


def kurt(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.std() == 0:
        return float("nan")
    return float(sps.kurtosis(x, fisher=True, bias=False))


def topk(x: np.ndarray, k: int = TOPK) -> np.ndarray:
    return np.sort(np.argsort(-np.asarray(x))[:k])


def shared_count(sets: list[np.ndarray], d: int, thresh: int) -> tuple[int, np.ndarray]:
    counts = np.zeros(d, dtype=int)
    for s in sets:
        counts[s] += 1
    return int(np.sum(counts >= thresh)), counts


def shared_null(L: int, d: int, thresh: int, rng: np.random.Generator, k: int = TOPK, n_sim: int = N_SIM) -> np.ndarray:
    """Random top-k sets per layer: count of dims appearing in >= thresh of L layers (hypergeometric reference)."""
    out = np.empty(n_sim, dtype=int)
    for s in range(n_sim):
        c = np.zeros(d, dtype=int)
        for _ in range(L):
            c[rng.choice(d, size=k, replace=False)] += 1
        out[s] = np.sum(c >= thresh)
    return out


def profile(W: np.ndarray, kind: str) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    return np.linalg.norm(W, axis=0) if kind == "reader" else np.linalg.norm(W, axis=1)


def run_model(model: str, n_null: int, seed: int, w: ResultsWriter, fig_dir: Path) -> dict:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    L = n_layers(cfg, fam)
    d = weights["embed.tok"].shape[1]
    thresh = int(np.ceil(L / 2))
    base = dict(hypothesis=HYP, model=model, family=fam, src_dtype=SRC_DTYPE[model], seed=seed)
    rng_sim = np.random.default_rng([seed, 99])
    # N4 fresh init for gains/biases
    fresh = nulls.n4_fresh_init(fam, cfg, np.random.default_rng([seed, 4]))

    profiles: dict[str, dict[int, np.ndarray]] = {}       # type -> layer -> profile over d dims
    kurt_tr: dict[str, dict[int, float]] = {}
    kurt_null: dict[str, dict[int, list[float]]] = {}
    top_sets: dict[str, dict[int, np.ndarray]] = {}
    top_null: dict[str, dict[int, list[np.ndarray]]] = {}

    for layer in range(L):
        for name in READERS + WRITERS:
            key = f"layers.{layer}.{name}"
            if key not in weights:
                continue
            kind = "reader" if name in READERS else "writer"
            W = weights[key]
            p = profile(W, kind)
            profiles.setdefault(name, {})[layer] = p
            kurt_tr.setdefault(name, {})[layer] = kurt(p)
            top_sets.setdefault(name, {})[layer] = topk(p)
            kn, tn = [], []
            for k in range(n_null):
                r = np.random.default_rng([seed, 1, layer, hash(name) % 1000, k])
                pn = profile(nulls.n1_gaussian(W, r), kind)
                kn.append(kurt(pn)); tn.append(topk(pn))
            kurt_null.setdefault(name, {})[layer] = kn
            top_null.setdefault(name, {})[layer] = tn
            s = null_summary(kn)
            w.row(**base, layer=layer, component=name, metric="kurtosis", value=kurt_tr[name][layer], null_kind="N1",
                  null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(kurt_tr[name][layer], kn),
                  n_rows=W.shape[0], n_cols=W.shape[1])
        for name in GAINS + BIASES:
            key = f"layers.{layer}.{name}"
            if key not in weights:
                continue
            g = weights[key].astype(np.float64)
            gf = fresh[key].astype(np.float64)
            if fam == "qwen3_5" and name in GAINS:
                g = 1.0 + g
                gf = 1.0 + gf
            profiles.setdefault(name, {})[layer] = g
            kurt_tr.setdefault(name, {})[layer] = kurt(g)
            top_sets.setdefault(name, {})[layer] = topk(np.abs(g - np.median(g)))
            kv = kurt(gf)
            w.row(**base, layer=layer, component=name, metric="kurtosis", value=kurt_tr[name][layer], null_kind="N4",
                  null_n=1, null_mean=kv, null_sd=0.0, z=float("nan") if not np.isfinite(kv) else zscore(kurt_tr[name][layer], [kv, kv]),
                  n_rows=d, n_cols=1)
            w.row(**base, layer=layer, component=name, metric="max_abs_dev_over_sd", value=float(np.max(np.abs(g - np.median(g))) / (g.std() + 1e-30)),
                  null_kind="N4", null_n=1, null_mean=float("nan"), null_sd=0.0, n_rows=d, n_cols=1)
        print(f"  {model} layer {layer + 1}/{L} ({time.time() - t0:.0f}s)", flush=True)

    # layer-aggregated kurtosis per type (mean over layers) vs N1 (mean over layers of each draw)
    for name in kurt_tr:
        layers = sorted(kurt_tr[name])
        v = float(np.nanmean([kurt_tr[name][l] for l in layers]))
        if name in kurt_null:
            nd = min(len(kurt_null[name][l]) for l in layers)
            nv = [float(np.nanmean([kurt_null[name][l][k] for l in layers])) for k in range(nd)]
            s = null_summary(nv)
            w.row(**base, layer="all", component=name, metric="kurtosis_mean_layers", value=v, null_kind="N1",
                  null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(v, nv), n_rows=len(layers), n_cols=d)
        else:
            w.row(**base, layer="all", component=name, metric="kurtosis_mean_layers", value=v, null_kind="N4",
                  null_n=1, null_mean=float("nan"), null_sd=0.0, n_rows=len(layers), n_cols=d)

    # cross-layer shared top-5 dims per type, and for the mean z-scored reader / writer profile
    zprof: dict[str, np.ndarray] = {}
    for name, per_layer in profiles.items():
        layers = sorted(per_layer)
        M = np.stack([per_layer[l] for l in layers])                      # L' x d
        zprof[name] = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-30)
    agg = {}
    for label, names in (("reader_mean", READERS), ("writer_mean", WRITERS), ("gain_mean", GAINS)):
        present = [n for n in names if n in zprof]
        if not present:
            continue
        # per layer: mean over types present in that layer (Qwen alternates attention / DeltaNet)
        per_layer = {}
        for n in present:
            for i, l in enumerate(sorted(profiles[n])):
                per_layer.setdefault(l, []).append(zprof[n][i] if label != "gain_mean" else np.abs(zprof[n][i]))
        agg[label] = {l: np.mean(np.stack(v), 0) for l, v in per_layer.items()}
        top_sets[label] = {l: topk(p) for l, p in agg[label].items()}

    outlier_dims = {}
    sim_cache = {}
    for name, sets in top_sets.items():
        layers = sorted(sets)
        Ln = len(layers)
        th = int(np.ceil(Ln / 2))
        n_shared, counts = shared_count([sets[l] for l in layers], d, th)
        if (Ln, th) not in sim_cache:
            sim_cache[(Ln, th)] = shared_null(Ln, d, th, rng_sim)
        sim = sim_cache[(Ln, th)]
        s = null_summary(sim)
        w.row(**base, layer="all", component=name, metric="n_dims_top5_in_half_layers", value=n_shared, null_kind="sim_hypergeom",
              null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(n_shared, sim), n_rows=Ln, n_cols=d)
        # same count on the N1 draws of this matrix type (where available)
        if name in top_null:
            nd = min(len(top_null[name][l]) for l in layers)
            nv = [shared_count([top_null[name][l][k] for l in layers], d, th)[0] for k in range(nd)]
            s = null_summary(nv)
            w.row(**base, layer="all", component=name, metric="n_dims_top5_in_half_layers", value=n_shared, null_kind="N1",
                  null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(n_shared, nv), n_rows=Ln, n_cols=d)
        outlier_dims[name] = np.flatnonzero(counts >= th)
        # the most-shared dims, for the record: top-10 by count
        order = np.argsort(-counts, kind="stable")[:10]
        w.row(**base, layer="all", component=name, metric="top_shared_dims", value=" ".join(f"{i}:{counts[i]}" for i in order if counts[i] > 0),
              null_kind="", null_n=0, n_rows=Ln, n_cols=d)

    raw = {f"top5__{n}": np.stack([s[l] for l in sorted(s)]) for n, s in top_sets.items()}
    raw.update({f"profile__{n}": np.stack([p[l] for l in sorted(p)]) for n, p in profiles.items()})
    raw.update({f"zprofile__{n}": np.stack([p[l] for l in sorted(p)]) for n, p in agg.items()})
    raw.update({f"outliers__{n}": v for n, v in outlier_dims.items()})
    raw.update({f"kurtosis__{n}": np.array([kurt_tr[n][l] for l in sorted(kurt_tr[n])]) for n in kurt_tr})
    save_raw(HYP, model, "outlier_dims", **raw)
    figure(model, agg.get("reader_mean"), agg.get("writer_mean"), outlier_dims, fig_dir)
    head = {k: outlier_dims[k].tolist() for k in ("reader_mean", "writer_mean", "gain_mean") if k in outlier_dims}
    print(f"{model} done in {time.time() - t0:.0f}s: outlier dims {head}", flush=True)
    return head


def figure(model: str, reader: dict | None, writer: dict | None, outlier_dims: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plots.style()
    # diverging blue <-> red with neutral gray midpoint (dataviz palette)
    cmap = LinearSegmentedColormap.from_list("div", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1a1a"])
    panels = [(k, v) for k, v in (("reader column norms", reader), ("writer row norms", writer)) if v is not None]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.4 * len(panels) + 1), squeeze=False)
    fig.subplots_adjust(hspace=0.55)
    for ax, (label, per_layer) in zip(axes[:, 0], panels):
        layers = sorted(per_layer)
        M = np.stack([per_layer[l] for l in layers])
        order = np.argsort(-M.mean(0))
        Ms = M[:, order]
        vmax = float(np.nanmax(np.abs(Ms)))
        im = ax.imshow(Ms, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax), interpolation="nearest")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers])
        ax.set_ylabel("layer")
        ax.set_xticks([])
        ax.set_xlabel("residual dim, sorted by mean z over layers (left = highest); top 20: "
                      + " ".join(str(order[i]) for i in range(20)), fontsize=8)
        ax.grid(False)
        key = "reader_mean" if label.startswith("reader") else "writer_mean"
        od = outlier_dims.get(key, np.array([], dtype=int))
        ax.set_title(f"{model}: per-dim z-scored {label}, mean over matrix types; top-5 in >= L/2 layers: {od.tolist()}")
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("z within layer")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"h05_{model}_zheatmap.png")
    plt.close(fig)


def figures_from_raw(model: str, fig_dir: Path) -> None:
    """Redraw the heatmap from results/raw/H05/<model>/outlier_dims.npz (no recomputation)."""
    from handi.analysis.results import RAW_DIR

    z = np.load(RAW_DIR / HYP / model / "outlier_dims.npz")
    agg = {k[len("zprofile__"):]: {i: r for i, r in enumerate(z[k])} for k in z.files if k.startswith("zprofile__")}
    outl = {k[len("outliers__"):]: z[k] for k in z.files if k.startswith("outliers__")}
    figure(model, agg.get("reader_mean"), agg.get("writer_mean"), outl, fig_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--nulls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "h05.csv"))
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from results/raw, no computation")
    args = ap.parse_args()
    fig_dir = ROOT / "results" / "figures" / "h05"
    if args.figures_only:
        for m in args.models:
            figures_from_raw(m, fig_dir)
        return
    with ResultsWriter(args.out) as w:
        for m in args.models:
            run_model(m, args.nulls, args.seed, w, fig_dir)


if __name__ == "__main__":
    main()
